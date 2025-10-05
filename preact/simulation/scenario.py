"""Scenario builder for PREACT MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from .economy import EconomyParameters, Shock
from .policy import PolicyParameters
from .events import EventTimeline


@dataclass(frozen=True)
class PopulationParameters:
    """Parameters describing the synthetic households."""

    size: int
    income_mean: float
    income_sigma: float
    employment_rate: float
    sector_shares: Mapping[str, float]
    household_size_mean: float = 2.4
    child_share: float = 0.45


@dataclass(frozen=True)
class FirmParameters:
    """Parameters describing the synthetic firms."""

    size: int
    sector_shares: Mapping[str, float]
    productivity_mean: float
    productivity_sigma: float
    employment_capacity_mean: float


@dataclass
class Scenario:
    """Concrete scenario used by the simulation engine."""

    name: str
    policy: PolicyParameters
    economy: EconomyParameters
    population: pd.DataFrame
    firms: pd.DataFrame
    simulation_config: "SimulationConfig"
    shock: Optional[Shock] = None
    metadata: Dict[str, object] | None = None
    events: EventTimeline | None = None

    def with_policy(self, policy: PolicyParameters, name: Optional[str] = None) -> "Scenario":
        """Return a clone of the scenario with a different policy."""

        return Scenario(
            name=name or self.name,
            policy=policy,
            economy=self.economy,
            population=self.population.copy(deep=True),
            firms=self.firms.copy(deep=True),
            simulation_config=self.simulation_config,
            shock=self.shock,
            metadata=dict(self.metadata or {}),
            events=self.events,
        )


class ScenarioBuilder:
    """Build reproducible scenarios matching the MVP blueprint."""

    def __init__(
        self,
        population_params: PopulationParameters,
        firm_params: FirmParameters,
        economy_params: EconomyParameters,
        policy_params: PolicyParameters,
        simulation_config: "SimulationConfig",
        *,
        shock: Optional[Shock] = None,
        events: EventTimeline | None = None,
        seed: int = 42,
    ) -> None:
        self.population_params = population_params
        self.firm_params = firm_params
        self.economy_params = economy_params
        self.policy_params = policy_params
        self.simulation_config = simulation_config
        self.shock = shock
        self.events = events
        self.rng = np.random.default_rng(seed)
        self._population_template = self._create_population()
        self._firms_template = self._create_firms()

    def _create_population(self) -> pd.DataFrame:
        params = self.population_params
        rng = self.rng
        mu = np.log(params.income_mean) - 0.5 * params.income_sigma ** 2
        incomes = rng.lognormal(mean=mu, sigma=params.income_sigma, size=params.size)
        household_size = rng.poisson(lam=params.household_size_mean, size=params.size) + 1
        has_children = rng.uniform(size=params.size) < params.child_share
        num_children = rng.binomial(n=np.maximum(household_size - 1, 0), p=0.6, size=params.size)
        num_children = np.where(has_children, num_children, 0)
        employment_status = np.where(
            rng.uniform(size=params.size) < params.employment_rate,
            "employed",
            "unemployed",
        )
        sectors = list(params.sector_shares.keys())
        weights = np.array(list(params.sector_shares.values()), dtype=float)
        weights = weights / weights.sum()
        assigned_sector = rng.choice(sectors, p=weights, size=params.size)
        propensity = np.where(incomes < np.median(incomes), 0.85, 0.65)
        frame = pd.DataFrame(
            {
                "agent_id": np.arange(params.size),
                "gross_income": incomes,
                "household_size": household_size,
                "num_children": num_children,
                "employment_status": employment_status,
                "sector": assigned_sector,
                "propensity_to_consume": propensity,
            }
        )
        return frame

    def _create_firms(self) -> pd.DataFrame:
        params = self.firm_params
        rng = self.rng
        sectors = list(params.sector_shares.keys())
        weights = np.array(list(params.sector_shares.values()), dtype=float)
        weights = weights / weights.sum()
        sector_assign = rng.choice(sectors, p=weights, size=params.size)
        productivity = rng.normal(loc=params.productivity_mean, scale=params.productivity_sigma, size=params.size)
        expected_demand = rng.lognormal(mean=np.log(params.productivity_mean), sigma=0.4, size=params.size)
        employment_capacity = np.maximum(
            1,
            rng.normal(loc=params.employment_capacity_mean, scale=params.employment_capacity_mean * 0.3, size=params.size),
        ).astype(int)
        return pd.DataFrame(
            {
                "firm_id": np.arange(params.size),
                "sector": sector_assign,
                "productivity": productivity,
                "expected_demand": expected_demand,
                "employment_capacity": employment_capacity,
            }
        )

    def build(
        self,
        name: str = "Base",
        *,
        policy: Optional[PolicyParameters] = None,
        shock: Optional[Shock] = None,
        events: Optional[EventTimeline] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Scenario:
        """Return a scenario ready to be consumed by the simulation engine."""

        population = self._population_template.copy(deep=True)
        firms = self._firms_template.copy(deep=True)
        return Scenario(
            name=name,
            policy=policy or self.policy_params,
            economy=self.economy_params,
            population=population,
            firms=firms,
            simulation_config=self.simulation_config,
            shock=shock or self.shock,
            metadata=metadata or {},
            events=events or self.events,
        )

    def refreshed(self) -> "ScenarioBuilder":
        """Return a new builder with fresh draws keeping parameters constant."""

        new_seed = int(self.rng.integers(0, 1_000_000))
        return ScenarioBuilder(
            population_params=self.population_params,
            firm_params=self.firm_params,
            economy_params=self.economy_params,
            policy_params=self.policy_params,
            simulation_config=self.simulation_config,
            shock=self.shock,
            events=self.events,
            seed=new_seed,
        )


if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .engine import SimulationConfig
