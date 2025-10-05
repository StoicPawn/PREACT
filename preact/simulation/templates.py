"""Scenario templates reflecting the MVP blueprint from ``building_map.md``."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from .economy import EconomyParameters
from .policy import PolicyParameters, TaxBracket
from .scenario import FirmParameters, PopulationParameters, ScenarioBuilder
from .engine import SimulationConfig


@dataclass(frozen=True)
class ScenarioTemplate:
    """Reusable bundle of parameters used to instantiate a scenario builder."""

    name: str
    description: str
    population: PopulationParameters
    firms: FirmParameters
    economy: EconomyParameters
    policy: PolicyParameters
    horizon: int = 12

    def builder(self, *, horizon: int | None = None, seed: int = 42) -> ScenarioBuilder:
        """Return a :class:`ScenarioBuilder` configured for this template."""

        config = SimulationConfig(horizon=horizon or self.horizon)
        return ScenarioBuilder(
            population_params=self.population,
            firm_params=self.firms,
            economy_params=self.economy,
            policy_params=self.policy,
            simulation_config=config,
            seed=seed,
        )

    def to_summary(self) -> Dict[str, object]:
        """Return a serialisable description of the template for API exposure."""

        return {
            "name": self.name,
            "description": self.description,
            "defaults": {
                "population_size": self.population.size,
                "income_mean": self.population.income_mean,
                "employment_rate": self.population.employment_rate,
                "firm_count": self.firms.size,
                "horizon": self.horizon,
                "policy": self._policy_summary(),
            },
        }

    def _policy_summary(self) -> Dict[str, object]:
        brackets = [
            {"threshold": bracket.threshold, "rate": bracket.rate}
            for bracket in sorted(self.policy.tax_brackets, key=lambda item: item.threshold)
        ]
        return {
            "tax_brackets": brackets,
            "base_deduction": self.policy.base_deduction,
            "child_subsidy": self.policy.child_subsidy,
            "unemployment_benefit": self.policy.unemployment_benefit,
        }


def _baseline_economy(population: PopulationParameters) -> EconomyParameters:
    baseline_consumption = population.size * population.income_mean * 0.7
    return EconomyParameters(baseline_consumption=baseline_consumption)


def default_templates() -> Mapping[str, ScenarioTemplate]:
    """Return the canonical set of templates shipped with the MVP."""

    medium_city_population = PopulationParameters(
        size=20000,
        income_mean=28000.0,
        income_sigma=0.6,
        employment_rate=0.9,
        sector_shares={"services": 0.65, "industry": 0.35},
        household_size_mean=2.6,
        child_share=0.48,
    )
    medium_city_firms = FirmParameters(
        size=750,
        sector_shares={"services": 0.7, "industry": 0.3},
        productivity_mean=1.15,
        productivity_sigma=0.25,
        employment_capacity_mean=35,
    )
    medium_city_policy = PolicyParameters(
        tax_brackets=(
            TaxBracket(threshold=30000, rate=0.18),
            TaxBracket(threshold=75000, rate=0.27),
        ),
        base_deduction=5000.0,
        child_subsidy=150.0,
        unemployment_benefit=900.0,
    )

    templates: Dict[str, ScenarioTemplate] = {
        "medium_city": ScenarioTemplate(
            name="Medium City",
            description="Città sintetica di medie dimensioni con economia diversificata",
            population=medium_city_population,
            firms=medium_city_firms,
            economy=_baseline_economy(medium_city_population),
            policy=medium_city_policy,
            horizon=12,
        ),
    }
    return templates
