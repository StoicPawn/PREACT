"""Economy core implementing simplified macro rules for the MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EconomyParameters:
    """Economy level parameters controlling behavioural rules."""

    baseline_consumption: float
    propensity_low_income: float = 0.85
    propensity_high_income: float = 0.65
    job_finding_rate: float = 0.08
    job_separation_rate: float = 0.03
    labour_demand_sensitivity: float = 0.4
    price_sensitivity: float = 0.15
    wage_growth: float = 0.01
    unemployment_income: float = 600.0
    base_wage: float = 1800.0
    initial_cpi: float = 100.0


@dataclass(frozen=True)
class Shock:
    """Represent an exogenous economic shock."""

    name: str
    intensity: float
    start_tick: int = 0
    end_tick: int | None = None

    def is_active(self, tick: int) -> bool:
        if tick < self.start_tick:
            return False
        if self.end_tick is None:
            return True
        return tick <= self.end_tick


@dataclass
class EconomyState:
    """State variables tracked by the simulation engine."""

    cpi: float
    unemployment_rate: float
    employment_rate: float
    labour_demand_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "cpi": self.cpi,
            "unemployment_rate": self.unemployment_rate,
            "employment_rate": self.employment_rate,
            "labour_demand_ratio": self.labour_demand_ratio,
        }


class EconomyCore:
    """Encapsulate rules for employment, consumption and price dynamics."""

    def __init__(self, parameters: EconomyParameters):
        self.parameters = parameters

    def compute_consumption(self, fiscal_frame: pd.DataFrame) -> pd.Series:
        """Derive consumption from disposable income with heterogeneous MPC."""

        disposable = fiscal_frame["disposable_income"].to_numpy()
        median_income = float(np.median(disposable)) if len(disposable) else 0.0
        prop_low = self.parameters.propensity_low_income
        prop_high = self.parameters.propensity_high_income
        propensity = np.where(disposable <= median_income, prop_low, prop_high)
        consumption = disposable * propensity
        return pd.Series(consumption, index=fiscal_frame.index, name="consumption")

    def update_labour_market(
        self,
        population: pd.DataFrame,
        consumption: pd.Series,
        firms: pd.DataFrame,
        tick: int,
        shock: Optional[Shock] = None,
    ) -> pd.DataFrame:
        """Update employment statuses deterministically based on demand signals."""

        frame = population.copy()
        employed_mask = frame["employment_status"].str.lower().eq("employed")
        unemployed_mask = ~employed_mask
        employed = frame.loc[employed_mask]
        unemployed = frame.loc[unemployed_mask]

        demand_ratio = consumption.sum() / max(self.parameters.baseline_consumption, 1.0)
        demand_ratio = float(np.clip(demand_ratio, 0.3, 1.7))

        job_finding = self.parameters.job_finding_rate + self.parameters.labour_demand_sensitivity * (demand_ratio - 1.0)
        job_separation = self.parameters.job_separation_rate - 0.5 * self.parameters.labour_demand_sensitivity * (demand_ratio - 1.0)

        if shock and shock.is_active(tick):
            if "energy" in shock.name.lower():
                job_finding *= 1 - 0.7 * shock.intensity
                job_separation *= 1 + 0.5 * shock.intensity
            else:  # demand shock
                job_finding *= 1 - 0.5 * shock.intensity
                job_separation *= 1 + 0.3 * shock.intensity

        job_finding = float(np.clip(job_finding, 0.0, 0.8))
        job_separation = float(np.clip(job_separation, 0.0, 0.5))

        hires = int(round(len(unemployed) * job_finding))
        separations = int(round(len(employed) * job_separation))

        capacity = int(firms.get("employment_capacity", pd.Series(dtype=float)).sum())
        if capacity:
            projected = len(employed) - separations + hires
            if projected > capacity:
                hires = max(0, hires - (projected - capacity))

        if hires > 0 and not unemployed.empty:
            to_hire = unemployed.sort_values("gross_income", ascending=False).head(hires).index
            frame.loc[to_hire, "employment_status"] = "employed"
            frame.loc[to_hire, "gross_income"] = np.maximum(
                self.parameters.base_wage,
                frame.loc[to_hire, "gross_income"],
            ) * (1 + self.parameters.wage_growth)

        if separations > 0 and not employed.empty:
            to_fire = employed.sort_values("gross_income").head(separations).index
            frame.loc[to_fire, "employment_status"] = "unemployed"
            frame.loc[to_fire, "gross_income"] = self.parameters.unemployment_income

        return frame

    def update_state(
        self,
        population: pd.DataFrame,
        consumption: pd.Series,
        previous_state: EconomyState,
        tick: int,
        shock: Optional[Shock] = None,
    ) -> EconomyState:
        """Compute macro aggregates given the latest micro outcomes."""

        demand_ratio = consumption.sum() / max(self.parameters.baseline_consumption, 1.0)
        demand_ratio = float(np.clip(demand_ratio, 0.3, 2.0))
        price_delta = self.parameters.price_sensitivity * (demand_ratio - 1.0)

        if shock and shock.is_active(tick):
            if "energy" in shock.name.lower():
                price_delta += 0.4 * shock.intensity
            else:
                price_delta -= 0.2 * shock.intensity

        new_cpi = max(40.0, previous_state.cpi * (1 + price_delta))

        employed_mask = population["employment_status"].str.lower().eq("employed")
        employment_rate = float(employed_mask.mean()) if len(population) else 0.0
        unemployment_rate = float(1.0 - employment_rate)

        return EconomyState(
            cpi=new_cpi,
            unemployment_rate=unemployment_rate,
            employment_rate=employment_rate,
            labour_demand_ratio=demand_ratio,
        )
