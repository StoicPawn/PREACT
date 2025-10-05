"""Policy core module implementing tax and transfer logic for the MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TaxBracket:
    """Represent a tax bracket with an upper income threshold."""

    threshold: float
    rate: float

    def __post_init__(self) -> None:  # pragma: no cover - dataclass validation
        if self.threshold <= 0:
            raise ValueError("Threshold must be positive")
        if not 0 <= self.rate <= 1:
            raise ValueError("Rate must be expressed as a share between 0 and 1")


@dataclass(frozen=True)
class PolicyParameters:
    """Container holding tax and transfer policy knobs."""

    tax_brackets: Iterable[TaxBracket]
    base_deduction: float
    child_subsidy: float
    unemployment_benefit: float = 900.0

    def as_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame describing tax brackets for reporting purposes."""

        brackets = sorted(self.tax_brackets, key=lambda item: item.threshold)
        data = {"threshold": [b.threshold for b in brackets], "rate": [b.rate for b in brackets]}
        return pd.DataFrame(data)


class PolicyCore:
    """Apply policy parameters to compute net income, tax and transfers."""

    REQUIRED_COLUMNS = {"gross_income", "employment_status"}

    def __init__(self, parameters: PolicyParameters):
        self.parameters = parameters
        self._brackets = sorted(parameters.tax_brackets, key=lambda b: b.threshold)

    def _compute_tax(self, taxable_income: np.ndarray) -> np.ndarray:
        """Compute tax liabilities for a taxable income vector."""

        tax_due = np.zeros_like(taxable_income)
        lower_bound = 0.0
        for bracket in self._brackets:
            upper = bracket.threshold
            mask = taxable_income > lower_bound
            taxable_segment = np.clip(taxable_income - lower_bound, 0, upper - lower_bound)
            tax_due = tax_due + mask * taxable_segment * bracket.rate
            lower_bound = upper
        # Handle open-ended bracket if the highest threshold < inf
        if self._brackets:
            top_threshold = self._brackets[-1].threshold
            top_rate = self._brackets[-1].rate
            mask_top = taxable_income > top_threshold
            tax_due = tax_due + mask_top * (taxable_income - top_threshold) * top_rate
        return tax_due

    def apply(self, population: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with fiscal metrics for the provided population."""

        missing = self.REQUIRED_COLUMNS - set(population.columns)
        if missing:
            raise KeyError(f"Population frame missing required columns: {sorted(missing)}")

        frame = population.copy()
        num_children = frame.get("num_children", pd.Series(0, index=frame.index))
        household_deduction = self.parameters.base_deduction
        taxable_income = np.clip(frame["gross_income"] - household_deduction, 0, None)
        tax_due = self._compute_tax(taxable_income.to_numpy())
        subsidies = num_children.to_numpy() * self.parameters.child_subsidy
        unemployment_mask = frame["employment_status"].str.lower().eq("unemployed")
        unemployment_transfers = unemployment_mask.to_numpy() * self.parameters.unemployment_benefit
        total_transfers = subsidies + unemployment_transfers
        net_income = frame["gross_income"].to_numpy() - tax_due + total_transfers

        result = frame.assign(
            tax_liability=tax_due,
            transfers=total_transfers,
            net_income=net_income,
            disposable_income=np.maximum(net_income, 0.0),
        )
        return result
