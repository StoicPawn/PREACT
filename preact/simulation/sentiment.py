"""Sentiment core computing socio-economic mood indicators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .economy import EconomyState


@dataclass(frozen=True)
class SentimentWeights:
    """Weights used to aggregate socio-economic signals into a score."""

    income: float = 0.45
    employment: float = 0.35
    inflation: float = 0.2
    baseline: float = 55.0


class SentimentCore:
    """Compute sentiment indexes based on income, employment and inflation."""

    def __init__(self, weights: SentimentWeights | None = None):
        self.weights = weights or SentimentWeights()

    def compute(
        self,
        disposable_income: pd.Series,
        state: Optional[EconomyState] = None,
        previous_state: Optional[EconomyState] = None,
        baseline_income: Optional[float] = None,
    ) -> float:
        """Return a 0-100 sentiment score."""

        income_mean = float(disposable_income.mean()) if len(disposable_income) else 0.0
        if baseline_income is None:
            baseline_income = income_mean
        income_change = 0.0 if baseline_income == 0 else (income_mean - baseline_income) / baseline_income

        employment_rate = state.employment_rate if state else 0.5
        employment_change = 0.0
        if state and previous_state:
            employment_change = employment_rate - previous_state.employment_rate

        inflation = state.cpi if state else 100.0
        inflation_change = 0.0
        if state and previous_state and previous_state.cpi:
            inflation_change = (inflation - previous_state.cpi) / previous_state.cpi
        elif not state:
            inflation_change = 0.0
        else:
            inflation_change = (inflation - 100.0) / 100.0

        weights = self.weights
        score = (
            weights.baseline
            + 100 * (weights.income * income_change)
            + 100 * (weights.employment * employment_change)
            - 100 * (weights.inflation * inflation_change)
        )
        return float(np.clip(score, 0.0, 100.0))

    def by_decile(self, disposable_income: pd.Series) -> pd.Series:
        """Compute sentiment per income decile using dispersion-aware adjustments."""

        if disposable_income.empty:
            return pd.Series(dtype=float)
        deciles = pd.qcut(disposable_income, 10, labels=False, duplicates="drop")
        baseline = disposable_income.mean()
        sentiments = []
        for decile in sorted(deciles.dropna().unique()):
            mask = deciles == decile
            score = self.compute(disposable_income[mask], state=None, baseline_income=baseline)
            sentiments.append((decile, score))
        if not sentiments:
            return pd.Series(dtype=float)
        return pd.Series({decile: score for decile, score in sentiments}, name="sentiment_decile")
