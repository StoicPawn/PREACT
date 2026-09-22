"""Negative-control tests for predictive research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .benchmark_suite import BenchmarkSuiteResult, run_benchmark_suite


@dataclass(frozen=True)
class PlaceboResult:
    target: pd.Series
    benchmark: BenchmarkSuiteResult
    max_brier_skill: float | None


def within_date_permutation(
    target: pd.Series,
    *,
    seed: int = 42,
) -> pd.Series:
    """Permute labels across entities within each date, preserving date event rates."""

    if not isinstance(target.index, pd.MultiIndex) or "date" not in target.index.names:
        raise TypeError("target must have a MultiIndex containing date")
    rng = np.random.default_rng(seed)
    result = target.copy()
    date_level = target.index.names.index("date")
    dates = target.index.get_level_values("date")
    for date in pd.Index(dates.unique()).sort_values():
        positions = np.flatnonzero(dates == date)
        values = target.iloc[positions].to_numpy(copy=True)
        rng.shuffle(values)
        result.iloc[positions] = values
    return result.astype(int)


def run_placebo_benchmark(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    horizon_days: int,
    min_train_dates: int = 20,
    calibration_dates: int = 5,
    test_dates_per_fold: int = 5,
    bootstrap_samples: int = 250,
    seed: int = 42,
) -> PlaceboResult:
    placebo = within_date_permutation(target, seed=seed)
    benchmark = run_benchmark_suite(
        features,
        placebo,
        horizon_days=horizon_days,
        min_train_dates=min_train_dates,
        calibration_dates=calibration_dates,
        test_dates_per_fold=test_dates_per_fold,
        bootstrap_samples=bootstrap_samples,
        random_state=seed,
    )
    skills = [
        model.metrics.brier_skill
        for model in benchmark.models.values()
        if model.metrics.brier_skill is not None
    ]
    return PlaceboResult(
        target=placebo,
        benchmark=benchmark,
        max_brier_skill=float(max(skills)) if skills else None,
    )
