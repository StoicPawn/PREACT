"""Feature-family ablations for PREACT predictive research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .benchmark_suite import BenchmarkSuiteResult, run_benchmark_suite


@dataclass(frozen=True)
class AblationResult:
    name: str
    removed_columns: tuple[str, ...]
    remaining_columns: tuple[str, ...]
    benchmark: BenchmarkSuiteResult


DEFAULT_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "graph": ("graph_", "graph:", "history:"),
    "world_context": ("world_context:",),
    "news_context": ("news_context:",),
    "capabilities": ("cow_nmc:",),
    "macro": (
        "world_bank:",
        "maddison:",
        "un_wpp:",
        "sipri:",
    ),
    "humanitarian": ("unhcr:",),
}


def columns_in_family(
    columns,
    prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in columns
        if any(str(column).startswith(prefix) for prefix in prefixes)
    )


def run_family_ablation(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    family_name: str,
    prefixes: tuple[str, ...],
    horizon_days: int,
    min_train_dates: int = 20,
    calibration_dates: int = 5,
    test_dates_per_fold: int = 5,
    bootstrap_samples: int = 250,
    random_state: int = 42,
) -> AblationResult:
    removed = columns_in_family(features.columns, prefixes)
    remaining = tuple(str(c) for c in features.columns if str(c) not in set(removed))
    if not remaining:
        raise ValueError(f"ablation {family_name} removes every feature")
    reduced = features.loc[:, list(remaining)]
    benchmark = run_benchmark_suite(
        reduced,
        target,
        horizon_days=horizon_days,
        min_train_dates=min_train_dates,
        calibration_dates=calibration_dates,
        test_dates_per_fold=test_dates_per_fold,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    return AblationResult(
        name=family_name,
        removed_columns=removed,
        remaining_columns=remaining,
        benchmark=benchmark,
    )
