"""Geographic/cross-entity and data-quality stress tests for predictive models."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import pandas as pd

from .benchmark_suite import BenchmarkSuiteResult, run_benchmark_suite


@dataclass(frozen=True)
class EntityHoldout:
    train_entities: tuple[str, ...]
    test_entities: tuple[str, ...]
    fraction: float
    salt: str


@dataclass(frozen=True)
class FeatureDegradation:
    """Audit record for a deterministic source/feature missingness stress."""

    columns: tuple[str, ...]
    fraction: float
    salt: str
    rows: int
    degraded_cells: int


@dataclass(frozen=True)
class ModelDegradationImpact:
    """OOS metric change caused by feature/source degradation."""

    model: str
    clean_brier_skill: float | None
    degraded_brier_skill: float | None
    brier_skill_delta: float | None
    clean_brier: float | None
    degraded_brier: float | None
    brier_delta: float | None


@dataclass(frozen=True)
class FeatureDegradationStressResult:
    audit: FeatureDegradation
    clean: BenchmarkSuiteResult
    degraded: BenchmarkSuiteResult
    impacts: tuple[ModelDegradationImpact, ...]


def deterministic_feature_degradation(
    features: pd.DataFrame,
    *,
    columns: Iterable[str],
    fraction: float = 0.20,
    salt: str = "preact-source-degradation-v1",
) -> tuple[pd.DataFrame, FeatureDegradation]:
    """Mask selected feature cells reproducibly without consulting outcomes."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    requested = tuple(dict.fromkeys(str(c) for c in columns))
    if not requested:
        raise ValueError("at least one feature column is required")
    missing = tuple(c for c in requested if c not in features.columns)
    if missing:
        raise KeyError(f"unknown feature columns: {missing}")
    if not features.index.is_unique:
        raise ValueError("features index must be unique for auditable degradation")

    degraded = features.copy()
    cells = 0
    threshold = int(fraction * (2**64))
    for column in requested:
        mask = []
        for key in features.index:
            digest = sha256(f"{salt}|{repr(key)}|{column}".encode("utf-8")).digest()
            score = int.from_bytes(digest[:8], "big")
            mask.append(score < threshold)
        if mask:
            cells += int(sum(mask))
            degraded.loc[mask, column] = float("nan")

    return degraded, FeatureDegradation(
        columns=requested,
        fraction=float(fraction),
        salt=salt,
        rows=len(features),
        degraded_cells=cells,
    )


def run_feature_degradation_stress(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    columns: Iterable[str],
    horizon_days: int,
    fraction: float = 0.20,
    salt: str = "preact-source-degradation-v1",
    min_train_dates: int = 20,
    calibration_dates: int = 5,
    test_dates_per_fold: int = 5,
    bootstrap_samples: int = 1000,
    random_state: int = 42,
) -> FeatureDegradationStressResult:
    """Measure OOS skill loss under deterministic source/feature missingness.

    Clean and degraded suites use identical targets, temporal protocol and random
    state. The mask is target-blind, so the resulting deltas quantify robustness
    without selecting outages from outcomes or model errors.
    """

    degraded_features, audit = deterministic_feature_degradation(
        features, columns=columns, fraction=fraction, salt=salt
    )
    kwargs = dict(
        horizon_days=horizon_days,
        min_train_dates=min_train_dates,
        calibration_dates=calibration_dates,
        test_dates_per_fold=test_dates_per_fold,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    clean = run_benchmark_suite(features, target, **kwargs)
    degraded = run_benchmark_suite(degraded_features, target, **kwargs)
    if clean.folds != degraded.folds:
        raise ValueError("degradation stress changed temporal folds")
    if tuple(clean.models) != tuple(degraded.models):
        raise ValueError("degradation stress changed benchmark model set")

    impacts = []
    for name in clean.models:
        clean_metrics = clean.models[name].metrics
        degraded_metrics = degraded.models[name].metrics
        skill_delta = None
        if clean_metrics.brier_skill is not None and degraded_metrics.brier_skill is not None:
            skill_delta = float(degraded_metrics.brier_skill - clean_metrics.brier_skill)
        brier_delta = None
        if clean_metrics.brier is not None and degraded_metrics.brier is not None:
            brier_delta = float(degraded_metrics.brier - clean_metrics.brier)
        impacts.append(
            ModelDegradationImpact(
                model=name,
                clean_brier_skill=clean_metrics.brier_skill,
                degraded_brier_skill=degraded_metrics.brier_skill,
                brier_skill_delta=skill_delta,
                clean_brier=clean_metrics.brier,
                degraded_brier=degraded_metrics.brier,
                brier_delta=brier_delta,
            )
        )
    return FeatureDegradationStressResult(audit, clean, degraded, tuple(impacts))


def deterministic_entity_holdout(
    entity_ids: Iterable[str],
    *,
    fraction: float = 0.20,
    salt: str = "preact-v1",
) -> EntityHoldout:
    """Assign entities to a stable holdout using a salted SHA-256 score."""

    if not 0.05 <= fraction <= 0.50:
        raise ValueError("fraction must be between 0.05 and 0.50")
    entities = tuple(sorted(set(str(x) for x in entity_ids)))
    if len(entities) < 4:
        raise ValueError("at least four entities are required")

    scored = []
    for entity in entities:
        digest = sha256(f"{salt}|{entity}".encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], "big") / float(2**64)
        scored.append((entity, score))

    test = tuple(entity for entity, score in scored if score < fraction)
    train = tuple(entity for entity, score in scored if score >= fraction)
    if not test:
        n_test = max(1, round(len(entities) * fraction))
        ordered = tuple(entity for entity, _ in sorted(scored, key=lambda x: x[1]))
        test = ordered[:n_test]
        train = tuple(entity for entity in entities if entity not in set(test))
    if not train:
        raise ValueError("holdout leaves no training entities")
    return EntityHoldout(train, test, float(fraction), salt)


def run_unseen_entity_stress(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    horizon_days: int,
    holdout_fraction: float = 0.20,
    holdout_salt: str = "preact-v1",
    min_train_dates: int = 20,
    calibration_dates: int = 5,
    test_dates_per_fold: int = 5,
    bootstrap_samples: int = 1000,
    random_state: int = 42,
) -> tuple[EntityHoldout, BenchmarkSuiteResult]:
    if not isinstance(features.index, pd.MultiIndex) or "entity_id" not in features.index.names:
        raise TypeError("features must have entity_id in a MultiIndex")
    entities = tuple(sorted(set(features.index.get_level_values("entity_id").astype(str))))
    split = deterministic_entity_holdout(entities, fraction=holdout_fraction, salt=holdout_salt)
    result = run_benchmark_suite(
        features,
        target,
        horizon_days=horizon_days,
        min_train_dates=min_train_dates,
        calibration_dates=calibration_dates,
        test_dates_per_fold=test_dates_per_fold,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
        fit_entity_ids=set(split.train_entities),
        test_entity_ids=set(split.test_entities),
    )
    return split, result
