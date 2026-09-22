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


def deterministic_feature_degradation(
    features: pd.DataFrame,
    *,
    columns: Iterable[str],
    fraction: float = 0.20,
    salt: str = "preact-source-degradation-v1",
) -> tuple[pd.DataFrame, FeatureDegradation]:
    """Mask selected feature cells reproducibly without consulting outcomes.

    Selection is a pure function of the row identity, feature name and salt.  It
    therefore cannot accidentally condition a stress scenario on the target or
    on model errors.  The original frame is never mutated.  This primitive is
    intended for source-outage/missingness sensitivity runs on the same OOS
    protocol as the primary benchmark.
    """

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
            identity = repr(key)
            digest = sha256(f"{salt}|{identity}|{column}".encode("utf-8")).digest()
            score = int.from_bytes(digest[:8], "big")
            mask.append(score < threshold)
        if mask:
            cells += int(sum(mask))
            degraded.loc[mask, column] = float("nan")

    audit = FeatureDegradation(
        columns=requested,
        fraction=float(fraction),
        salt=salt,
        rows=len(features),
        degraded_cells=cells,
    )
    return degraded, audit


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
    split = deterministic_entity_holdout(
        entities,
        fraction=holdout_fraction,
        salt=holdout_salt,
    )
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
