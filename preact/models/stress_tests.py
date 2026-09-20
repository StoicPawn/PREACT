"""Geographic/cross-entity stress tests for predictive models."""

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

    # Small samples can occasionally produce an empty side; use deterministic
    # ordering rather than random mutation so the split remains reproducible.
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
