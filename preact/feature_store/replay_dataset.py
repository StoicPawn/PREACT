"""Build point-in-time replay datasets from the historical warehouse."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

from preact.history.warehouse import HistoricalWarehouse
from .targets import binary_event_target
from .temporal import entity_feature_frame


@dataclass(frozen=True)
class ReplayDataset:
    features: pd.DataFrame
    target: pd.Series
    entity_id: str
    feature_variables: tuple[str, ...]
    target_variable: str
    horizon_days: int


def build_replay_dataset(
    warehouse: HistoricalWarehouse,
    *,
    entity_id: str,
    cutoffs: Iterable[datetime],
    feature_variables: Iterable[str],
    target_variable: str,
    horizon_days: int,
) -> ReplayDataset:
    ordered_cutoffs = tuple(sorted(cutoffs))
    variables = tuple(feature_variables)
    features = entity_feature_frame(
        warehouse,
        entity_id=entity_id,
        cutoffs=ordered_cutoffs,
        variables=variables,
    )
    target = binary_event_target(
        warehouse,
        entity_id=entity_id,
        cutoffs=ordered_cutoffs,
        target_variable=target_variable,
        horizon_days=horizon_days,
    )
    target = target.reindex(features.index)
    return ReplayDataset(
        features=features,
        target=target,
        entity_id=entity_id,
        feature_variables=variables,
        target_variable=target_variable,
        horizon_days=horizon_days,
    )
