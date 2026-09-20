"""Multi-entity point-in-time panels for country-level event targets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

from preact.history.graph_store import HistoricalGraphStore
from preact.history.schema import KnowledgeMode
from preact.history.warehouse import HistoricalWarehouse
from .event_history import event_history_features
from .event_variable_history import event_variable_history_features
from .graph import graph_feature_snapshot
from .targets import binary_event_target
from .temporal import entity_feature_snapshot
from .temporal_dynamics import entity_temporal_dynamics_snapshot


@dataclass(frozen=True)
class EventPanelDataset:
    features: pd.DataFrame
    target: pd.Series
    entities: tuple[str, ...]
    horizon_days: int
    target_variable: str


def build_event_risk_panel(
    *,
    warehouse: HistoricalWarehouse,
    graph: HistoricalGraphStore,
    entity_ids: Iterable[str],
    cutoffs: Iterable[datetime],
    feature_variables: Iterable[str],
    target_variable: str,
    horizon_days: int,
    graph_recent_days: int = 365,
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
    include_relation_history: bool = True,
    include_temporal_dynamics: bool = True,
    include_target_history: bool = True,
) -> EventPanelDataset:
    """Build a panel for outcomes such as coup attempts or successful coups."""

    entities = tuple(sorted(set(str(x) for x in entity_ids)))
    dates = tuple(sorted(set(cutoffs)))
    variables = tuple(dict.fromkeys(str(v) for v in feature_variables))
    feature_rows: list[dict[str, object]] = []
    target_values: dict[tuple[pd.Timestamp, str], int] = {}

    for entity_id in entities:
        target = binary_event_target(
            warehouse,
            entity_id=entity_id,
            cutoffs=dates,
            target_variable=target_variable,
            horizon_days=horizon_days,
        )
        for cutoff in dates:
            row: dict[str, object] = {
                "date": pd.Timestamp(cutoff),
                "entity_id": entity_id,
            }
            row.update(
                entity_feature_snapshot(
                    warehouse,
                    entity_id=entity_id,
                    cutoff=cutoff,
                    variables=variables,
                    knowledge_mode=knowledge_mode,
                )
            )
            if include_temporal_dynamics:
                row.update(
                    entity_temporal_dynamics_snapshot(
                        warehouse,
                        entity_id=entity_id,
                        cutoff=cutoff,
                        variables=variables,
                        knowledge_mode=knowledge_mode,
                    )
                )
            row.update(
                graph_feature_snapshot(
                    graph,
                    entity_id=entity_id,
                    cutoff=cutoff,
                    recent_days=graph_recent_days,
                    knowledge_mode=knowledge_mode,
                )
            )
            if include_relation_history:
                row.update(
                    event_history_features(
                        graph,
                        entity_id=entity_id,
                        cutoff=cutoff,
                        knowledge_mode=knowledge_mode,
                    )
                )
            if include_target_history:
                row.update(
                    event_variable_history_features(
                        warehouse,
                        entity_id=entity_id,
                        cutoff=cutoff,
                        variables=[target_variable],
                        knowledge_mode=knowledge_mode,
                    )
                )
            feature_rows.append(row)
            target_values[(pd.Timestamp(cutoff), entity_id)] = int(
                target.loc[pd.Timestamp(cutoff)]
            )

    if not feature_rows:
        return EventPanelDataset(
            pd.DataFrame(),
            pd.Series(dtype=int),
            entities,
            horizon_days,
            target_variable,
        )
    frame = (
        pd.DataFrame(feature_rows)
        .set_index(["date", "entity_id"])
        .sort_index()
    )
    y = pd.Series(target_values, dtype=int)
    y.index = pd.MultiIndex.from_tuples(
        y.index, names=["date", "entity_id"]
    )
    y = y.reindex(frame.index)
    return EventPanelDataset(
        features=frame,
        target=y,
        entities=entities,
        horizon_days=horizon_days,
        target_variable=target_variable,
    )
