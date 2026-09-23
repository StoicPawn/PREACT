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
from .temporal import entity_feature_snapshot_with_lineage, feature_snapshot_fingerprint
from .temporal_dynamics import entity_temporal_dynamics_snapshot
from .world_context import (
    build_world_context_snapshot,
    contextual_feature_fingerprint,
)


@dataclass(frozen=True)
class EventPanelDataset:
    features: pd.DataFrame
    target: pd.Series
    entities: tuple[str, ...]
    horizon_days: int
    target_variable: str
    feature_snapshot_fingerprints: pd.Series


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
    include_world_context: bool = True,
    world_context_windows: Iterable[int] = (90, 365, 1825),
    outcome_observed_through: datetime | None = None,
) -> EventPanelDataset:
    """Build a panel for outcomes such as coup attempts or successful coups."""

    entities = tuple(sorted(set(str(x) for x in entity_ids)))
    dates = tuple(sorted(set(cutoffs)))
    variables = tuple(dict.fromkeys(str(v) for v in feature_variables))
    feature_rows: list[dict[str, object]] = []
    target_values: dict[tuple[pd.Timestamp, str], object] = {}
    snapshot_fingerprints: dict[tuple[pd.Timestamp, str], str] = {}
    world_snapshots = (
        {
            cutoff: build_world_context_snapshot(
                graph,
                cutoff=cutoff,
                windows_days=world_context_windows,
                knowledge_mode=knowledge_mode,
            )
            for cutoff in dates
        }
        if include_world_context
        else {}
    )

    for entity_id in entities:
        target = binary_event_target(
            warehouse,
            entity_id=entity_id,
            cutoffs=dates,
            target_variable=target_variable,
            horizon_days=horizon_days,
            outcome_observed_through=outcome_observed_through,
        )
        for cutoff in dates:
            row: dict[str, object] = {
                "date": pd.Timestamp(cutoff),
                "entity_id": entity_id,
            }
            point_features, lineage = entity_feature_snapshot_with_lineage(
                warehouse,
                entity_id=entity_id,
                cutoff=cutoff,
                variables=variables,
                knowledge_mode=knowledge_mode,
            )
            row.update(point_features)
            point_fingerprint = feature_snapshot_fingerprint(lineage)
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
            if include_world_context:
                world_snapshot = world_snapshots[cutoff]
                row.update(world_snapshot.features_for(entity_id))
                snapshot_fingerprints[(pd.Timestamp(cutoff), entity_id)] = (
                    contextual_feature_fingerprint(
                        point_fingerprint,
                        world_snapshot.evidence_fingerprint,
                    )
                )
            else:
                snapshot_fingerprints[(pd.Timestamp(cutoff), entity_id)] = point_fingerprint
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
            target_values[(pd.Timestamp(cutoff), entity_id)] = target.loc[pd.Timestamp(cutoff)]

    if not feature_rows:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["date", "entity_id"])
        fingerprints = pd.Series(index=empty_index, dtype="string", name="feature_snapshot_fingerprint")
        return EventPanelDataset(
            pd.DataFrame(),
            pd.Series(dtype="Int64"),
            entities,
            horizon_days,
            target_variable,
            fingerprints,
        )
    frame = (
        pd.DataFrame(feature_rows)
        .set_index(["date", "entity_id"])
        .sort_index()
    )
    y = pd.Series(target_values, dtype="Int64")
    y.index = pd.MultiIndex.from_tuples(
        y.index, names=["date", "entity_id"]
    )
    y = y.reindex(frame.index)
    fingerprints = pd.Series(snapshot_fingerprints, dtype="string", name="feature_snapshot_fingerprint")
    fingerprints.index = pd.MultiIndex.from_tuples(fingerprints.index, names=["date", "entity_id"])
    fingerprints = fingerprints.reindex(frame.index)
    if fingerprints.isna().any():
        raise ValueError("missing point-in-time feature provenance for event panel rows")
    return EventPanelDataset(
        features=frame,
        target=y,
        entities=entities,
        horizon_days=horizon_days,
        target_variable=target_variable,
        feature_snapshot_fingerprints=fingerprints,
    )