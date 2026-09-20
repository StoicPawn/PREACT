"""Build multi-entity point-in-time panels for geopolitical forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

from preact.history.graph_store import HistoricalGraphStore
from preact.history.warehouse import HistoricalWarehouse
from .graph import graph_feature_snapshot
from .graph_targets import binary_relation_target
from .temporal import entity_feature_snapshot


@dataclass(frozen=True)
class PanelDataset:
    features: pd.DataFrame
    target: pd.Series
    entities: tuple[str, ...]
    horizon_days: int
    target_relation_type: str


def build_relation_risk_panel(
    *,
    warehouse: HistoricalWarehouse,
    graph: HistoricalGraphStore,
    entity_ids: Iterable[str],
    cutoffs: Iterable[datetime],
    feature_variables: Iterable[str],
    target_relation_type: str,
    horizon_days: int,
    graph_recent_days: int = 365,
) -> PanelDataset:
    entities=tuple(sorted(set(entity_ids)))
    dates=tuple(sorted(set(cutoffs)))
    variables=tuple(feature_variables)
    feature_rows=[]
    target_values={}

    for entity_id in entities:
        y=binary_relation_target(
            graph,
            entity_id=entity_id,
            cutoffs=dates,
            relation_type=target_relation_type,
            horizon_days=horizon_days,
        )
        for cutoff in dates:
            row={"date":pd.Timestamp(cutoff),"entity_id":entity_id}
            row.update(entity_feature_snapshot(
                warehouse,
                entity_id=entity_id,
                cutoff=cutoff,
                variables=variables,
            ))
            row.update(graph_feature_snapshot(
                graph,
                entity_id=entity_id,
                cutoff=cutoff,
                recent_days=graph_recent_days,
            ))
            feature_rows.append(row)
            target_values[(pd.Timestamp(cutoff),entity_id)]=int(y.loc[pd.Timestamp(cutoff)])

    if not feature_rows:
        return PanelDataset(pd.DataFrame(),pd.Series(dtype=int),entities,horizon_days,target_relation_type)

    frame=pd.DataFrame(feature_rows).set_index(["date","entity_id"]).sort_index()
    target=pd.Series(target_values,dtype=int)
    target.index=pd.MultiIndex.from_tuples(target.index,names=["date","entity_id"])
    target=target.reindex(frame.index)
    return PanelDataset(frame,target,entities,horizon_days,target_relation_type)
