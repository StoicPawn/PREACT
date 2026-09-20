"""Future relation-event labels for replay evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd

from preact.history.graph_store import HistoricalGraphStore


def binary_relation_target(
    graph: HistoricalGraphStore,
    *,
    entity_id: str,
    cutoffs: Iterable[datetime],
    relation_type: str,
    horizon_days: int,
) -> pd.Series:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    horizon = timedelta(days=int(horizon_days))
    values: dict[pd.Timestamp, int] = {}
    with graph.connect() as conn:
        for cutoff in sorted(cutoffs):
            end = cutoff + horizon
            row = conn.execute(
                """
                SELECT 1
                FROM historical_relations
                WHERE relation_type = ?
                  AND valid_from > ?
                  AND valid_from <= ?
                  AND (subject_entity_id = ? OR object_entity_id = ?)
                LIMIT 1
                """,
                [relation_type, cutoff, end, entity_id, entity_id],
            ).fetchone()
            values[pd.Timestamp(cutoff)] = int(row is not None)
    return pd.Series(values, dtype=int).sort_index()
