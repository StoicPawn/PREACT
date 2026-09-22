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
    outcome_observed_through: datetime | None = None,
) -> pd.Series:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    horizon = timedelta(days=int(horizon_days))
    values: dict[pd.Timestamp, object] = {}
    with graph.connect() as conn:
        for cutoff in sorted(cutoffs):
            end = cutoff + horizon
            if (
                outcome_observed_through is not None
                and end > outcome_observed_through
            ):
                values[pd.Timestamp(cutoff)] = pd.NA
                continue
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
    return pd.Series(values, dtype="Int64").sort_index()
