"""Future-outcome labels for historical replay datasets."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd

from preact.history.warehouse import HistoricalWarehouse


def binary_event_target(
    warehouse: HistoricalWarehouse,
    *,
    entity_id: str,
    cutoffs: Iterable[datetime],
    target_variable: str,
    horizon_days: int,
    outcome_observed_through: datetime | None = None,
) -> pd.Series:
    """Return 1 when a realized target event occurs after cutoff within horizon."""

    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    values: dict[pd.Timestamp, int | pd._libs.missing.NAType] = {}
    horizon = timedelta(days=int(horizon_days))
    for cutoff in sorted(cutoffs):
        if (
            outcome_observed_through is not None
            and cutoff + horizon > outcome_observed_through
        ):
            values[pd.Timestamp(cutoff)] = pd.NA
            continue
        events = warehouse.records_in_valid_window(
            start_exclusive=cutoff,
            end_inclusive=cutoff + horizon,
            entity_id=entity_id,
            variable=target_variable,
        )
        values[pd.Timestamp(cutoff)] = int(bool(events))
    return pd.Series(values, dtype="Int64").sort_index()


def first_event_time(
    warehouse: HistoricalWarehouse,
    *,
    entity_id: str,
    cutoff: datetime,
    target_variable: str,
    horizon_days: int,
) -> datetime | None:
    events = warehouse.records_in_valid_window(
        start_exclusive=cutoff,
        end_inclusive=cutoff + timedelta(days=int(horizon_days)),
        entity_id=entity_id,
        variable=target_variable,
    )
    if not events:
        return None
    return min(row["valid_from"] for row in events)
