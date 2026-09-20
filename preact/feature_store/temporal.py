"""Point-in-time feature materialization from the bitemporal warehouse."""

from __future__ import annotations

from datetime import datetime
import json
from typing import Iterable

import pandas as pd

from preact.history.schema import KnowledgeMode
from preact.history.warehouse import HistoricalWarehouse


def _numeric_scalar(value_json: str | None) -> float | None:
    if value_json is None:
        return None
    try:
        value = json.loads(value_json)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def entity_feature_snapshot(
    warehouse: HistoricalWarehouse,
    *,
    entity_id: str,
    cutoff: datetime,
    valid_at: datetime | None = None,
    variables: Iterable[str] | None = None,
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
) -> dict[str, float]:
    rows = warehouse.latest_observations_as_of(
        cutoff=cutoff,
        entity_id=entity_id,
        variables=variables,
        knowledge_mode=knowledge_mode,
    )
    features: dict[str, float] = {}
    for row in rows:
        value = _numeric_scalar(row.get("value_json"))
        if value is not None:
            features[str(row["variable"])] = value
    return features


def entity_feature_frame(
    warehouse: HistoricalWarehouse,
    *,
    entity_id: str,
    cutoffs: Iterable[datetime],
    variables: Iterable[str] | None = None,
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for cutoff in sorted(cutoffs):
        row: dict[str, object] = {"date": pd.Timestamp(cutoff)}
        row.update(
            entity_feature_snapshot(
                warehouse,
                entity_id=entity_id,
                cutoff=cutoff,
                valid_at=cutoff,
                variables=variables,
                knowledge_mode=knowledge_mode,
            )
        )
        records.append(row)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).set_index("date").sort_index()
    return frame
