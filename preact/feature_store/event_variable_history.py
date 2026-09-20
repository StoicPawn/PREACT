"""History features for country-level event variables such as coups."""

from __future__ import annotations

from datetime import datetime, timedelta
import math
from typing import Iterable

from preact.history.schema import KnowledgeMode
from preact.history.warehouse import HistoricalWarehouse


def event_variable_history_features(
    warehouse: HistoricalWarehouse,
    *,
    entity_id: str,
    cutoff: datetime,
    variables: Iterable[str],
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
    windows_days: Iterable[int] = (365, 1825, 3650),
) -> dict[str, float]:
    output: dict[str, float] = {}
    for variable in tuple(dict.fromkeys(str(v) for v in variables if str(v))):
        clauses = [
            "entity_id = ?",
            "variable = ?",
            "valid_from <= ?",
        ]
        params: list[object] = [entity_id, variable, cutoff]
        if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN:
            clauses.append("known_at <= ?")
            params.append(cutoff)
        with warehouse.connect() as conn:
            rows = conn.execute(
                "SELECT MIN(valid_from) AS valid_from "
                "FROM temporal_records WHERE "
                + " AND ".join(clauses)
                + " GROUP BY source, source_ref "
                "ORDER BY valid_from",
                params,
            ).fetchall()
        dates = [row[0] for row in rows]
        if dates:
            output[f"event_history:ever:{variable}"] = 1.0
            output[f"event_history:days_since_last:{variable}"] = float(
                max(0.0, (cutoff - dates[-1]).total_seconds() / 86400.0)
            )
        else:
            output[f"event_history:ever:{variable}"] = 0.0
            output[f"event_history:days_since_last:{variable}"] = 36500.0
        for window in sorted({max(1, int(x)) for x in windows_days}):
            start = cutoff - timedelta(days=window)
            count = sum(start < event_time <= cutoff for event_time in dates)
            output[f"event_history:count_{window}d:{variable}"] = float(count)
        for half_life in (180, 730, 1825):
            decay = math.log(2.0) / half_life
            intensity = sum(
                math.exp(
                    -decay
                    * max(
                        0.0,
                        (cutoff - event_time).total_seconds() / 86400.0,
                    )
                )
                for event_time in dates
            )
            output[
                f"event_history:decay_{half_life}d:{variable}"
            ] = float(intensity)
    return output
