"""Point-in-time dynamics for numeric country/polity indicators."""

from __future__ import annotations

from datetime import datetime
import json
from typing import Iterable

import numpy as np

from preact.history.schema import KnowledgeMode
from preact.history.warehouse import HistoricalWarehouse


def _numeric(value_json: str | None) -> float | None:
    if value_json is None:
        return None
    try:
        value = json.loads(value_json)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return None


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    x = x - x.mean()
    denom = float(x @ x)
    return float((x @ (y - y.mean())) / denom) if denom > 0 else 0.0


def entity_temporal_dynamics_snapshot(
    warehouse: HistoricalWarehouse,
    *,
    entity_id: str,
    cutoff: datetime,
    variables: Iterable[str],
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
    history_points: int = 5,
) -> dict[str, float]:
    """Compute leakage-safe level/change/trend features from reported indicators."""

    selected = tuple(dict.fromkeys(str(v) for v in variables if str(v)))
    if not selected:
        return {}
    placeholders = ",".join("?" for _ in selected)
    clauses = [
        "entity_id = ?",
        f"variable IN ({placeholders})",
        "valid_from <= ?",
    ]
    params: list[object] = [entity_id, *selected, cutoff]
    if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN:
        clauses.append("known_at <= ?")
        params.append(cutoff)

    # Keep only the latest known revision of each historical observation date.
    sql = f"""
        SELECT * EXCLUDE (rn)
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY variable, valid_from
                       ORDER BY known_at DESC, retrieved_at DESC, record_id DESC
                   ) AS rn
            FROM temporal_records
            WHERE {' AND '.join(clauses)}
        )
        WHERE rn = 1
        ORDER BY variable, valid_from
    """

    with warehouse.connect() as conn:
        cursor = conn.execute(sql, params)
        columns = [item[0] for item in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    grouped: dict[str, list[tuple[datetime, float]]] = {}
    for row in rows:
        value = _numeric(row.get("value_json"))
        if value is None:
            continue
        grouped.setdefault(str(row["variable"]), []).append(
            (row["valid_from"], value)
        )

    features: dict[str, float] = {}
    n = max(2, int(history_points))
    for variable in selected:
        observations = grouped.get(variable, [])
        if not observations:
            continue
        observations = observations[-n:]
        dates = [item[0] for item in observations]
        values = [item[1] for item in observations]
        last = float(values[-1])
        features[f"dyn:last:{variable}"] = last
        features[f"dyn:observations:{variable}"] = float(len(values))
        features[f"dyn:age_days:{variable}"] = float(
            max(0.0, (cutoff - dates[-1]).total_seconds() / 86400.0)
        )
        if len(values) >= 2:
            delta = float(values[-1] - values[-2])
            features[f"dyn:delta1:{variable}"] = delta
            denominator = max(abs(values[-2]), 1e-9)
            features[f"dyn:relative_delta1:{variable}"] = float(
                np.clip(delta / denominator, -10.0, 10.0)
            )
        else:
            features[f"dyn:delta1:{variable}"] = 0.0
            features[f"dyn:relative_delta1:{variable}"] = 0.0
        features[f"dyn:slope:{variable}"] = _slope(values)
        features[f"dyn:volatility:{variable}"] = float(
            np.std(values, ddof=1) if len(values) >= 2 else 0.0
        )
    return features
