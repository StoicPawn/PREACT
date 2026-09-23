"""Point-in-time news/event context for geopolitical forecasting.

GDELT event records are retained as bitemporal observations in the historical warehouse.
This module summarizes the entire admissible event stream at a cutoff, not just events
directly attached to the focal country. It provides system-wide and focal-country news
pressure features while keeping the underlying event rows auditable and leakage-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from typing import Iterable

import pandas as pd

from preact.history.schema import KnowledgeMode
from preact.history.warehouse import HistoricalWarehouse


def _number(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(parsed) else float(parsed)


def _event_payload(row: dict) -> dict:
    raw = row.get("value_json")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _fingerprint(rows: Iterable[dict]) -> str:
    canonical = []
    for row in rows:
        canonical.append(
            {
                "record_id": str(row.get("record_id") or ""),
                "entity_id": str(row.get("entity_id") or ""),
                "valid_from": pd.Timestamp(row["valid_from"]).isoformat(),
                "known_at": pd.Timestamp(row["known_at"]).isoformat(),
                "retrieved_at": pd.Timestamp(row["retrieved_at"]).isoformat(),
                "source_ref": str(row.get("source_ref") or ""),
                "dataset_version": (
                    None
                    if row.get("dataset_version") is None
                    else str(row.get("dataset_version"))
                ),
                "value_json": str(row.get("value_json") or ""),
            }
        )
    payload = json.dumps(
        sorted(canonical, key=lambda item: (item["record_id"], item["known_at"])),
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _event_signals(payload: dict) -> tuple[float, float, float, float]:
    """Return cooperation, conflict, tone and log-volume contributions."""

    goldstein = _number(payload.get("goldstein_scale"))
    tone = _number(payload.get("avg_tone"))
    articles = max(
        0.0,
        _number(
            payload.get("num_articles"),
            _number(payload.get("num_mentions"), 1.0),
        ),
    )
    volume = math.log1p(articles)
    if volume <= 0:
        volume = math.log(2.0)

    quad = str(payload.get("quad_class") or "").strip()
    if goldstein > 0:
        cooperation = min(1.0, goldstein / 10.0) * volume
        conflict = 0.0
    elif goldstein < 0:
        cooperation = 0.0
        conflict = min(1.0, -goldstein / 10.0) * volume
    elif quad in {"1", "2"}:
        cooperation = 0.25 * volume
        conflict = 0.0
    elif quad in {"3", "4"}:
        cooperation = 0.0
        conflict = 0.25 * volume
    else:
        cooperation = 0.0
        conflict = 0.0
    return cooperation, conflict, tone * volume, volume


@dataclass(frozen=True)
class NewsContextSnapshot:
    cutoff: datetime
    windows_days: tuple[int, ...]
    rows: tuple[dict, ...]
    evidence_fingerprint: str

    def features_for(self, entity_id: str) -> dict[str, float]:
        entity = str(entity_id)
        output: dict[str, float] = {}

        for window in self.windows_days:
            start = pd.Timestamp(self.cutoff) - pd.Timedelta(days=int(window))
            rows = [
                row
                for row in self.rows
                if pd.Timestamp(row["valid_from"]) > start
                and pd.Timestamp(row["valid_from"]) <= pd.Timestamp(self.cutoff)
            ]
            focal = [row for row in rows if str(row.get("entity_id")) == entity]

            for scope, selected in (("system", rows), ("focal", focal)):
                cooperation = 0.0
                conflict = 0.0
                tone_weighted = 0.0
                volume = 0.0
                root_counts: dict[str, int] = {}
                for row in selected:
                    payload = _event_payload(row)
                    coop, conf, tone_contribution, event_volume = _event_signals(payload)
                    cooperation += coop
                    conflict += conf
                    tone_weighted += tone_contribution
                    volume += event_volume
                    root = str(payload.get("event_root_code") or "").strip()
                    if root:
                        root_counts[root] = root_counts.get(root, 0) + 1

                prefix = f"news_context:{scope}_{window}d"
                output[f"{prefix}:events"] = float(len(selected))
                output[f"{prefix}:log_volume"] = float(volume)
                output[f"{prefix}:cooperation_pressure"] = float(cooperation)
                output[f"{prefix}:conflict_pressure"] = float(conflict)
                output[f"{prefix}:net_pressure"] = float(cooperation - conflict)
                output[f"{prefix}:weighted_tone"] = float(
                    tone_weighted / volume if volume > 0 else 0.0
                )
                for root, count in root_counts.items():
                    output[f"{prefix}:root_{root}"] = float(count)

            output[f"news_context:focal_share_{window}d"] = float(
                len(focal) / len(rows) if rows else 0.0
            )

        for half_life in (30, 90, 365):
            decay = math.log(2.0) / float(half_life)
            system_conflict = 0.0
            focal_conflict = 0.0
            system_cooperation = 0.0
            focal_cooperation = 0.0
            for row in self.rows:
                age_days = max(
                    0.0,
                    (
                        pd.Timestamp(self.cutoff) - pd.Timestamp(row["valid_from"])
                    ).total_seconds()
                    / 86400.0,
                )
                recency = math.exp(-decay * age_days)
                coop, conf, _, _ = _event_signals(_event_payload(row))
                system_cooperation += recency * coop
                system_conflict += recency * conf
                if str(row.get("entity_id")) == entity:
                    focal_cooperation += recency * coop
                    focal_conflict += recency * conf
            output[f"news_context:system_decay_{half_life}d:cooperation"] = float(
                system_cooperation
            )
            output[f"news_context:system_decay_{half_life}d:conflict"] = float(
                system_conflict
            )
            output[f"news_context:focal_decay_{half_life}d:cooperation"] = float(
                focal_cooperation
            )
            output[f"news_context:focal_decay_{half_life}d:conflict"] = float(
                focal_conflict
            )

        return output


def build_news_context_snapshot(
    warehouse: HistoricalWarehouse,
    *,
    cutoff: datetime,
    windows_days: Iterable[int] = (30, 90, 365),
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
) -> NewsContextSnapshot:
    """Materialize all admissible GDELT event observations for a cutoff."""

    windows = tuple(sorted({max(1, int(days)) for days in windows_days}))
    if not windows:
        raise ValueError("windows_days must contain at least one positive window")

    start = cutoff - timedelta(days=max(windows))
    clauses = [
        "variable = 'gdelt_event'",
        "valid_from > ?",
        "valid_from <= ?",
    ]
    params: list[object] = [start, cutoff]
    if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN:
        clauses.append("known_at <= ?")
        params.append(cutoff)

    with warehouse.connect() as conn:
        cursor = conn.execute(
            """
            SELECT record_id, entity_id, value_json, valid_from, known_at,
                   source, source_ref, retrieved_at, dataset_version
            FROM temporal_records
            WHERE """
            + " AND ".join(clauses)
            + " ORDER BY valid_from, record_id",
            params,
        )
        columns = [item[0] for item in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    return NewsContextSnapshot(
        cutoff=cutoff,
        windows_days=windows,
        rows=tuple(rows),
        evidence_fingerprint=_fingerprint(rows),
    )


__all__ = ["NewsContextSnapshot", "build_news_context_snapshot"]
