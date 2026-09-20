"""Leakage-safe event-history and network-pressure features."""

from __future__ import annotations

from datetime import datetime, timedelta
import math
from typing import Iterable

from preact.history.graph_store import HistoricalGraphStore
from preact.history.schema import KnowledgeMode


def _knowledge_sql(mode: KnowledgeMode) -> tuple[str, list[object]]:
    if mode is KnowledgeMode.STRICT_AS_KNOWN:
        return "known_at <= ? AND ", []
    return "", []


def event_history_features(
    graph: HistoricalGraphStore,
    *,
    entity_id: str,
    cutoff: datetime,
    relation_types: Iterable[str] = (
        "militarized_interstate_dispute",
        "formal_alliance",
        "direct_contiguity",
    ),
    windows_days: Iterable[int] = (365, 1825),
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
) -> dict[str, float]:
    """Summarize only relations that are admissible at the requested cutoff."""

    features: dict[str, float] = {}
    types = tuple(dict.fromkeys(str(x) for x in relation_types))
    windows = tuple(sorted({max(1, int(x)) for x in windows_days}))

    with graph.connect() as conn:
        for relation_type in types:
            knowledge_clause = (
                "known_at <= ? AND "
                if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN
                else ""
            )
            prefix_params: list[object] = (
                [cutoff] if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN else []
            )

            last = conn.execute(
                f"""
                SELECT MAX(valid_from)
                FROM historical_relations
                WHERE {knowledge_clause}relation_type = ?
                  AND valid_from <= ?
                  AND (subject_entity_id = ? OR object_entity_id = ?)
                """,
                [*prefix_params, relation_type, cutoff, entity_id, entity_id],
            ).fetchone()[0]
            if last is None:
                features[f"history:days_since_last:{relation_type}"] = 36500.0
                features[f"history:ever:{relation_type}"] = 0.0
            else:
                delta = cutoff - last
                features[f"history:days_since_last:{relation_type}"] = float(
                    max(0.0, delta.total_seconds() / 86400.0)
                )
                features[f"history:ever:{relation_type}"] = 1.0

            active = conn.execute(
                f"""
                SELECT COUNT(*), COUNT(DISTINCT
                    CASE
                        WHEN subject_entity_id = ? THEN object_entity_id
                        ELSE subject_entity_id
                    END
                )
                FROM historical_relations
                WHERE {knowledge_clause}relation_type = ?
                  AND valid_from <= ?
                  AND (valid_to IS NULL OR ? < valid_to)
                  AND (subject_entity_id = ? OR object_entity_id = ?)
                """,
                [
                    entity_id,
                    *prefix_params,
                    relation_type,
                    cutoff,
                    cutoff,
                    entity_id,
                    entity_id,
                ],
            ).fetchone()
            features[f"history:active_count:{relation_type}"] = float(active[0] or 0)
            features[f"history:active_counterparties:{relation_type}"] = float(
                active[1] or 0
            )

            # Hawkes-like excitation summaries: recent events contribute
            # exponentially more than remote ones, without fitting a Hawkes model.
            history_rows = conn.execute(
                f"""
                SELECT valid_from
                FROM historical_relations
                WHERE {knowledge_clause}relation_type = ?
                  AND valid_from <= ?
                  AND (subject_entity_id = ? OR object_entity_id = ?)
                """,
                [*prefix_params, relation_type, cutoff, entity_id, entity_id],
            ).fetchall()
            for half_life in (90, 365, 1825):
                decay = math.log(2.0) / float(half_life)
                intensity = 0.0
                for (event_time,) in history_rows:
                    age_days = max(
                        0.0,
                        (cutoff - event_time).total_seconds() / 86400.0,
                    )
                    intensity += math.exp(-decay * age_days)
                features[
                    f"history:decay_{half_life}d:{relation_type}"
                ] = float(intensity)

            for days in windows:
                start = cutoff - timedelta(days=days)
                count = conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM historical_relations
                    WHERE {knowledge_clause}relation_type = ?
                      AND valid_from > ?
                      AND valid_from <= ?
                      AND (subject_entity_id = ? OR object_entity_id = ?)
                    """,
                    [
                        *prefix_params,
                        relation_type,
                        start,
                        cutoff,
                        entity_id,
                        entity_id,
                    ],
                ).fetchone()[0]
                features[f"history:count_{days}d:{relation_type}"] = float(count or 0)

        disputes = features.get(
            "history:active_count:militarized_interstate_dispute", 0.0
        )
        alliances = features.get("history:active_count:formal_alliance", 0.0)
        features["history:alliance_dispute_balance"] = float(
            (alliances + 1.0) / (disputes + 1.0)
        )

        # Network pressure: recent disputes involving an active ally/neighbor.
        counterparties = conn.execute(
            f"""
            SELECT DISTINCT
                CASE
                    WHEN subject_entity_id = ? THEN object_entity_id
                    ELSE subject_entity_id
                END AS other
            FROM historical_relations
            WHERE {
                "known_at <= ? AND "
                if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN
                else ""
            }relation_type IN ('formal_alliance','direct_contiguity')
              AND valid_from <= ?
              AND (valid_to IS NULL OR ? < valid_to)
              AND (subject_entity_id = ? OR object_entity_id = ?)
            """,
            (
                [entity_id, cutoff, cutoff, cutoff, entity_id, entity_id]
                if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN
                else [entity_id, cutoff, cutoff, entity_id, entity_id]
            ),
        ).fetchall()
        peers = [str(row[0]) for row in counterparties if row[0]]
        if peers:
            placeholders = ",".join("?" for _ in peers)
            start = cutoff - timedelta(days=365)
            params: list[object] = []
            knowledge_clause = ""
            if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN:
                knowledge_clause = "known_at <= ? AND "
                params.append(cutoff)
            params.extend([start, cutoff, *peers, *peers])
            pressure = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM historical_relations
                WHERE {knowledge_clause}
                      relation_type = 'militarized_interstate_dispute'
                  AND valid_from > ?
                  AND valid_from <= ?
                  AND (
                        subject_entity_id IN ({placeholders})
                     OR object_entity_id IN ({placeholders})
                  )
                """,
                params,
            ).fetchone()[0]
        else:
            pressure = 0
        features["history:neighbor_dispute_pressure_365d"] = float(pressure or 0)

    return features
