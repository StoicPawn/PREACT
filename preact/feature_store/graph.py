"""Point-in-time graph features from historical geopolitical relations."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Iterable

from preact.history.graph_store import HistoricalGraphStore
from preact.history.schema import KnowledgeMode


def graph_feature_snapshot(
    graph: HistoricalGraphStore,
    *,
    entity_id: str,
    cutoff: datetime,
    recent_days: int = 365,
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
) -> dict[str, float]:
    """Build leakage-safe structural features at a cutoff."""

    active = graph.as_of(
        cutoff=cutoff,
        valid_at=cutoff,
        entity_id=entity_id,
        knowledge_mode=knowledge_mode,
    )
    counts = Counter(str(row["relation_type"]) for row in active)
    features: dict[str, float] = {
        f"graph_active:{kind}": float(count)
        for kind, count in counts.items()
    }
    features["graph_active:total"] = float(len(active))

    recent_start = cutoff - timedelta(days=max(1, int(recent_days)))
    knowledge_clause = (
        "known_at <= ? AND "
        if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN
        else ""
    )
    params = (
        [cutoff, recent_start, cutoff, entity_id, entity_id]
        if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN
        else [recent_start, cutoff, entity_id, entity_id]
    )
    with graph.connect() as conn:
        cursor = conn.execute(
            f"""
            SELECT relation_type, COUNT(*)
            FROM historical_relations
            WHERE {knowledge_clause}valid_from > ?
              AND valid_from <= ?
              AND (subject_entity_id = ? OR object_entity_id = ?)
            GROUP BY relation_type
            """,
            params,
        )
        for relation_type, count in cursor.fetchall():
            features[f"graph_recent:{relation_type}"] = float(count)
    return features
