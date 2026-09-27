"""Nightly bounded local semantic enrichment for material GDELT events."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preact.history.geopolitical_state_store import GeopoliticalStateStore
from preact.intelligence.event_clustering import cluster_event_evidence
from preact.intelligence.geopolitical_state import interpret_event
from preact.intelligence.semantic_event_enrichment import (
    classify_with_ollama,
    materiality_score,
)


@dataclass(frozen=True)
class SemanticEnrichmentCycleResult:
    as_of: datetime
    candidates: int
    attempted: int
    enriched: int
    failed: int
    skipped_existing: int
    model: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["as_of"] = self.as_of.isoformat()
        return payload


def run_semantic_enrichment_cycle(
    *,
    world_knowledge_db: str | Path,
    max_events: int = 40,
    lookback_days: int = 3,
    model: str = "qwen3:1.7b",
    as_of: datetime | None = None,
) -> SemanticEnrichmentCycleResult:
    cutoff = as_of or datetime.now(timezone.utc)
    if cutoff.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if max_events < 1:
        raise ValueError("max_events must be >= 1")

    store = GeopoliticalStateStore(world_knowledge_db)
    rows = store.load_recent_event_evidence(
        as_of=cutoff,
        known_cutoff=cutoff,
        lookback_days=lookback_days,
    )
    clustered_rows = cluster_event_evidence(rows, window_hours=6.0)
    existing = store.latest_semantic_enrichments(as_of=cutoff)

    ranked: list[tuple[float, dict[str, Any], Any]] = []
    skipped = 0
    for row in clustered_rows:
        impact = interpret_event(row)
        if impact is None:
            continue
        if impact.provider_event_id in existing:
            skipped += 1
            continue
        score = materiality_score(impact)
        if score < 0.40:
            continue
        ranked.append((score, row, impact))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[: int(max_events)]

    enriched = failed = 0
    for _score, row, impact in selected:
        try:
            result = classify_with_ollama(
                row,
                impact,
                model=model,
                timeout_seconds=75.0,
            )
            if store.record_semantic_enrichment(result, known_at=cutoff):
                enriched += 1
        except Exception:
            failed += 1

    return SemanticEnrichmentCycleResult(
        as_of=cutoff,
        candidates=len(ranked),
        attempted=len(selected),
        enriched=enriched,
        failed=failed,
        skipped_existing=skipped,
        model=model,
    )


__all__ = ["SemanticEnrichmentCycleResult", "run_semantic_enrichment_cycle"]
