"""Scheduled estimation cycle for PREACT geopolitical relationship states."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from preact.history.geopolitical_state_store import GeopoliticalStateStore
from preact.intelligence.event_clustering import cluster_event_evidence
from preact.intelligence.geopolitical_state import (
    MODEL_VERSION,
    RelationshipState,
    build_states,
    interpret_event,
)
from preact.intelligence.structural_relationships import (
    load_documented_alliance_pairs,
)
from preact.intelligence.semantic_event_enrichment import apply_semantic_enrichment


@dataclass(frozen=True)
class GeopoliticalStateCycleResult:
    mode: str
    model_version: str
    as_of: datetime
    raw_events: int
    clustered_events: int
    interpreted_impacts: int
    active_impacts: int
    structural_anchor_pairs: int
    states: int
    inserted_impacts: int
    inserted_states: int

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["as_of"] = self.as_of.isoformat()
        return payload


def _latest_map(
    states: list[RelationshipState],
) -> dict[str, RelationshipState]:
    return {state.pair_key: state for state in states}


def run_geopolitical_state_cycle(
    *,
    world_knowledge_db: str | Path,
    shared_hub_root: str | Path,
    graph_path: str | Path,
    mode: str,
    as_of: datetime | None = None,
) -> GeopoliticalStateCycleResult:
    if mode not in {"live", "canonical"}:
        raise ValueError("mode must be live or canonical")

    cutoff = as_of or datetime.now(timezone.utc)
    if cutoff.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")

    store = GeopoliticalStateStore(world_knowledge_db)
    raw_lookback_days = 1
    state_lookback_days = 3 if mode == "live" else 180
    last_impact_known_at = store.latest_impact_known_at()
    known_since = (
        max(
            cutoff - timedelta(days=1),
            last_impact_known_at - timedelta(hours=6),
        )
        if last_impact_known_at is not None
        else None
    )
    rows = store.load_recent_event_evidence(
        as_of=cutoff,
        known_cutoff=cutoff,
        lookback_days=raw_lookback_days,
        known_since=known_since,
    )
    clustered_rows = cluster_event_evidence(rows, window_hours=6.0)
    semantic = store.latest_semantic_enrichments(as_of=cutoff)
    impacts = []
    for row in clustered_rows:
        deterministic = interpret_event(row)
        if deterministic is None:
            continue
        impacts.append(
            apply_semantic_enrichment(
                deterministic,
                semantic.get(deterministic.provider_event_id),
            )
        )

    anchors = load_documented_alliance_pairs(
        shared_hub_root,
        graph_path=graph_path,
        valid_at=cutoff,
        knowledge_cutoff=cutoff,
    )
    canonical_reference = (
        _latest_map(store.latest_states(mode="canonical", as_of=cutoff))
        if mode == "live"
        else {}
    )

    inserted_impacts = store.record_impacts(impacts)
    active_impacts = store.latest_impacts(
        as_of=cutoff,
        lookback_days=state_lookback_days,
    )
    states = build_states(
        active_impacts,
        as_of=cutoff,
        mode=mode,
        anchor_pairs=anchors,
        canonical_states=canonical_reference,
    )
    inserted_states = store.record_states(states)

    return GeopoliticalStateCycleResult(
        mode=mode,
        model_version=MODEL_VERSION,
        as_of=cutoff,
        raw_events=len(rows),
        clustered_events=len(clustered_rows),
        interpreted_impacts=len(impacts),
        active_impacts=len(active_impacts),
        structural_anchor_pairs=len(anchors),
        states=len(states),
        inserted_impacts=inserted_impacts,
        inserted_states=inserted_states,
    )


__all__ = [
    "GeopoliticalStateCycleResult",
    "run_geopolitical_state_cycle",
]
