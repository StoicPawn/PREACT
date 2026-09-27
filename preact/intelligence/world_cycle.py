"""Local fan-out from Shared Data Hub snapshots into PREACT World Intelligence.

This module performs no provider fetches. It reads immutable snapshots already owned
by the ACEPC Shared Data Hub and projects them into PREACT's point-in-time world
event memory.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preact.history.world_knowledge_store import WorldKnowledgeStore
from preact.intelligence.gdelt_relationships import load_recent_relationship_edges
from preact.intelligence.world_events import project_gdelt_events


@dataclass(frozen=True)
class WorldIntelligenceCycleResult:
    status: str
    shared_snapshot_count: int
    raw_event_count: int
    resolved_interaction_count: int
    projected_event_observations: int
    inserted_event_observations: int
    newest_shared_retrieval: datetime | None
    country_map_snapshot_checksum: str | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["newest_shared_retrieval"] = (
            self.newest_shared_retrieval.astimezone(timezone.utc).isoformat()
            if self.newest_shared_retrieval is not None
            else None
        )
        return payload


def run_world_intelligence_cycle(
    *,
    shared_hub_root: str | Path,
    world_knowledge_db: str | Path,
    lookback_days: int = 2,
    as_of: datetime | None = None,
) -> WorldIntelligenceCycleResult:
    """Project shared GDELT realtime snapshots into PREACT without refetching GDELT."""

    cutoff = as_of or datetime.now(timezone.utc)
    if cutoff.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")

    batch = load_recent_relationship_edges(
        shared_hub_root,
        as_of=cutoff,
        lookback_days=lookback_days,
        min_events=1,
        acquire_country_map_if_missing=False,
    )
    observations = project_gdelt_events(batch.events)
    store = WorldKnowledgeStore(world_knowledge_db)
    inserted = store.record_world_events(observations)

    if batch.snapshot_count == 0:
        status = "no_shared_snapshots"
    elif batch.country_map_snapshot_checksum is None:
        status = "missing_shared_country_map"
    elif batch.resolved_interaction_count == 0:
        status = "no_resolved_interactions"
    else:
        status = "ready"

    return WorldIntelligenceCycleResult(
        status=status,
        shared_snapshot_count=batch.snapshot_count,
        raw_event_count=batch.raw_event_count,
        resolved_interaction_count=batch.resolved_interaction_count,
        projected_event_observations=len(observations),
        inserted_event_observations=inserted,
        newest_shared_retrieval=batch.newest_retrieved_at,
        country_map_snapshot_checksum=batch.country_map_snapshot_checksum,
    )


__all__ = ["WorldIntelligenceCycleResult", "run_world_intelligence_cycle"]
