from __future__ import annotations

from datetime import datetime, timezone
import io
import zipfile

import pandas as pd

from preact.history.snapshot_store import SourceSnapshotStore
from preact.history.world_knowledge_store import WorldKnowledgeStore
from preact.intelligence.world_cycle import run_world_intelligence_cycle
from preact.intelligence.world_events import WorldEventObservation, project_gdelt_events


def utc(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, 27, hour, minute, tzinfo=timezone.utc)


def test_world_event_store_is_point_in_time_and_deduplicates_provider_event(tmp_path):
    store = WorldKnowledgeStore(tmp_path / "world-events.duckdb")

    first = WorldEventObservation(
        provider="gdelt",
        provider_event_id="123",
        event_time=utc(8),
        known_at=utc(9),
        actor1_entity_id="country:ITA",
        actor2_entity_id="country:FRA",
        event_code="042",
        event_root_code="04",
        quad_class=1,
        goldstein=3.0,
        tone=1.2,
        num_articles=2,
        source_url="https://example.test/first",
        snapshot_checksum="aaa",
    )
    revised = WorldEventObservation(
        provider="gdelt",
        provider_event_id="123",
        event_time=utc(8),
        known_at=utc(10),
        actor1_entity_id="country:ITA",
        actor2_entity_id="country:FRA",
        event_code="042",
        event_root_code="04",
        quad_class=1,
        goldstein=3.0,
        tone=1.4,
        num_articles=8,
        source_url="https://example.test/revised",
        snapshot_checksum="bbb",
    )

    assert store.record_world_events([first, revised]) == 2
    assert store.record_world_events([first]) == 0

    at_0930 = store.event_timeline(
        "country:ITA",
        as_of=utc(9, 30),
        known_cutoff=utc(9, 30),
    )
    assert len(at_0930) == 1
    assert at_0930[0]["num_articles"] == 2
    assert at_0930[0]["snapshot_checksum"] == "aaa"

    at_1030 = store.event_timeline(
        "country:ITA",
        as_of=utc(10, 30),
        known_cutoff=utc(10, 30),
    )
    assert len(at_1030) == 1
    assert at_1030[0]["num_articles"] == 8
    assert at_1030[0]["snapshot_checksum"] == "bbb"


def test_project_gdelt_events_preserves_source_and_semantics():
    frame = pd.DataFrame(
        [
            {
                "event_id": "456",
                "event_date": pd.Timestamp("2026-09-27"),
                "actor1_country": "ITA",
                "actor2_country": "FRA",
                "actor1_name": "Italian Government",
                "actor2_name": "French Government",
                "event_code": "040",
                "event_base_code": "040",
                "event_root_code": "04",
                "quad_class": 1,
                "goldstein": 1.0,
                "tone": 0.4,
                "num_mentions": 5,
                "num_sources": 3,
                "num_articles": 4,
                "action_location": "Rome, Lazio, Italy",
                "source_url": "https://example.test/story",
                "retrieved_at": utc(9, 15),
                "snapshot_checksum": "snapshot-1",
            }
        ]
    )

    events = project_gdelt_events(frame)
    assert len(events) == 1
    event = events[0]
    assert event.actor1_entity_id == "country:ITA"
    assert event.actor2_entity_id == "country:FRA"
    assert event.event_root_code == "04"
    assert event.evidence_class == "provider_derived_event"
    assert event.snapshot_checksum == "snapshot-1"


def _event_zip() -> bytes:
    values = [""] * 61
    values[0] = "999"
    values[1] = "20260927"
    values[6] = "Italy"
    values[7] = "ITA"
    values[16] = "France"
    values[17] = "FRA"
    values[26] = "040"
    values[27] = "040"
    values[28] = "04"
    values[29] = "1"
    values[30] = "1.0"
    values[31] = "5"
    values[32] = "3"
    values[33] = "4"
    values[34] = "0.5"
    values[52] = "Rome, Lazio, Italy"
    values[59] = "20260927091500"
    values[60] = "https://example.test/gdelt-story"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("events.csv", "\t".join(values) + "\n")
    return buffer.getvalue()


def test_world_cycle_consumes_shared_snapshots_without_provider_fetch(tmp_path):
    hub_root = tmp_path / "shared"
    snapshots = SourceSnapshotStore(hub_root / "snapshots")

    snapshots.put(
        source_id="gdelt",
        payload=b"code\tlabel\nITA\tItaly\nFRA\tFrance\n",
        retrieved_at=utc(8),
        source_url="https://example.test/CAMEO.country.txt",
        source_release="GDELT CAMEO country lookup",
        operation="reference_cameo_country",
    )
    snapshots.put(
        source_id="gdelt",
        payload=_event_zip(),
        retrieved_at=utc(9, 15),
        source_url="https://example.test/20260927091500.export.CSV.zip",
        source_release="20260927091500.export.CSV.zip",
        operation="realtime_events",
    )

    world_db = tmp_path / "world.duckdb"
    result = run_world_intelligence_cycle(
        shared_hub_root=hub_root,
        world_knowledge_db=world_db,
        lookback_days=1,
        as_of=utc(10),
    )

    assert result.status == "ready"
    assert result.shared_snapshot_count == 1
    assert result.raw_event_count == 1
    assert result.resolved_interaction_count == 1
    assert result.projected_event_observations == 1
    assert result.inserted_event_observations == 1

    store = WorldKnowledgeStore(world_db)
    events = store.event_timeline(
        "country:ITA",
        as_of=utc(10),
        known_cutoff=utc(10),
    )
    assert len(events) == 1
    assert events[0]["provider_event_id"] == "999"
    assert events[0]["actor2_entity_id"] == "country:FRA"

    second = run_world_intelligence_cycle(
        shared_hub_root=hub_root,
        world_knowledge_db=world_db,
        lookback_days=1,
        as_of=utc(10),
    )
    assert second.inserted_event_observations == 0
