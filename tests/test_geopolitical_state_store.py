from datetime import datetime, timezone

from preact.history.geopolitical_state_store import GeopoliticalStateStore
from preact.history.world_knowledge_store import WorldKnowledgeStore
from preact.intelligence.geopolitical_state import build_states, interpret_event
from preact.intelligence.world_events import WorldEventObservation


NOW = datetime(2026, 9, 27, 16, tzinfo=timezone.utc)


def test_state_store_joins_event_mentions_and_gkg_context(tmp_path):
    db = tmp_path / "world.duckdb"
    world = WorldKnowledgeStore(db)
    world.record_world_events(
        [
            WorldEventObservation(
                provider="gdelt",
                provider_event_id="e1",
                event_time=NOW,
                known_at=NOW,
                actor1_entity_id="country:ITA",
                actor2_entity_id="country:FRA",
                event_code="163",
                event_base_code="163",
                event_root_code="16",
                quad_class=3,
                goldstein=-5.0,
                tone=-2.0,
                num_sources=2,
                num_articles=3,
                source_url="https://wire.example/story",
                snapshot_checksum="events-1",
            )
        ]
    )
    world.record_gdelt_context(
        mention_observations=[
            {
                "provider_event_id": "e1",
                "known_at": NOW,
                "mention_count": 7,
                "distinct_source_count": 4,
                "mention_sources": ["a", "b", "c", "d"],
                "mean_confidence": 85.0,
                "max_confidence": 92.0,
                "mean_document_tone": -2.5,
                "latest_mention_time": "20260927160000",
                "snapshot_checksum": "mentions-1",
            }
        ],
        gkg_documents=[
            {
                "gkg_record_id": "g1",
                "known_at": NOW,
                "source": "wire.example",
                "document_url": "https://wire.example/story",
                "country_iso3": ["FRA", "ITA"],
                "country_codes": ["FR", "IT"],
                "themes": ["SANCTIONS", "ECONOMIC"],
                "persons": [],
                "organizations": [],
                "overall_tone": -2.0,
                "all_names": [],
                "snapshot_checksum": "gkg-1",
                "country_mapping_checksum": "map-1",
            }
        ],
        processed_snapshots=[],
    )

    store = GeopoliticalStateStore(db)
    rows = store.load_recent_event_evidence(
        as_of=NOW,
        known_cutoff=NOW,
        lookback_days=3,
    )
    assert len(rows) == 1
    assert rows[0]["mention_source_count"] == 4
    assert "SANCTIONS" in rows[0]["themes"]

    impact = interpret_event(rows[0])
    assert impact is not None
    assert impact.vector["economic_alignment"] <= -0.7
    assert impact.confidence > 0.6

    states = build_states([impact], as_of=NOW, mode="canonical")
    assert store.record_impacts([impact]) == 1
    assert store.record_states(states) == 1
    latest = store.states_for_focal("ITA", mode="canonical", as_of=NOW)
    assert len(latest) == 1
    assert latest[0].pair_key == "FRA|ITA"
    assert latest[0].overall_score < 0


def test_latest_states_returns_only_latest_complete_run(tmp_path):
    from dataclasses import replace
    from datetime import timedelta

    db = tmp_path / "snapshots.duckdb"
    WorldKnowledgeStore(db)
    store = GeopoliticalStateStore(db)

    impact_a = interpret_event(
        {
            "provider_event_id": "a",
            "event_time": NOW,
            "known_at": NOW,
            "actor1_entity_id": "country:ITA",
            "actor2_entity_id": "country:FRA",
            "event_code": "040",
            "event_base_code": "040",
            "event_root_code": "04",
            "quad_class": 1,
            "goldstein": 3.0,
            "num_sources": 3,
            "num_articles": 3,
            "mention_source_count": 3,
            "corroborating_mentions": 3,
            "mention_max_confidence": 90,
            "is_root_event": True,
            "source_url": "https://a.example/story",
            "themes": [],
        }
    )
    assert impact_a is not None
    impact_b = replace(
        impact_a,
        provider_event_id="b",
        source_iso3="DEU",
        target_iso3="USA",
        source_url="https://b.example/story",
    )

    store.record_states(build_states([impact_a, impact_b], as_of=NOW, mode="canonical"))
    later = NOW + timedelta(hours=1)
    store.record_states(build_states([impact_a], as_of=later, mode="canonical"))

    latest = store.latest_states(mode="canonical", as_of=later)
    assert [state.pair_key for state in latest] == ["FRA|ITA"]
