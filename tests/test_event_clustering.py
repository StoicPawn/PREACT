from datetime import datetime, timedelta, timezone

from preact.intelligence.event_clustering import cluster_event_evidence


NOW = datetime(2026, 9, 27, 16, tzinfo=timezone.utc)


def row(
    event_id: str,
    *,
    known_at=NOW,
    location="Rome",
    base="040",
    source_url=None,
):
    return {
        "provider_event_id": event_id,
        "event_time": NOW,
        "known_at": known_at,
        "actor1_entity_id": "country:ITA",
        "actor2_entity_id": "country:FRA",
        "event_code": base,
        "event_base_code": base,
        "event_root_code": base[:2],
        "actor1_name": "Italian Government",
        "actor2_name": "French Government",
        "action_location": location,
        "num_sources": 3,
        "num_articles": 5,
        "mention_source_count": 4,
        "mention_max_confidence": 90,
        "goldstein": 3.0,
        "tone": 1.0,
        "themes": ["DIPLOMACY"],
        "persons": [],
        "organizations": [],
        "source_url": source_url or f"https://wire.example/{event_id}",
    }


def test_same_episode_clusters_without_multiplying_counts():
    rows = [
        row("e1", source_url="https://wire.example/same-story"),
        row(
            "e2",
            known_at=NOW + timedelta(minutes=30),
            source_url="https://wire.example/same-story",
        ),
    ]
    clustered = cluster_event_evidence(rows)
    assert len(clustered) == 1
    item = clustered[0]
    assert item["cluster_size"] == 2
    assert item["cluster_event_ids"] == ["e1", "e2"]
    assert item["num_articles"] == 5
    assert item["mention_source_count"] == 4
    assert item["provider_event_id"].startswith("gdc_")


def test_distinct_location_or_time_window_stays_separate():
    rows = [
        row("e1", location="Rome"),
        row("e2", location="Paris"),
        row("e3", known_at=NOW + timedelta(hours=8), location="Rome"),
    ]
    clustered = cluster_event_evidence(rows, window_hours=6)
    assert len(clustered) == 3
