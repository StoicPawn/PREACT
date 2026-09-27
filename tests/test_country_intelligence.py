from __future__ import annotations

from datetime import datetime, timezone

from preact.data_hub.news_store import SharedNewsStore
from preact.history.schema import EvidenceClass, Provenance, TemporalRecord
from preact.history.warehouse import HistoricalWarehouse
from preact.history.world_knowledge_store import WorldKnowledgeStore
from preact.intelligence.country_intelligence import assemble_country_intelligence_profile
from preact.intelligence.knowledge_update import AutonomousKnowledgeUpdater
from preact.intelligence.world_events import WorldEventObservation
from preact.intelligence.world_knowledge import (
    KnowledgeDomain,
    KnowledgeUpdateCandidate,
    KnowledgeUpdateKind,
    SourceEvidence,
)


def utc(day: int, hour: int = 12) -> datetime:
    return datetime(2026, 9, day, hour, tzinfo=timezone.utc)


def promote(
    updater: AutonomousKnowledgeUpdater,
    *,
    field: str,
    value,
    domain: KnowledgeDomain,
    day: int = 20,
) -> None:
    when = utc(day)
    source = SourceEvidence(
        source="official_test_source",
        source_ref=f"https://official.example/{field}/{day}",
        published_at=when,
        retrieved_at=when,
        independent_group="official_institution",
        authoritative=True,
    )
    result = updater.process(
        KnowledgeUpdateCandidate(
            entity_id="country:ITA",
            field=field,
            value=value,
            valid_from=when,
            detected_at=when,
            domain=domain,
            kind=KnowledgeUpdateKind.FACT,
            confidence=0.98,
            evidence=(source,),
        )
    )
    assert result.assertion_id is not None


def build_fixture(tmp_path):
    world = WorldKnowledgeStore(tmp_path / "world.duckdb")
    updater = AutonomousKnowledgeUpdater(world)

    promote(
        updater,
        field="government_form",
        value="Test parliamentary republic",
        domain=KnowledgeDomain.POLITICS,
    )
    promote(
        updater,
        field="head_of_government",
        value="Person A",
        domain=KnowledgeDomain.GOVERNANCE,
    )
    promote(
        updater,
        field="capital",
        value="Rome",
        domain=KnowledgeDomain.GOVERNANCE,
    )

    world.record_world_events(
        [
            WorldEventObservation(
                provider="gdelt",
                provider_event_id="evt-1",
                event_time=utc(26, 8),
                known_at=utc(26, 9),
                actor1_entity_id="country:ITA",
                actor2_entity_id="country:FRA",
                event_code="040",
                event_root_code="04",
                quad_class=1,
                num_sources=3,
                num_articles=5,
                source_url="https://news.example/event",
                snapshot_checksum="evt-snapshot",
            )
        ]
    )

    warehouse = HistoricalWarehouse(tmp_path / "history.duckdb")
    warehouse.insert_records(
        [
            TemporalRecord(
                record_id="wb-pop-test",
                entity_id="iso3:ITA",
                variable="world_bank:SP.POP.TOTL",
                value=58_900_000,
                valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
                valid_to=datetime(2026, 1, 1, tzinfo=timezone.utc),
                known_at=utc(21),
                evidence_class=EvidenceClass.OBSERVATION,
                provenance=Provenance(
                    source="world_bank",
                    source_ref="ITA:SP.POP.TOTL:2025",
                    retrieved_at=utc(21),
                ),
            )
        ]
    )

    news = SharedNewsStore(tmp_path / "news.duckdb")
    news.upsert_articles(
        provider="google_news_rss",
        articles=[
            {
                "title": "Italy policy update - Example News",
                "url": "https://news.google.com/rss/articles/italy-policy",
                "seendate": utc(26, 10).isoformat(),
                "publisher": "Example News",
                "domain": "example.com",
                "language": "en",
                "snippet": "Italy announced a policy update.",
            }
        ],
        retrieved_at=utc(26, 10),
        snapshot_checksum="news-snapshot",
        feed_id="world-politics-en",
    )
    return world, updater, warehouse, news


def test_country_profile_assembles_sourced_facts_local_events_news_and_indicators(tmp_path):
    world, _updater, warehouse, news = build_fixture(tmp_path)

    profile = assemble_country_intelligence_profile(
        "ITA",
        world=world,
        as_of=utc(27),
        known_cutoff=utc(27),
        warehouse=warehouse,
        news=news,
    )

    assert profile["identity"]["name"] == "Italy"
    assert profile["identity"]["capital"] == "Rome"

    government_form = next(
        item
        for item in profile["political_system"]["fields"]
        if item["key"] == "government_form"
    )
    assert government_form["status"] == "known"
    assert government_form["value"] == "Test parliamentary republic"
    assert government_form["evidence"][0]["authoritative"] is True
    assert government_form["semantic_class"] == "FACT"

    legislature = next(
        item
        for item in profile["political_system"]["fields"]
        if item["key"] == "legislature"
    )
    assert legislature["status"] == "unknown"
    assert "legislature" in profile["data_gaps"]["political_system"]

    head = next(
        item
        for item in profile["current_government"]["fields"]
        if item["key"] == "head_of_government"
    )
    assert head["value"] == "Person A"

    population = profile["socioeconomic_indicators"]["indicators"]["population"]
    assert population["status"] == "known"
    assert population["value"] == 58_900_000
    assert population["source"] == "world_bank"

    assert len(profile["recent_events"]["events"]) == 1
    assert (
        profile["recent_events"]["semantic_class"]
        == "PROVIDER_DERIVED_OBSERVATION"
    )
    assert len(profile["recent_news"]["articles"]) == 1
    assert profile["recent_news"]["match_mode"] == "country_name_text"
    assert profile["recent_news"]["entity_resolution"] == "not_yet_applied"

    contract = profile["semantic_contract"]
    assert contract["unknown_fields_remain_unknown"] is True
    assert contract["provider_events_are_not_promoted_facts"] is True
    assert contract["forecasts_are_separate"] is True


def test_country_profile_respects_knowledge_cutoff_across_facts_events_and_news(tmp_path):
    world, updater, warehouse, news = build_fixture(tmp_path)

    promote(
        updater,
        field="head_of_government",
        value="Person B",
        domain=KnowledgeDomain.GOVERNANCE,
        day=28,
    )
    world.record_world_events(
        [
            WorldEventObservation(
                provider="gdelt",
                provider_event_id="evt-later",
                event_time=utc(27, 15),
                known_at=utc(28, 8),
                actor1_entity_id="country:ITA",
                actor2_entity_id="country:DEU",
                event_code="040",
                snapshot_checksum="later-event",
            )
        ]
    )
    news.upsert_articles(
        provider="gdelt",
        articles=[
            {
                "title": "Italy later development",
                "url": "https://example.net/italy-later",
                "seendate": utc(27, 16).isoformat(),
                "domain": "example.net",
                "language": "English",
            }
        ],
        retrieved_at=utc(28, 8),
        snapshot_checksum="later-news",
        feed_id="world-gdelt-enrichment",
    )

    profile = assemble_country_intelligence_profile(
        "ITA",
        world=world,
        as_of=utc(27, 23),
        known_cutoff=utc(27, 23),
        warehouse=warehouse,
        news=news,
    )

    head = next(
        item
        for item in profile["current_government"]["fields"]
        if item["key"] == "head_of_government"
    )
    assert head["value"] == "Person A"
    assert {event["provider_event_id"] for event in profile["recent_events"]["events"]} == {
        "evt-1"
    }
    assert [row["title"] for row in profile["recent_news"]["articles"]] == [
        "Italy policy update - Example News"
    ]


def test_country_profile_rejects_unknown_iso3(tmp_path):
    world = WorldKnowledgeStore(tmp_path / "world.duckdb")
    try:
        assemble_country_intelligence_profile("ZZZ", world=world)
    except ValueError as exc:
        assert "Unknown ISO-3" in str(exc)
    else:
        raise AssertionError("Unknown ISO-3 code should be rejected")
