from datetime import datetime, timezone

from preact.data_hub.news_store import SharedNewsStore
from preact.data_hub.wikidata import (
    WikidataCountryReference,
    parse_country_reference_payload,
)
from preact.history.world_knowledge_store import WorldKnowledgeStore
from preact.intelligence.country_intelligence import assemble_country_intelligence_profile
from preact.intelligence.political_reference import refresh_political_reference


def t(hour: int) -> datetime:
    return datetime(2026, 9, 27, hour, tzinfo=timezone.utc)


def reference(
    *,
    hog: tuple[str, ...] = (),
    hos: tuple[str, ...] = (),
    retrieved_at: datetime,
) -> WikidataCountryReference:
    return WikidataCountryReference(
        iso3="ITA",
        qid="Q38",
        country_label="Italy",
        capital=("Rome",),
        government_forms=("parliamentary republic",),
        heads_of_state=hos,
        heads_of_government=hog,
        official_languages=("Italian",),
        official_websites=("https://www.italia.it/",),
        retrieved_at=retrieved_at,
        snapshot_checksum=f"snapshot-{retrieved_at.hour}",
    )


def test_wikidata_payload_is_collapsed_to_one_country_reference():
    payload = {
        "results": {
            "bindings": [
                {
                    "iso3": {"value": "ITA"},
                    "country": {"value": "http://www.wikidata.org/entity/Q38"},
                    "countryLabel": {"value": "Italy"},
                    "capitalLabel": {"value": "Rome"},
                    "governmentFormLabel": {"value": "parliamentary republic"},
                    "headOfGovernmentLabel": {"value": "Person A"},
                    "officialLanguageLabel": {"value": "Italian"},
                },
                {
                    "iso3": {"value": "ITA"},
                    "country": {"value": "http://www.wikidata.org/entity/Q38"},
                    "countryLabel": {"value": "Italy"},
                    "capitalLabel": {"value": "Rome"},
                    "governmentFormLabel": {"value": "parliamentary republic"},
                    "headOfStateLabel": {"value": "Person S"},
                    "officialLanguageLabel": {"value": "Italian"},
                },
            ]
        }
    }

    refs = parse_country_reference_payload(payload, retrieved_at=t(8), snapshot_checksum="abc")
    assert len(refs) == 1
    item = refs[0]
    assert item.iso3 == "ITA"
    assert item.qid == "Q38"
    assert item.capital == ("Rome",)
    assert item.heads_of_government == ("Person A",)
    assert item.heads_of_state == ("Person S",)
    assert item.source_ref.endswith("/Q38")


def test_reference_seeds_unknown_fields_but_does_not_overwrite_without_corroboration(tmp_path):
    store = WorldKnowledgeStore(tmp_path / "world.duckdb")

    first = refresh_political_reference(
        [reference(hog=("Person A",), retrieved_at=t(8))],
        store=store,
    )
    assert first.fields_seeded >= 5
    assert store.current_state("country:ITA")["head_of_government"]["value"] == "Person A"

    second = refresh_political_reference(
        [reference(hog=("Person B",), retrieved_at=t(10))],
        store=store,
    )
    assert second.change_candidates == 1
    assert second.promoted_changes == 0
    assert second.held_changes == 1
    assert store.current_state("country:ITA")["head_of_government"]["value"] == "Person A"


def test_reference_change_promotes_after_two_independent_news_groups(tmp_path):
    store = WorldKnowledgeStore(tmp_path / "world.duckdb")
    news = SharedNewsStore(tmp_path / "news.duckdb")

    refresh_political_reference(
        [reference(hog=("Person A",), retrieved_at=t(8))],
        store=store,
    )

    news.upsert_articles(
        provider="google_news_rss",
        retrieved_at=t(9),
        snapshot_checksum="news-a",
        feed_id="politics",
        articles=[
            {
                "title": "Italy appoints Person B as prime minister",
                "url": "https://publisher-a.example/story",
                "publisher": "Publisher A",
                "domain": "publisher-a.example",
                "published_at": t(9),
                "snippet": "Person B becomes Italy's prime minister.",
            }
        ],
    )
    news.upsert_articles(
        provider="gdelt",
        retrieved_at=datetime(2026, 9, 27, 9, 30, tzinfo=timezone.utc),
        snapshot_checksum="news-b",
        feed_id="politics",
        articles=[
            {
                "title": "Person B sworn in as Italy prime minister",
                "url": "https://publisher-b.example/story",
                "publisher": "Publisher B",
                "domain": "publisher-b.example",
                "published_at": datetime(2026, 9, 27, 9, 20, tzinfo=timezone.utc),
                "snippet": "Italy has sworn in Person B as prime minister.",
            }
        ],
    )

    result = refresh_political_reference(
        [reference(hog=("Person B",), retrieved_at=t(10))],
        store=store,
        news=news,
    )

    assert result.change_candidates == 1
    assert result.promoted_changes == 1
    current = store.current_state("country:ITA")
    assert current["head_of_government"]["value"] == "Person B"
    assert len(current["head_of_government"]["evidence"]) == 3


def test_country_profile_builds_prose_only_from_promoted_facts(tmp_path):
    store = WorldKnowledgeStore(tmp_path / "world.duckdb")
    refresh_political_reference(
        [reference(hog=("Person A",), hos=("Person S",), retrieved_at=t(8))],
        store=store,
    )

    profile = assemble_country_intelligence_profile(
        "ITA",
        world=store,
        as_of=t(9),
        known_cutoff=t(9),
    )

    descriptions = profile["descriptions"]
    assert descriptions["current_government"]["semantic_class"] == "FACT_DERIVED"
    assert "Person A" in descriptions["current_government"]["text"]
    assert "Person S" in descriptions["current_government"]["text"]
    assert "parliamentary republic" in descriptions["political_system"]["text"]
    assert "Rome" in descriptions["country_today"]["text"]
    assert descriptions["current_government"]["supporting_assertion_ids"]
