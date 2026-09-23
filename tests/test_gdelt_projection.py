from datetime import datetime, timezone

from preact.history.documents import TextAvailability
from preact.projections.gdelt_history import (
    gdelt_article_documents,
    gdelt_event_records,
    gdelt_event_relations,
)


UTC = timezone.utc


def test_gdelt_event_projection_preserves_event_and_knowledge_time() -> None:
    records = gdelt_event_records(
        [
            {
                "GLOBALEVENTID": "123",
                "SQLDATE": "20200101",
                "DATEADDED": "20200102123000",
                "ActionGeo_CountryCode": "IT",
                "EventCode": "190",
                "SOURCEURL": "https://example.test/story",
            }
        ],
        acquired_at=datetime(2026, 1, 1, tzinfo=UTC),
        snapshot_checksum="abc",
        fips_to_iso3={"IT": "ITA"},
    )
    record = records[0]
    assert record.valid_from.isoformat().startswith("2020-01-01")
    assert record.known_at.isoformat().startswith("2020-01-02T12:30")
    assert record.entity_id == "iso3:ITA"
    assert record.attributes["snapshot_checksum"] == "abc"


def test_gdelt_article_projection_is_metadata_only() -> None:
    docs = gdelt_article_documents(
        [
            {
                "url": "https://example.test/story",
                "title": "Example",
                "seendate": "20200102T120000Z",
                "language": "English",
                "sourcecountry": "US",
            }
        ],
        acquired_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert docs[0].text is None
    assert docs[0].text_availability is TextAvailability.METADATA_ONLY
    assert docs[0].known_at.year == 2020


def test_gdelt_event_projection_builds_temporary_country_relation() -> None:
    relations = gdelt_event_relations(
        [
            {
                "GLOBALEVENTID": "999",
                "SQLDATE": "20200101",
                "DATEADDED": "20200101120000",
                "Actor1CountryCode": "IT",
                "Actor2CountryCode": "FR",
                "Actor1Name": "Italy",
                "Actor2Name": "France",
                "QuadClass": "4",
                "GoldsteinScale": "-7.0",
                "AvgTone": "-4.0",
                "NumArticles": "12",
                "SOURCEURL": "https://example.test/interaction",
            }
        ],
        acquired_at=datetime(2026, 1, 1, tzinfo=UTC),
        snapshot_checksum="snap",
        code_to_iso3={"IT": "ITA", "FR": "FRA"},
    )

    assert len(relations) == 1
    relation = relations[0]
    assert relation.subject_entity_id == "iso3:ITA"
    assert relation.object_entity_id == "iso3:FRA"
    assert relation.relation_type == "gdelt_material_conflict"
    assert relation.directed is True
    assert relation.valid_to is not None
    assert (relation.valid_to - relation.valid_from).days == 1
    assert relation.attributes["goldstein_scale"] == "-7.0"
    assert relation.attributes["snapshot_checksum"] == "snap"
