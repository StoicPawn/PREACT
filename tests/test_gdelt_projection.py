from datetime import datetime, timezone

from preact.history.documents import TextAvailability
from preact.projections.gdelt_history import gdelt_article_documents, gdelt_event_records


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
