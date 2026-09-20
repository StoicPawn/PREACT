from datetime import datetime, timezone

from preact.projections.google_news_history import google_news_documents


UTC = timezone.utc


def test_google_news_projection_keeps_metadata_and_provenance() -> None:
    docs = google_news_documents(
        [
            {
                "url": "https://news.google.com/rss/articles/x",
                "title": "Example",
                "seendate": "2026-09-20T06:00:00+00:00",
                "publisher": "Example News",
                "domain": "example.com",
                "language": "en",
                "snippet": "Summary",
            }
        ],
        acquired_at=datetime(2026, 9, 20, 6, 5, tzinfo=UTC),
        snapshot_checksum="abc",
    )
    assert docs[0].source_id == "google_news_rss"
    assert docs[0].known_at.isoformat().startswith("2026-09-20T06:00:00")
    assert docs[0].snapshot_checksum == "abc"
    assert docs[0].attributes["publisher"] == "Example News"
