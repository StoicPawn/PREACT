from __future__ import annotations

from datetime import datetime, timezone
import json

from preact.data_hub.news_projection import SharedNewsProjector
from preact.data_hub.news_store import SharedNewsStore
from preact.data_hub.projection_ledger import ProjectionLedger
from preact.history.snapshot_store import SourceSnapshotStore


def utc(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, 27, hour, minute, tzinfo=timezone.utc)


def test_shared_news_deduplicates_same_story_across_google_and_gdelt(tmp_path):
    store = SharedNewsStore(tmp_path / "news.duckdb")

    google = {
        "title": "Example headline - Reuters",
        "url": "https://news.google.com/rss/articles/google-id",
        "seendate": "2026-09-27T08:00:00+00:00",
        "publisher": "Reuters",
        "domain": "reuters.com",
        "language": "en",
        "snippet": "First summary",
    }
    gdelt = {
        "title": "Example headline",
        "url": "https://www.reuters.com/world/example",
        "seendate": "20260927T080000Z",
        "domain": "reuters.com",
        "language": "English",
        "snippet": "Second summary",
    }

    first = store.upsert_articles(
        provider="google_news_rss",
        articles=[google],
        retrieved_at=utc(8, 5),
        snapshot_checksum="g1",
        feed_id="google-feed",
    )
    second = store.upsert_articles(
        provider="gdelt",
        articles=[gdelt],
        retrieved_at=utc(8, 10),
        snapshot_checksum="d1",
        feed_id="gdelt-feed",
    )

    assert first == {"inserted_articles": 1, "inserted_observations": 1}
    assert second == {"inserted_articles": 0, "inserted_observations": 1}

    stats = store.stats()
    assert stats["articles"] == 1
    assert stats["observations"] == 2

    rows = store.latest(limit=10)
    assert len(rows) == 1
    assert rows[0]["observation_count"] == 2
    assert rows[0]["provider"] == "gdelt"
    assert rows[0]["url"] == "https://www.reuters.com/world/example"


def test_shared_news_known_cutoff_does_not_leak_later_metadata(tmp_path):
    store = SharedNewsStore(tmp_path / "pit-news.duckdb")

    first = {
        "title": "Government announces reform - Example News",
        "url": "https://news.google.com/rss/articles/x",
        "seendate": "2026-09-27T08:00:00+00:00",
        "publisher": "Example News",
        "domain": "example.com",
        "language": "en",
        "snippet": "Initial report",
    }
    later = {
        "title": "Government announces reform",
        "url": "https://example.com/reform",
        "seendate": "2026-09-27T08:00:00+00:00",
        "domain": "example.com",
        "language": "en",
        "snippet": "Later enriched report with more detail",
    }

    store.upsert_articles(
        provider="google_news_rss",
        articles=[first],
        retrieved_at=utc(8, 5),
        snapshot_checksum="early",
        feed_id="feed-a",
    )
    store.upsert_articles(
        provider="gdelt",
        articles=[later],
        retrieved_at=utc(9),
        snapshot_checksum="late",
        feed_id="feed-b",
    )

    early = store.latest(known_cutoff=utc(8, 30))
    assert len(early) == 1
    assert early[0]["provider"] == "google_news_rss"
    assert early[0]["snippet"] == "Initial report"
    assert early[0]["last_seen_at"] == utc(8, 5)
    assert early[0]["observation_count"] == 1

    current = store.latest()
    assert current[0]["provider"] == "gdelt"
    assert current[0]["snippet"] == "Later enriched report with more detail"
    assert current[0]["last_seen_at"] == utc(9)
    assert current[0]["observation_count"] == 2


def test_shared_news_projector_reads_snapshots_without_provider_calls(tmp_path):
    root = tmp_path / "hub"
    snapshots = SourceSnapshotStore(root / "snapshots")

    xml = """<?xml version="1.0"?>
    <rss version="2.0"><channel><item>
      <title>Diplomatic meeting - Example News</title>
      <link>https://news.google.com/rss/articles/a</link>
      <pubDate>Sun, 27 Sep 2026 08:00:00 GMT</pubDate>
      <source url="https://example.com">Example News</source>
      <description>Summary</description>
    </item></channel></rss>
    """
    snapshots.put(
        source_id="google_news_rss",
        payload=xml.encode("utf-8"),
        retrieved_at=utc(8, 5),
        source_url="https://news.google.com/rss/search?q=x",
        operation="rss_search",
        request={"q": "diplomacy", "hl": "en-US", "gl": "US", "ceid": "US:en"},
    )

    gdelt_payload = {
        "articles": [
            {
                "title": "Market update",
                "url": "https://example.net/market",
                "seendate": "20260927T081000Z",
                "domain": "example.net",
                "language": "English",
            }
        ]
    }
    snapshots.put(
        source_id="gdelt",
        payload=json.dumps(gdelt_payload).encode("utf-8"),
        retrieved_at=utc(8, 15),
        source_url="https://api.gdeltproject.org/api/v2/doc/doc?query=market",
        operation="doc_artlist",
        request={"query": "market", "timespan": "1d"},
    )

    store = SharedNewsStore(root / "shared_news.duckdb")
    projector = SharedNewsProjector(
        snapshot_store=snapshots,
        ledger=ProjectionLedger(root / "projection.sqlite3"),
        news=store,
    )

    first = projector.project_pending()
    assert first["inserted_articles"] == 2
    assert first["inserted_observations"] == 2
    assert len(store.latest()) == 2

    second = projector.project_pending()
    assert second["inserted_articles"] == 0
    assert second["inserted_observations"] == 0
