from datetime import datetime, timezone
import json

from preact.data_hub.projection_ledger import ProjectionLedger
from preact.history.document_store import HistoricalDocumentStore
from preact.history.snapshot_store import SourceSnapshotStore
from preact.history.warehouse import HistoricalWarehouse
from preact.projections.shared_snapshots import SharedSnapshotProjector


UTC = timezone.utc


def test_goldenbull_gdelt_snapshot_can_be_projected_without_refetch(tmp_path) -> None:
    store = SourceSnapshotStore(tmp_path / "hub" / "snapshots")
    payload = json.dumps(
        {
            "articles": [
                {
                    "url": "https://example.test/a",
                    "title": "Shared evidence",
                    "seendate": "20200102T120000Z",
                    "sourcecountry": "US",
                }
            ]
        }
    ).encode("utf-8")
    snapshot = store.put(
        source_id="gdelt",
        payload=payload,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_url="https://api.gdeltproject.org/api/v2/doc/doc?q=x",
        operation="doc_artlist",
        content_type="application/json",
        request={"query": "x"},
    )
    projector = SharedSnapshotProjector(
        snapshot_store=store,
        ledger=ProjectionLedger(tmp_path / "ledger.sqlite3"),
        history=HistoricalWarehouse(tmp_path / "history.duckdb"),
        documents=HistoricalDocumentStore(tmp_path / "documents.duckdb"),
    )

    first = projector.project_gdelt_snapshot(snapshot)
    second = projector.project_gdelt_snapshot(snapshot)

    assert first["documents"] == 1
    assert second["documents"] == 0
    rows = projector.documents.as_of(
        cutoff=datetime(2020, 1, 3, tzinfo=UTC)
    )
    assert rows[0]["title"] == "Shared evidence"



def test_goldenbull_google_news_snapshot_can_be_projected_without_refetch(tmp_path) -> None:
    store = SourceSnapshotStore(tmp_path / "hub" / "snapshots")
    xml = """<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>Shared RSS evidence</title>
          <link>https://news.google.com/rss/articles/x</link>
          <pubDate>Sun, 20 Sep 2026 06:00:00 GMT</pubDate>
          <source url="https://example.com">Example News</source>
        </item>
      </channel>
    </rss>
    """.encode("utf-8")
    snapshot = store.put(
        source_id="google_news_rss",
        payload=xml,
        retrieved_at=datetime(2026, 9, 20, 6, 5, tzinfo=UTC),
        source_url="https://news.google.com/rss/search?q=x",
        operation="rss_search",
        content_type="application/rss+xml",
        request={"q": "x", "hl": "en-US", "gl": "US", "ceid": "US:en"},
    )
    projector = SharedSnapshotProjector(
        snapshot_store=store,
        ledger=ProjectionLedger(tmp_path / "ledger.sqlite3"),
        history=HistoricalWarehouse(tmp_path / "history.duckdb"),
        documents=HistoricalDocumentStore(tmp_path / "documents.duckdb"),
    )

    first = projector.project_google_news_snapshot(snapshot)
    second = projector.project_google_news_snapshot(snapshot)

    assert first["documents"] == 1
    assert second["documents"] == 0
    rows = projector.documents.as_of(
        cutoff=datetime(2026, 9, 20, 6, 1, tzinfo=UTC)
    )
    assert rows[0]["title"] == "Shared RSS evidence"
    assert rows[0]["source_id"] == "google_news_rss"
