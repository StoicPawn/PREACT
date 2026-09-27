"""Scheduled shared news feeds for the ACEPC data hub.

These feeds are acquisition concerns, not product semantics. Products consume the
local SharedNewsStore and may later register additional shared feed definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from preact.data_hub.gateway import SharedProviderGateway
from preact.data_hub.gdelt import gdelt_doc_articles
from preact.data_hub.google_news import google_news_search
from preact.data_hub.news_projection import SharedNewsProjector
from preact.data_hub.news_store import SharedNewsStore
from preact.data_hub.projection_ledger import ProjectionLedger
from preact.history.snapshot_store import SourceSnapshotStore


@dataclass(frozen=True)
class SharedNewsFeed:
    feed_id: str
    provider: str
    query: str
    max_records: int
    ttl_seconds: int
    hl: str | None = None
    gl: str | None = None
    ceid: str | None = None
    timespan: str | None = None


DEFAULT_SHARED_NEWS_FEEDS: tuple[SharedNewsFeed, ...] = (
    SharedNewsFeed(
        feed_id="markets-global-en",
        provider="google_news_rss",
        query='("stock market" OR "Wall Street" OR "S&P 500" OR Nasdaq OR "Dow Jones") when:1d',
        max_records=80,
        ttl_seconds=900,
        hl="en-US",
        gl="US",
        ceid="US:en",
    ),
    SharedNewsFeed(
        feed_id="markets-italy-it",
        provider="google_news_rss",
        query='("Piazza Affari" OR "FTSE MIB" OR "Borsa Italiana") when:1d',
        max_records=80,
        ttl_seconds=900,
        hl="it",
        gl="IT",
        ceid="IT:it",
    ),
    SharedNewsFeed(
        feed_id="world-politics-en",
        provider="google_news_rss",
        query='(government OR parliament OR election OR sanctions OR diplomacy OR conflict OR ceasefire) when:1d',
        max_records=100,
        ttl_seconds=900,
        hl="en-US",
        gl="US",
        ceid="US:en",
    ),
    SharedNewsFeed(
        feed_id="world-politics-it",
        provider="google_news_rss",
        query='(governo OR parlamento OR elezioni OR sanzioni OR diplomazia OR conflitto OR tregua) when:1d',
        max_records=100,
        ttl_seconds=900,
        hl="it",
        gl="IT",
        ceid="IT:it",
    ),
    SharedNewsFeed(
        feed_id="world-gdelt-enrichment",
        provider="gdelt",
        query='(sanctions OR diplomacy OR election OR parliament OR ceasefire OR conflict)',
        max_records=100,
        ttl_seconds=7200,
        timespan="1d",
    ),
    SharedNewsFeed(
        feed_id="markets-gdelt-enrichment",
        provider="gdelt",
        query='("Wall Street" OR "stock market" OR "Piazza Affari")',
        max_records=80,
        ttl_seconds=7200,
        timespan="1d",
    ),
)


def refresh_shared_news(
    root: str | Path,
    *,
    feeds: tuple[SharedNewsFeed, ...] = DEFAULT_SHARED_NEWS_FEEDS,
) -> dict[str, Any]:
    """Refresh configured feeds through the shared gateway, then project locally."""

    root = Path(root)
    gateway = SharedProviderGateway(root)
    results: list[dict[str, Any]] = []

    for feed in feeds:
        try:
            if feed.provider == "google_news_rss":
                response = google_news_search(
                    gateway,
                    query=feed.query,
                    hl=feed.hl or "en-US",
                    gl=feed.gl or "US",
                    ceid=feed.ceid or "US:en",
                    max_records=feed.max_records,
                    ttl_seconds=feed.ttl_seconds,
                )
            elif feed.provider == "gdelt":
                response = gdelt_doc_articles(
                    gateway,
                    query=feed.query,
                    timespan=feed.timespan or "1d",
                    max_records=feed.max_records,
                    ttl_seconds=feed.ttl_seconds,
                )
            else:
                raise ValueError(f"Unsupported shared news provider: {feed.provider}")

            payload = response.payload if isinstance(response.payload, dict) else {}
            results.append(
                {
                    "feed_id": feed.feed_id,
                    "provider": feed.provider,
                    "status": "ready",
                    "cached": response.cached,
                    "retrieved_at": response.retrieved_at.isoformat(),
                    "snapshot_checksum": response.snapshot_checksum,
                    "articles": len(payload.get("articles", [])),
                }
            )
        except Exception as exc:
            # One unstable provider/feed must not suppress the rest of the shared archive.
            results.append(
                {
                    "feed_id": feed.feed_id,
                    "provider": feed.provider,
                    "status": "degraded",
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                }
            )

    projector = SharedNewsProjector(
        snapshot_store=SourceSnapshotStore(root / "snapshots"),
        ledger=ProjectionLedger(root / "shared_news_projection.sqlite3"),
        news=SharedNewsStore(root / "shared_news.duckdb"),
    )
    projected = projector.project_pending()
    stats = projector.news.stats()

    return {
        "feeds": results,
        "projection": projected,
        "archive": stats,
        "feed_definitions": [asdict(feed) for feed in feeds],
    }


__all__ = [
    "DEFAULT_SHARED_NEWS_FEEDS",
    "SharedNewsFeed",
    "refresh_shared_news",
]
