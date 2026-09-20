"""PREACT projection for shared Google News RSS snapshots."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Any, Iterable, Mapping

from preact.history.documents import HistoricalDocument, TextAvailability


def google_news_documents(
    articles: Iterable[Mapping[str, Any]],
    *,
    acquired_at: datetime,
    snapshot_checksum: str | None = None,
) -> list[HistoricalDocument]:
    documents: list[HistoricalDocument] = []
    for article in articles:
        url = str(article.get("url") or "").strip()
        title = str(article.get("title") or url or "Untitled").strip()
        raw_published = str(article.get("seendate") or "").strip()
        try:
            published_at = datetime.fromisoformat(raw_published.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            published_at = acquired_at
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=acquired_at.tzinfo)

        digest = sha256(
            (url or repr(sorted(article.items()))).encode("utf-8")
        ).hexdigest()
        documents.append(
            HistoricalDocument(
                document_id=f"google_news_rss:document:{digest}",
                source_id="google_news_rss",
                source_ref=url or digest,
                title=title,
                published_at=published_at,
                known_at=published_at,
                acquired_at=acquired_at,
                url=url or None,
                language=str(article.get("language") or "").strip() or None,
                text=None,
                text_availability=TextAvailability.METADATA_ONLY,
                snapshot_checksum=snapshot_checksum,
                attributes={
                    "domain": article.get("domain"),
                    "publisher": article.get("publisher"),
                    "social_image": article.get("socialimage"),
                    "snippet": article.get("snippet"),
                },
            )
        )
    return documents
