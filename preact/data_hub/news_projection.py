"""Project provider snapshots into the provider-neutral Shared News archive."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping

from preact.data_hub.google_news import parse_google_news_rss
from preact.data_hub.news_store import SharedNewsStore
from preact.data_hub.projection_ledger import ProjectionLedger
from preact.history.snapshot_store import SnapshotMetadata, SourceSnapshotStore


GOOGLE_NEWS_ARCHIVE_CONSUMER = "shared-news:google-rss:v1"
GDELT_DOC_ARCHIVE_CONSUMER = "shared-news:gdelt-doc:v1"


def _feed_id(provider: str, request: Mapping[str, Any]) -> str:
    query = str(request.get("q") or request.get("query") or "").strip()
    locale = "|".join(
        str(request.get(key) or "").strip()
        for key in ("hl", "gl", "ceid", "timespan")
    )
    digest = sha256(f"{provider}|{query}|{locale}".encode("utf-8")).hexdigest()[:16]
    return f"{provider}:{digest}"


class SharedNewsProjector:
    def __init__(
        self,
        *,
        snapshot_store: SourceSnapshotStore,
        ledger: ProjectionLedger,
        news: SharedNewsStore,
    ) -> None:
        self.snapshot_store = snapshot_store
        self.ledger = ledger
        self.news = news

    def project_google_news(self, snapshot: SnapshotMetadata) -> dict[str, int]:
        if snapshot.source_id != "google_news_rss" or snapshot.operation != "rss_search":
            return {"articles": 0, "observations": 0}
        if self.ledger.seen(GOOGLE_NEWS_ARCHIVE_CONSUMER, snapshot.snapshot_id):
            return {"articles": 0, "observations": 0}

        request = dict(snapshot.request or {})
        hl = str(request.get("hl") or "")
        language = hl.split("-", 1)[0] if hl else None
        xml = self.snapshot_store.read_payload(snapshot).decode("utf-8", errors="replace")
        articles = parse_google_news_rss(xml, max_records=500, language=language)
        result = self.news.upsert_articles(
            provider="google_news_rss",
            articles=articles,
            retrieved_at=snapshot.retrieved_at,
            snapshot_checksum=snapshot.checksum_sha256,
            feed_id=_feed_id("google_news_rss", request),
        )
        self.ledger.mark(GOOGLE_NEWS_ARCHIVE_CONSUMER, snapshot.snapshot_id)
        return {
            "articles": result["inserted_articles"],
            "observations": result["inserted_observations"],
        }

    def project_gdelt_doc(self, snapshot: SnapshotMetadata) -> dict[str, int]:
        if snapshot.source_id != "gdelt" or snapshot.operation != "doc_artlist":
            return {"articles": 0, "observations": 0}
        if self.ledger.seen(GDELT_DOC_ARCHIVE_CONSUMER, snapshot.snapshot_id):
            return {"articles": 0, "observations": 0}

        payload = json.loads(
            self.snapshot_store.read_payload(snapshot).decode("utf-8", errors="replace")
        )
        raw = payload.get("articles", []) if isinstance(payload, dict) else []
        articles = [item for item in raw if isinstance(item, dict)]
        result = self.news.upsert_articles(
            provider="gdelt",
            articles=articles,
            retrieved_at=snapshot.retrieved_at,
            snapshot_checksum=snapshot.checksum_sha256,
            feed_id=_feed_id("gdelt", dict(snapshot.request or {})),
        )
        self.ledger.mark(GDELT_DOC_ARCHIVE_CONSUMER, snapshot.snapshot_id)
        return {
            "articles": result["inserted_articles"],
            "observations": result["inserted_observations"],
        }

    def project_pending(self) -> dict[str, int]:
        totals = {
            "snapshots_examined": 0,
            "inserted_articles": 0,
            "inserted_observations": 0,
        }
        for source_id in ("google_news_rss", "gdelt"):
            for snapshot in self.snapshot_store.iter_metadata(source_id=source_id):
                if source_id == "google_news_rss":
                    if snapshot.operation != "rss_search":
                        continue
                    result = self.project_google_news(snapshot)
                else:
                    if snapshot.operation != "doc_artlist":
                        continue
                    result = self.project_gdelt_doc(snapshot)

                totals["snapshots_examined"] += 1
                totals["inserted_articles"] += result["articles"]
                totals["inserted_observations"] += result["observations"]
        return totals


__all__ = ["SharedNewsProjector"]
