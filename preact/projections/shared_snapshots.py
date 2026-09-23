"""Fan shared raw snapshots into PREACT's historical evidence stores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from preact.data_hub.gdelt_realtime import parse_event_zip
from preact.data_hub.google_news import parse_google_news_rss
from preact.data_hub.projection_ledger import ProjectionLedger
from preact.history.document_store import HistoricalDocumentStore
from preact.history.graph_store import HistoricalGraphStore
from preact.history.snapshot_store import SnapshotMetadata, SourceSnapshotStore
from preact.history.warehouse import HistoricalWarehouse

from .gdelt_history import gdelt_article_documents, gdelt_event_records, gdelt_event_relations
from .google_news_history import google_news_documents

GDELT_EVENTS_CONSUMER = "preact:gdelt_events:v1"
GDELT_RELATIONS_CONSUMER = "preact:gdelt_relations:v1"
GDELT_DOCUMENTS_CONSUMER = "preact:gdelt_documents:v1"
GOOGLE_NEWS_DOCUMENTS_CONSUMER = "preact:google_news_documents:v1"


class SharedSnapshotProjector:
    def __init__(
        self,
        *,
        snapshot_store: SourceSnapshotStore,
        ledger: ProjectionLedger,
        history: HistoricalWarehouse,
        documents: HistoricalDocumentStore,
        graph: HistoricalGraphStore | None = None,
        fips_to_iso3: Mapping[str, str] | None = None,
    ) -> None:
        self.snapshot_store = snapshot_store
        self.ledger = ledger
        self.history = history
        self.documents = documents
        self.graph = graph
        self.fips_to_iso3 = dict(fips_to_iso3 or {})

    def project_gdelt_snapshot(self, snapshot: SnapshotMetadata) -> dict[str, int]:
        if snapshot.source_id != "gdelt":
            return {"records": 0, "documents": 0, "relations": 0}

        if snapshot.operation == "realtime_events":
            payload = self.snapshot_store.read_payload(snapshot)
            rows = parse_event_zip(payload)
            inserted_records = 0
            inserted_relations = 0

            if not self.ledger.seen(GDELT_EVENTS_CONSUMER, snapshot.snapshot_id):
                records = gdelt_event_records(
                    rows,
                    acquired_at=snapshot.retrieved_at,
                    snapshot_checksum=snapshot.checksum_sha256,
                    fips_to_iso3=self.fips_to_iso3,
                )
                inserted_records = self.history.insert_records(records)
                self.ledger.mark(GDELT_EVENTS_CONSUMER, snapshot.snapshot_id)

            if (
                self.graph is not None
                and not self.ledger.seen(GDELT_RELATIONS_CONSUMER, snapshot.snapshot_id)
            ):
                relations = gdelt_event_relations(
                    rows,
                    acquired_at=snapshot.retrieved_at,
                    snapshot_checksum=snapshot.checksum_sha256,
                    code_to_iso3=self.fips_to_iso3,
                )
                inserted_relations = self.graph.insert(relations)
                self.ledger.mark(GDELT_RELATIONS_CONSUMER, snapshot.snapshot_id)

            return {
                "records": inserted_records,
                "documents": 0,
                "relations": inserted_relations,
            }

        if snapshot.operation == "doc_artlist":
            consumer = GDELT_DOCUMENTS_CONSUMER
            if self.ledger.seen(consumer, snapshot.snapshot_id):
                return {"records": 0, "documents": 0}
            payload = json.loads(
                self.snapshot_store.read_payload(snapshot).decode("utf-8")
            )
            articles = payload.get("articles", []) if isinstance(payload, dict) else []
            documents = gdelt_article_documents(
                [item for item in articles if isinstance(item, dict)],
                acquired_at=snapshot.retrieved_at,
                snapshot_checksum=snapshot.checksum_sha256,
            )
            inserted = self.documents.insert(documents)
            self.ledger.mark(consumer, snapshot.snapshot_id)
            return {"records": 0, "documents": inserted}

        return {"records": 0, "documents": 0}

    def project_google_news_snapshot(
        self,
        snapshot: SnapshotMetadata,
    ) -> dict[str, int]:
        if snapshot.source_id != "google_news_rss" or snapshot.operation != "rss_search":
            return {"records": 0, "documents": 0}

        consumer = GOOGLE_NEWS_DOCUMENTS_CONSUMER
        if self.ledger.seen(consumer, snapshot.snapshot_id):
            return {"records": 0, "documents": 0}

        xml = self.snapshot_store.read_payload(snapshot).decode(
            "utf-8",
            errors="replace",
        )
        language = None
        request = dict(snapshot.request or {})
        hl = str(request.get("hl") or "")
        if hl:
            language = hl.split("-", 1)[0]
        articles = parse_google_news_rss(
            xml,
            max_records=200,
            language=language,
        )
        documents = google_news_documents(
            articles,
            acquired_at=snapshot.retrieved_at,
            snapshot_checksum=snapshot.checksum_sha256,
        )
        inserted = self.documents.insert(documents)
        self.ledger.mark(consumer, snapshot.snapshot_id)
        return {"records": 0, "documents": inserted}

    def project_pending_gdelt(self) -> dict[str, int]:
        totals = {"snapshots": 0, "records": 0, "documents": 0, "relations": 0}
        for snapshot in self.snapshot_store.iter_metadata(source_id="gdelt"):
            if snapshot.operation not in {"realtime_events", "doc_artlist"}:
                continue
            result = self.project_gdelt_snapshot(snapshot)
            totals["snapshots"] += 1
            totals["records"] += result["records"]
            totals["documents"] += result["documents"]
            totals["relations"] += result.get("relations", 0)
        return totals


    def project_pending_shared_news(self) -> dict[str, int]:
        totals = {"snapshots": 0, "records": 0, "documents": 0}
        for source_id in ("gdelt", "google_news_rss"):
            for snapshot in self.snapshot_store.iter_metadata(source_id=source_id):
                if source_id == "gdelt":
                    if snapshot.operation not in {"realtime_events", "doc_artlist"}:
                        continue
                    result = self.project_gdelt_snapshot(snapshot)
                else:
                    if snapshot.operation != "rss_search":
                        continue
                    result = self.project_google_news_snapshot(snapshot)
                totals["snapshots"] += 1
                totals["records"] += result["records"]
                totals["documents"] += result["documents"]
        return totals
