"""Fan shared raw snapshots into PREACT's historical evidence stores."""

from __future__ import annotations

import json
from pathlib import Path

from preact.data_hub.gdelt_realtime import parse_event_zip
from preact.data_hub.projection_ledger import ProjectionLedger
from preact.history.document_store import HistoricalDocumentStore
from preact.history.snapshot_store import SnapshotMetadata, SourceSnapshotStore
from preact.history.warehouse import HistoricalWarehouse

from .gdelt_history import gdelt_article_documents, gdelt_event_records

GDELT_EVENTS_CONSUMER = "preact:gdelt_events:v1"
GDELT_DOCUMENTS_CONSUMER = "preact:gdelt_documents:v1"


class SharedSnapshotProjector:
    def __init__(
        self,
        *,
        snapshot_store: SourceSnapshotStore,
        ledger: ProjectionLedger,
        history: HistoricalWarehouse,
        documents: HistoricalDocumentStore,
    ) -> None:
        self.snapshot_store = snapshot_store
        self.ledger = ledger
        self.history = history
        self.documents = documents

    def project_gdelt_snapshot(self, snapshot: SnapshotMetadata) -> dict[str, int]:
        if snapshot.source_id != "gdelt":
            return {"records": 0, "documents": 0}

        if snapshot.operation == "realtime_events":
            consumer = GDELT_EVENTS_CONSUMER
            if self.ledger.seen(consumer, snapshot.snapshot_id):
                return {"records": 0, "documents": 0}
            payload = self.snapshot_store.read_payload(snapshot)
            rows = parse_event_zip(payload)
            records = gdelt_event_records(
                rows,
                acquired_at=snapshot.retrieved_at,
                snapshot_checksum=snapshot.checksum_sha256,
            )
            inserted = self.history.insert_records(records)
            self.ledger.mark(consumer, snapshot.snapshot_id)
            return {"records": inserted, "documents": 0}

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

    def project_pending_gdelt(self) -> dict[str, int]:
        totals = {"snapshots": 0, "records": 0, "documents": 0}
        for snapshot in self.snapshot_store.iter_metadata(source_id="gdelt"):
            if snapshot.operation not in {"realtime_events", "doc_artlist"}:
                continue
            result = self.project_gdelt_snapshot(snapshot)
            totals["snapshots"] += 1
            totals["records"] += result["records"]
            totals["documents"] += result["documents"]
        return totals
