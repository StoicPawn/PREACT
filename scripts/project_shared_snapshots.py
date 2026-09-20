"""Project shared provider snapshots into PREACT without refetching providers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from preact.data_hub.projection_ledger import ProjectionLedger
from preact.history.country_codes import load_fips_to_iso3
from preact.history.document_store import HistoricalDocumentStore
from preact.history.snapshot_store import SourceSnapshotStore
from preact.history.warehouse import HistoricalWarehouse
from preact.projections.shared_snapshots import SharedSnapshotProjector


if __name__ == "__main__":
    hub_root = Path(os.getenv("SHARED_DATA_HUB_ROOT", "data/shared_hub"))
    history_db = os.getenv("PREACT_HISTORY_DB", "data/history/preact_history.duckdb")
    document_db = os.getenv("PREACT_DOCUMENT_DB", "data/history/preact_documents.duckdb")
    ledger_db = os.getenv(
        "PREACT_PROJECTION_LEDGER",
        "data/history/projection_ledger.sqlite3",
    )
    country_code_map = os.getenv(
        "PREACT_COUNTRY_CODE_MAP",
        "data/history/country_codes.json",
    )
    projector = SharedSnapshotProjector(
        snapshot_store=SourceSnapshotStore(hub_root / "snapshots"),
        ledger=ProjectionLedger(ledger_db),
        history=HistoricalWarehouse(history_db),
        documents=HistoricalDocumentStore(document_db),
        fips_to_iso3=load_fips_to_iso3(country_code_map),
    )
    print(json.dumps(projector.project_pending_gdelt(), indent=2))
