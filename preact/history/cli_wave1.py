"""Canonical CLI implementation for PREACT Wave-1 ingestion."""

from __future__ import annotations

import argparse
import json
import os

from .wave1_runner import Wave1Runner


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--hub-root",default=os.getenv("SHARED_DATA_HUB_ROOT","data/shared_hub"))
    parser.add_argument("--history-db",default=os.getenv("PREACT_HISTORY_DB","data/history/preact_history.duckdb"))
    parser.add_argument("--graph-db",default=os.getenv("PREACT_GRAPH_DB","data/history/preact_graph.duckdb"))
    parser.add_argument("--state-db",default=os.getenv("PREACT_INGESTION_STATE","data/history/ingestion_state.sqlite3"))
    parser.add_argument("--ucdp-version",default=os.getenv("UCDP_DATASET_VERSION"))
    parser.add_argument("--lightweight",action="store_true")
    args=parser.parse_args()

    runner=Wave1Runner(
        hub_root=args.hub_root,
        history_db=args.history_db,
        graph_db=args.graph_db,
        state_db=args.state_db,
    )
    results=runner.run_available(
        ucdp_version=args.ucdp_version,
        lightweight=args.lightweight,
    )
    print(json.dumps([{
        "source_id":x.source_id,
        "status":x.status,
        "rows":x.rows,
        "snapshots":x.snapshots,
        "message":x.message,
        "metadata":x.metadata,
    } for x in results],indent=2,default=str))
