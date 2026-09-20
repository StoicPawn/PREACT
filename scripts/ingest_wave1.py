"""Run the currently automated portion of PREACT Wave-1 ingestion."""

from __future__ import annotations

import argparse
import json
import os

from preact.history.wave1_runner import Wave1Runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hub-root",
        default=os.getenv("SHARED_DATA_HUB_ROOT", "data/shared_hub"),
    )
    parser.add_argument(
        "--ucdp-version",
        default=os.getenv("UCDP_DATASET_VERSION"),
        help="Explicit UCDP version, e.g. 26.1. Never infer latest in replay pipelines.",
    )
    parser.add_argument("--lightweight", action="store_true")
    args = parser.parse_args()

    runner = Wave1Runner(hub_root=args.hub_root)
    results = runner.run_available(
        ucdp_version=args.ucdp_version,
        lightweight=args.lightweight,
    )
    print(
        json.dumps(
            [
                {
                    "source_id": item.source_id,
                    "status": item.status,
                    "rows": item.rows,
                    "snapshots": item.snapshots,
                    "message": item.message,
                    "metadata": item.metadata,
                }
                for item in results
            ],
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
