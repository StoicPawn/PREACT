"""Run the PREACT Geopolitical State Engine."""

from __future__ import annotations

import argparse
import json
import os

from preact.intelligence.geopolitical_state_cycle import (
    run_geopolitical_state_cycle,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("live", "canonical"),
        required=True,
    )
    args = parser.parse_args()

    result = run_geopolitical_state_cycle(
        world_knowledge_db=os.getenv(
            "PREACT_WORLD_KNOWLEDGE_DB",
            "data/history/world_knowledge.duckdb",
        ),
        shared_hub_root=os.getenv(
            "SHARED_DATA_HUB_ROOT",
            "data/shared_hub",
        ),
        graph_path=os.getenv(
            "PREACT_GRAPH_DB",
            "data/history/preact_graph.duckdb",
        ),
        mode=args.mode,
    )
    print(json.dumps(result.as_dict(), indent=2, default=str))
