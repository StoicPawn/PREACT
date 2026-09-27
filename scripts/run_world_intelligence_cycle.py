"""Project shared ACEPC snapshots into PREACT World Intelligence."""

from __future__ import annotations

import json
import os

from preact.intelligence.world_cycle import run_world_intelligence_cycle


if __name__ == "__main__":
    result = run_world_intelligence_cycle(
        shared_hub_root=os.getenv("SHARED_DATA_HUB_ROOT", "data/shared_hub"),
        world_knowledge_db=os.getenv(
            "PREACT_WORLD_KNOWLEDGE_DB",
            "data/history/world_knowledge.duckdb",
        ),
        lookback_days=max(1, int(os.getenv("PREACT_WORLD_LOOKBACK_DAYS", "2"))),
    )
    print(json.dumps(result.as_dict(), indent=2, default=str))
