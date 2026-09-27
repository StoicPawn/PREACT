"""Run bounded local semantic review for material geopolitical events."""

from __future__ import annotations

import json
import os

from preact.intelligence.semantic_enrichment_cycle import (
    run_semantic_enrichment_cycle,
)


if __name__ == "__main__":
    result = run_semantic_enrichment_cycle(
        world_knowledge_db=os.getenv(
            "PREACT_WORLD_KNOWLEDGE_DB",
            "data/history/world_knowledge.duckdb",
        ),
        max_events=max(
            1,
            int(os.getenv("PREACT_SEMANTIC_MAX_EVENTS", "3")),
        ),
        lookback_days=max(
            1,
            int(os.getenv("PREACT_SEMANTIC_LOOKBACK_DAYS", "3")),
        ),
        model=os.getenv("PREACT_SEMANTIC_MODEL", "qwen3:1.7b"),
    )
    print(json.dumps(result.as_dict(), indent=2, default=str))
