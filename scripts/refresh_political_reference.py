"""Refresh structured current political reference facts into World Knowledge."""

from __future__ import annotations

import json
import os
from pathlib import Path

from preact.data_hub.gateway import SharedProviderGateway
from preact.data_hub.news_store import SharedNewsStore
from preact.data_hub.wikidata import fetch_country_references
from preact.history.world_knowledge_store import WorldKnowledgeStore
from preact.intelligence.political_reference import refresh_political_reference


if __name__ == "__main__":
    hub_root = Path(os.getenv("SHARED_DATA_HUB_ROOT", "data/shared_hub"))
    world_db = Path(
        os.getenv("PREACT_WORLD_KNOWLEDGE_DB", "data/history/world_knowledge.duckdb")
    )
    news_db = Path(
        os.getenv("SHARED_NEWS_DB", str(hub_root / "shared_news.duckdb"))
    )

    gateway = SharedProviderGateway(hub_root)
    references, provider = fetch_country_references(
        gateway,
        ttl_seconds=max(0, int(os.getenv("WIKIDATA_POLITICAL_TTL_SECONDS", "7200"))),
    )
    result = refresh_political_reference(
        references,
        store=WorldKnowledgeStore(world_db),
        news=SharedNewsStore(news_db) if news_db.exists() else None,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "provider": "wikidata",
                "retrieved_at": provider.retrieved_at.isoformat(),
                "cached": provider.cached,
                "snapshot_checksum": provider.snapshot_checksum,
                **result.as_dict(),
            },
            indent=2,
            default=str,
        )
    )
