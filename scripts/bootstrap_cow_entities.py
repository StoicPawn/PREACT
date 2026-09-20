"""Build the initial historical polity registry from COW State System v2024."""

from __future__ import annotations

import json
import os
from pathlib import Path

from preact.history.connectors.base import BulkFileConnector
from preact.history.connectors.cow import COWStateSystemConnector
from preact.history.snapshot_store import SourceSnapshotStore


if __name__ == "__main__":
    hub_root = Path(os.getenv("SHARED_DATA_HUB_ROOT", "data/shared_hub"))
    output = Path(
        os.getenv("PREACT_COW_ENTITY_REGISTRY", "data/history/cow_entities_v2024.json")
    )
    store = SourceSnapshotStore(hub_root / "snapshots")
    connector = COWStateSystemConnector(BulkFileConnector("cow", store))
    acquired = connector.acquire()
    entities = connector.to_entities(connector.parse_rows(acquired.payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "source": "cow",
                "release": "State System Membership v2024",
                "snapshot_checksum": acquired.snapshot.checksum_sha256,
                "retrieved_at": acquired.retrieved_at.isoformat(),
                "strict_replay_eligible_before_retrieval": (
                    acquired.replay_eligible_before_retrieval
                ),
                "entities": [
                    {
                        "entity_id": entity.entity_id,
                        "name": entity.name,
                        "valid_from": entity.valid_from.isoformat(),
                        "valid_to": entity.valid_to.isoformat()
                        if entity.valid_to
                        else None,
                        "codes": [
                            {"namespace": code.namespace, "value": code.value}
                            for code in entity.codes
                        ],
                        "aliases": list(entity.aliases),
                        "predecessors": list(entity.predecessors),
                        "successors": list(entity.successors),
                        "attributes": dict(entity.attributes),
                    }
                    for entity in entities
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(output)
