"""Archive the latest GDELT 2.0 raw update files once."""

from __future__ import annotations

import json
import os

from preact.data_hub.gdelt_realtime import GDELTRealtimeCollector


if __name__ == "__main__":
    root = os.getenv("SHARED_DATA_HUB_ROOT", "data/shared_hub")
    collector = GDELTRealtimeCollector(root)
    country_lookup = collector.ensure_country_lookup()
    collected = collector.collect_latest()
    print(
        json.dumps(
            {
                "country_lookup": {
                    "checksum": country_lookup.checksum_sha256,
                    "retrieved_at": country_lookup.retrieved_at.isoformat(),
                },
                "files": [
                    {
                        "kind": item.ref.kind,
                        "url": item.ref.url,
                        "checksum": item.snapshot.checksum_sha256,
                    }
                    for item in collected
                ],
            },
            indent=2,
        )
    )
