"""Build current provider-code crosswalks from a snapshotted GeoNames release."""

from __future__ import annotations

import json
import os
from pathlib import Path

from preact.history.connectors.base import BulkFileConnector
from preact.history.connectors.geonames import (
    CountryCodeMap,
    GeoNamesCountryInfoConnector,
)
from preact.history.snapshot_store import SourceSnapshotStore


if __name__ == "__main__":
    hub_root = Path(os.getenv("SHARED_DATA_HUB_ROOT", "data/shared_hub"))
    output = Path(
        os.getenv("PREACT_COUNTRY_CODE_MAP", "data/history/country_codes.json")
    )
    store = SourceSnapshotStore(hub_root / "snapshots")
    acquired = GeoNamesCountryInfoConnector(
        BulkFileConnector("geonames", store)
    ).acquire()
    rows = GeoNamesCountryInfoConnector.parse(acquired.payload)
    mapping = CountryCodeMap(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "source": "geonames",
                "snapshot_checksum": acquired.snapshot.checksum_sha256,
                "retrieved_at": acquired.retrieved_at.isoformat(),
                "fips_to_iso3": mapping.fips_to_iso3(),
                "countries": [
                    {
                        "iso2": row.iso2,
                        "iso3": row.iso3,
                        "fips": row.fips,
                        "country": row.country,
                        "geoname_id": row.geoname_id,
                    }
                    for row in rows
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)
