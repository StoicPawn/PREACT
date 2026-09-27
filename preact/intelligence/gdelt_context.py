"""PREACT consumer for GDELT Mentions and GKG snapshots owned by ACEPC Data Hub."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pycountry

from preact.history.connectors.gdelt_cameo import build_cameo_country_map
from preact.history.connectors.geonames import CountryCodeMap, GeoNamesCountryInfoConnector
from preact.history.snapshot_store import SourceSnapshotStore
from preact.intelligence.gdelt_ingest import (
    normalize_gkg_documents,
    parse_gkg_zip,
    parse_mentions_zip,
    summarize_mentions,
)


_GKG_FIPS_OVERRIDES = {
    "UK": "GBR", "GM": "DEU", "JA": "JPN", "KS": "KOR",
    "KN": "PRK", "CH": "CHN", "SP": "ESP", "SW": "SWE",
    "SZ": "CHE", "TU": "TUR", "AS": "AUS", "AU": "AUT",
    "PO": "PRT", "RO": "ROU", "BU": "BGR", "GR": "GRC",
    "HR": "HRV", "LO": "SVK", "SI": "SVN", "EZ": "CZE",
}


@dataclass(frozen=True)
class GDELTContextBatch:
    mention_observations: list[dict[str, Any]]
    gkg_documents: list[dict[str, Any]]
    mention_snapshot_count: int
    gkg_snapshot_count: int
    mention_raw_rows: int
    gkg_raw_rows: int
    mapping_snapshot_checksum: str | None
    processed_snapshots: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    return value.astimezone(timezone.utc)


def _latest_country_map(store: SourceSnapshotStore, cutoff: datetime):
    candidates = [
        item
        for item in store.iter_metadata(source_id="gdelt")
        if item.operation == "reference_cameo_country"
        and item.retrieved_at <= cutoff
    ]
    if not candidates:
        return None, None
    snapshot = candidates[-1]
    return (
        build_cameo_country_map(store.read_payload(snapshot)),
        snapshot.checksum_sha256,
    )


def _latest_gkg_country_map(
    store: SourceSnapshotStore,
    cutoff: datetime,
) -> tuple[CountryCodeMap | None, str | None]:
    candidates = [
        item
        for item in store.iter_metadata(source_id="geonames")
        if item.operation == "reference_country_codes"
        and item.retrieved_at <= cutoff
    ]
    if not candidates:
        return None, None
    snapshot = candidates[-1]
    rows = GeoNamesCountryInfoConnector.parse(store.read_payload(snapshot))
    return CountryCodeMap(rows), snapshot.checksum_sha256


def _resolve_gkg_country_code(
    code: str,
    mapping: CountryCodeMap | None,
) -> str | None:
    """Resolve GKG location codes with FIPS taking precedence over ISO-2."""
    key = str(code or "").strip().upper()
    if not key:
        return None
    if mapping is not None:
        if key in mapping.by_fips and mapping.by_fips[key].iso3:
            return mapping.by_fips[key].iso3
        if key in mapping.by_iso2 and mapping.by_iso2[key].iso3:
            return mapping.by_iso2[key].iso3
        if key in mapping.by_iso3 and mapping.by_iso3[key].iso3:
            return mapping.by_iso3[key].iso3
    override = _GKG_FIPS_OVERRIDES.get(key)
    if override:
        return override
    if len(key) == 3:
        direct = pycountry.countries.get(alpha_3=key)
        if direct is not None:
            return direct.alpha_3
    if len(key) == 2:
        alpha2 = pycountry.countries.get(alpha_2=key)
        if alpha2 is not None:
            return alpha2.alpha_3
    return None


def load_recent_gdelt_context(
    shared_hub_root: str | Path,
    *,
    as_of: datetime,
    lookback_days: int = 2,
    skip_snapshot_checksums: set[str] | None = None,
) -> GDELTContextBatch:
    """Load Mentions/GKG from external immutable snapshots without provider calls."""

    cutoff = _utc(as_of)
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    lower = cutoff - timedelta(days=int(lookback_days))
    store = SourceSnapshotStore(Path(shared_hub_root) / "snapshots")
    country_map, map_checksum = _latest_country_map(store, cutoff)
    gkg_country_map, gkg_map_checksum = _latest_gkg_country_map(store, cutoff)
    skipped = set(skip_snapshot_checksums or set())

    mention_snapshots = [
        item
        for item in store.iter_metadata(source_id="gdelt")
        if item.operation == "realtime_mentions"
        and lower <= item.retrieved_at <= cutoff
        and item.checksum_sha256 not in skipped
    ]
    gkg_snapshots = [
        item
        for item in store.iter_metadata(source_id="gdelt")
        if item.operation == "realtime_gkg"
        and lower <= item.retrieved_at <= cutoff
        and item.checksum_sha256 not in skipped
    ]

    processed_snapshots: list[dict[str, Any]] = []
    mention_observations: list[dict[str, Any]] = []
    mention_raw_rows = 0
    for snapshot in mention_snapshots:
        rows = parse_mentions_zip(store.read_payload(snapshot))
        processed_snapshots.append(
            {
                "snapshot_checksum": snapshot.checksum_sha256,
                "operation": "realtime_mentions",
                "retrieved_at": snapshot.retrieved_at,
            }
        )
        mention_raw_rows += len(rows)
        for summary in summarize_mentions(rows):
            summary["known_at"] = snapshot.retrieved_at
            summary["snapshot_checksum"] = snapshot.checksum_sha256
            mention_observations.append(summary)
    gkg_documents: list[dict[str, Any]] = []
    gkg_raw_rows = 0
    for snapshot in gkg_snapshots:
        rows = parse_gkg_zip(store.read_payload(snapshot))
        processed_snapshots.append(
            {
                "snapshot_checksum": snapshot.checksum_sha256,
                "operation": "realtime_gkg",
                "retrieved_at": snapshot.retrieved_at,
            }
        )
        gkg_raw_rows += len(rows)
        for item in normalize_gkg_documents(rows):
            raw_codes = list(item.get("country_codes") or [])
            iso3: set[str] = set()
            for code in raw_codes:
                resolved = _resolve_gkg_country_code(code, gkg_country_map)
                if resolved is None and country_map is not None:
                    resolved = country_map.resolve(code)
                if resolved:
                    iso3.add(resolved)
            item["country_iso3"] = sorted(iso3)
            item["country_mapping_checksum"] = gkg_map_checksum or map_checksum
            item["known_at"] = snapshot.retrieved_at
            item["snapshot_checksum"] = snapshot.checksum_sha256
            gkg_documents.append(item)

    return GDELTContextBatch(
        mention_observations=mention_observations,
        gkg_documents=gkg_documents,
        mention_snapshot_count=len(mention_snapshots),
        gkg_snapshot_count=len(gkg_snapshots),
        mention_raw_rows=mention_raw_rows,
        gkg_raw_rows=gkg_raw_rows,
        mapping_snapshot_checksum=map_checksum,
        processed_snapshots=processed_snapshots,
    )


__all__ = ["GDELTContextBatch", "load_recent_gdelt_context"]
