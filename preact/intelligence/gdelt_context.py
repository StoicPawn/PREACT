"""PREACT consumer for GDELT Mentions and GKG snapshots owned by ACEPC Data Hub."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from preact.history.connectors.gdelt_cameo import build_cameo_country_map
from preact.history.snapshot_store import SourceSnapshotStore
from preact.intelligence.gdelt_ingest import (
    normalize_gkg_documents,
    parse_gkg_zip,
    parse_mentions_zip,
    summarize_mentions,
)


@dataclass(frozen=True)
class GDELTContextBatch:
    mention_observations: list[dict[str, Any]]
    gkg_documents: list[dict[str, Any]]
    mention_snapshot_count: int
    gkg_snapshot_count: int
    mention_raw_rows: int
    gkg_raw_rows: int
    mapping_snapshot_checksum: str | None

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


def load_recent_gdelt_context(
    shared_hub_root: str | Path,
    *,
    as_of: datetime,
    lookback_days: int = 2,
) -> GDELTContextBatch:
    """Load Mentions/GKG from external immutable snapshots without provider calls."""

    cutoff = _utc(as_of)
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    lower = cutoff - timedelta(days=int(lookback_days))
    store = SourceSnapshotStore(Path(shared_hub_root) / "snapshots")
    country_map, map_checksum = _latest_country_map(store, cutoff)

    mention_snapshots = [
        item
        for item in store.iter_metadata(source_id="gdelt")
        if item.operation == "realtime_mentions"
        and lower <= item.retrieved_at <= cutoff
    ]
    gkg_snapshots = [
        item
        for item in store.iter_metadata(source_id="gdelt")
        if item.operation == "realtime_gkg"
        and lower <= item.retrieved_at <= cutoff
    ]

    mention_observations: list[dict[str, Any]] = []
    mention_raw_rows = 0
    for snapshot in mention_snapshots:
        rows = parse_mentions_zip(store.read_payload(snapshot))
        mention_raw_rows += len(rows)
        for summary in summarize_mentions(rows):
            summary["known_at"] = snapshot.retrieved_at
            summary["snapshot_checksum"] = snapshot.checksum_sha256
            mention_observations.append(summary)
    gkg_documents: list[dict[str, Any]] = []
    gkg_raw_rows = 0
    for snapshot in gkg_snapshots:
        rows = parse_gkg_zip(store.read_payload(snapshot))
        gkg_raw_rows += len(rows)
        for item in normalize_gkg_documents(rows):
            raw_codes = list(item.get("country_codes") or [])
            iso3: set[str] = set()
            if country_map is not None:
                for code in raw_codes:
                    resolved = country_map.resolve(code)
                    if resolved:
                        iso3.add(resolved)
            item["country_iso3"] = sorted(iso3)
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
    )


__all__ = ["GDELTContextBatch", "load_recent_gdelt_context"]
