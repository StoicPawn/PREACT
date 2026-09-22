"""Build recent country-interaction edges from archived GDELT realtime snapshots.

This loader is intended for the PREACT lab: it consumes only raw files already archived
by the shared data hub, never synthetic API fallbacks, and enforces an as-of cutoff before
constructing the World Explorer relationship layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import pycountry

from preact.analytics.gdelt_graphs import build_state_graph
from preact.data_hub.gdelt_realtime import parse_event_zip
from preact.history.snapshot_store import SourceSnapshotStore


_VALID_ISO3 = {country.alpha_3 for country in pycountry.countries}


@dataclass(frozen=True)
class GDELTRelationshipBatch:
    edges: pd.DataFrame
    snapshot_checksums: tuple[str, ...]
    snapshot_count: int
    raw_event_count: int
    resolved_interaction_count: int
    resolution_rate: float
    newest_retrieved_at: datetime | None


def _utc(value: datetime | pd.Timestamp) -> datetime:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def _numeric(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(parsed) else float(parsed)


def _event_frame(
    rows: Iterable[dict[str, str]],
    *,
    retrieved_at: datetime,
    as_of: datetime,
) -> tuple[pd.DataFrame, int]:
    records: list[dict[str, object]] = []
    raw_count = 0
    for row in rows:
        raw_count += 1
        actor1 = str(row.get("Actor1CountryCode") or "").strip().upper()
        actor2 = str(row.get("Actor2CountryCode") or "").strip().upper()

        # CAMEO/GDELT actor codes are not always ISO-3. Until the entity-resolution
        # crosswalk is installed, keep only codes that are unambiguously valid ISO-3.
        if actor1 not in _VALID_ISO3 or actor2 not in _VALID_ISO3 or actor1 == actor2:
            continue

        event_date = pd.to_datetime(
            str(row.get("SQLDATE") or ""), format="%Y%m%d", errors="coerce", utc=True
        )
        if pd.isna(event_date):
            continue
        if event_date.to_pydatetime() > as_of:
            continue

        event_id = str(row.get("GLOBALEVENTID") or "").strip()
        if not event_id:
            continue

        records.append(
            {
                "event_id": event_id,
                "event_date": event_date.tz_convert(None),
                "actor1_country": actor1,
                "actor2_country": actor2,
                "country": actor1,
                "tone": _numeric(row.get("AvgTone")),
                "goldstein": _numeric(row.get("GoldsteinScale")),
                "num_articles": max(0.0, _numeric(row.get("NumArticles"), 1.0)),
                "source_url": str(row.get("SOURCEURL") or "").strip(),
                "retrieved_at": retrieved_at,
            }
        )

    if not records:
        return pd.DataFrame(), raw_count
    frame = pd.DataFrame.from_records(records)
    return frame, raw_count


def load_recent_relationship_edges(
    root: str | Path,
    *,
    as_of: datetime | pd.Timestamp | None = None,
    lookback_days: int = 14,
    min_events: int = 1,
) -> GDELTRelationshipBatch:
    """Read archived realtime event ZIPs and build leakage-safe state-interaction edges."""

    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    if min_events < 1:
        raise ValueError("min_events must be >= 1")

    cutoff = _utc(as_of or datetime.now(timezone.utc))
    window_start = cutoff - timedelta(days=lookback_days)

    store = SourceSnapshotStore(Path(root) / "snapshots")
    candidates = [
        item
        for item in store.iter_metadata(source_id="gdelt")
        if item.operation == "realtime_events"
        and window_start <= item.retrieved_at <= cutoff
    ]

    frames: list[pd.DataFrame] = []
    raw_event_count = 0
    checksums: list[str] = []
    newest: datetime | None = None

    for snapshot in candidates:
        payload = store.read_payload(snapshot)
        rows = parse_event_zip(payload)
        frame, raw_count = _event_frame(
            rows,
            retrieved_at=snapshot.retrieved_at,
            as_of=cutoff,
        )
        raw_event_count += raw_count
        checksums.append(snapshot.checksum_sha256)
        newest = (
            snapshot.retrieved_at
            if newest is None or snapshot.retrieved_at > newest
            else newest
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return GDELTRelationshipBatch(
            edges=pd.DataFrame(),
            snapshot_checksums=tuple(sorted(set(checksums))),
            snapshot_count=len(candidates),
            raw_event_count=raw_event_count,
            resolved_interaction_count=0,
            resolution_rate=0.0,
            newest_retrieved_at=newest,
        )

    events = pd.concat(frames, ignore_index=True)
    # A provider event can appear in more than one archived update. Keep one logical
    # event so repeated snapshots do not inflate evidence.
    events = (
        events.sort_values(["event_id", "retrieved_at"])
        .drop_duplicates(subset=["event_id"], keep="first")
        .reset_index(drop=True)
    )
    resolved = len(events)
    graph = build_state_graph(
        events,
        weight_column="num_articles",
        min_events=min_events,
        include_self_loops=False,
    )

    return GDELTRelationshipBatch(
        edges=graph.edges,
        snapshot_checksums=tuple(sorted(set(checksums))),
        snapshot_count=len(candidates),
        raw_event_count=raw_event_count,
        resolved_interaction_count=resolved,
        resolution_rate=(resolved / raw_event_count) if raw_event_count else 0.0,
        newest_retrieved_at=newest,
    )


__all__ = ["GDELTRelationshipBatch", "load_recent_relationship_edges"]
