"""Structured point-in-time world-event observations.

These records are derived from provider-native event streams (for example GDELT
Events). They are not promoted country facts. They feed timelines, relationship
analysis and later claim extraction while preserving provider provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class WorldEventObservation:
    provider: str
    provider_event_id: str
    event_time: datetime
    known_at: datetime
    actor1_entity_id: str
    actor2_entity_id: str | None
    event_code: str | None = None
    event_base_code: str | None = None
    event_root_code: str | None = None
    quad_class: int | None = None
    goldstein: float | None = None
    tone: float | None = None
    num_mentions: float | None = None
    num_sources: float | None = None
    num_articles: float | None = None
    actor1_name: str | None = None
    actor2_name: str | None = None
    action_location: str | None = None
    source_url: str | None = None
    snapshot_checksum: str | None = None
    evidence_class: str = "provider_derived_event"

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider is required")
        if not self.provider_event_id.strip():
            raise ValueError("provider_event_id is required")
        if not self.actor1_entity_id.strip():
            raise ValueError("actor1_entity_id is required")
        if self.event_time.tzinfo is None or self.known_at.tzinfo is None:
            raise ValueError("event_time and known_at must be timezone-aware")
        if self.known_at < self.event_time:
            # Event time is usually day-level for GDELT while knowledge time is exact.
            # A record known before its represented event time would be a leakage risk.
            raise ValueError("known_at cannot precede event_time")

    @property
    def observation_id(self) -> str:
        material = "|".join(
            [
                self.provider,
                self.provider_event_id,
                self.known_at.astimezone(timezone.utc).isoformat(),
                self.snapshot_checksum or "",
            ]
        )
        return "wev_" + sha256(material.encode("utf-8")).hexdigest()[:24]


def _utc_datetime(value: object) -> datetime:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("invalid event timestamp")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def _optional_float(value: object) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def project_gdelt_events(frame: pd.DataFrame) -> list[WorldEventObservation]:
    """Project already-archived GDELT event rows into PREACT world-event records."""

    if frame.empty:
        return []

    required = {
        "event_id",
        "event_date",
        "actor1_country",
        "actor2_country",
        "retrieved_at",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"GDELT event frame missing required columns: {sorted(missing)}")

    output: list[WorldEventObservation] = []
    for _, row in frame.iterrows():
        actor1 = str(row.get("actor1_country") or "").strip().upper()
        actor2 = str(row.get("actor2_country") or "").strip().upper()
        event_id = str(row.get("event_id") or "").strip()
        if len(actor1) != 3 or len(actor2) != 3 or not event_id:
            continue

        event_time = _utc_datetime(row.get("event_date"))
        known_at = _utc_datetime(row.get("retrieved_at"))
        if known_at < event_time:
            # Date-only provider event times can be later than a malformed retrieval
            # timestamp; fail closed rather than creating a time-leaking record.
            continue

        quad_value = _optional_float(row.get("quad_class"))
        output.append(
            WorldEventObservation(
                provider="gdelt",
                provider_event_id=event_id,
                event_time=event_time,
                known_at=known_at,
                actor1_entity_id=f"country:{actor1}",
                actor2_entity_id=f"country:{actor2}",
                event_code=(str(row.get("event_code")).strip() if pd.notna(row.get("event_code")) else None),
                event_base_code=(str(row.get("event_base_code")).strip() if pd.notna(row.get("event_base_code")) else None),
                event_root_code=(str(row.get("event_root_code")).strip() if pd.notna(row.get("event_root_code")) else None),
                quad_class=(int(quad_value) if quad_value is not None else None),
                goldstein=_optional_float(row.get("goldstein")),
                tone=_optional_float(row.get("tone")),
                num_mentions=_optional_float(row.get("num_mentions")),
                num_sources=_optional_float(row.get("num_sources")),
                num_articles=_optional_float(row.get("num_articles")),
                actor1_name=(str(row.get("actor1_name")).strip() if pd.notna(row.get("actor1_name")) else None),
                actor2_name=(str(row.get("actor2_name")).strip() if pd.notna(row.get("actor2_name")) else None),
                action_location=(str(row.get("action_location")).strip() if pd.notna(row.get("action_location")) else None),
                source_url=(str(row.get("source_url")).strip() if pd.notna(row.get("source_url")) else None),
                snapshot_checksum=(
                    str(row.get("snapshot_checksum")).strip()
                    if pd.notna(row.get("snapshot_checksum"))
                    else None
                ),
            )
        )
    return output


__all__ = ["WorldEventObservation", "project_gdelt_events"]
