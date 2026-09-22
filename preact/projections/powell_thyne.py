"""Normalize Powell-Thyne coup vintages into PREACT event records."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Iterable, Mapping

from preact.history.connectors.powell_thyne import PowellThyneRelease
from preact.history.schema import EvidenceClass, Provenance, TemporalRecord


def _lower(row: Mapping[str, object]) -> dict[str, object]:
    return {str(k).strip().lower(): v for k, v in row.items()}


def _int(row: Mapping[str, object], *keys: str) -> int | None:
    lowered = _lower(row)
    for key in keys:
        value = lowered.get(key.lower())
        try:
            if value not in (None, ""):
                return int(float(str(value)))
        except (TypeError, ValueError):
            continue
    return None


def _str(row: Mapping[str, object], *keys: str) -> str:
    lowered = _lower(row)
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _date(row: Mapping[str, object]) -> datetime | None:
    year = _int(row, "year", "yr")
    if year is None:
        return None
    month = _int(row, "month", "mon") or 1
    day = _int(row, "day") or 1
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return datetime(year, 1, 1, tzinfo=timezone.utc)


def powell_thyne_records(
    rows: Iterable[Mapping[str, object]],
    *,
    release: PowellThyneRelease,
    retrieved_at: datetime,
    snapshot_checksum: str | None = None,
) -> list[TemporalRecord]:
    output: list[TemporalRecord] = []
    for row in rows:
        event_date = _date(row)
        if event_date is None:
            continue
        coup_code = _int(row, "coup", "outcome", "result")
        if coup_code not in (1, 2):
            continue
        ccode = _str(row, "ccode", "cowcode", "cow_code", "gwn")
        country = _str(row, "country", "country_name", "state")
        entity_id = (
            f"cow_ccode:{ccode}"
            if ccode
            else f"powell_country:{country.casefold().replace(' ', '-')}"
        )
        logical_material = (
            entity_id,
            event_date.isoformat(),
            coup_code,
            country,
        )
        logical_digest = sha256(
            repr(logical_material).encode("utf-8")
        ).hexdigest()
        logical_event_id = f"powell_thyne:{logical_digest}"
        vintage_digest = sha256(
            f"{release.release_id}|{logical_event_id}".encode("utf-8")
        ).hexdigest()
        common = dict(
            entity_id=entity_id,
            valid_from=event_date,
            known_at=release.published_at,
            evidence_class=EvidenceClass.OBSERVATION,
            provenance=Provenance(
                source="powell_thyne_coups",
                source_ref=logical_event_id,
                retrieved_at=retrieved_at,
                dataset_version=release.release_id,
                licence="provider page; preserve citation and vintage",
                transform="Powell-Thyne event row -> PREACT coup event",
                notes=(
                    "Native published vintage. "
                    f"provisional={release.provisional}"
                ),
            ),
            attributes={
                "country": country or None,
                "ccode": ccode or None,
                "coup_code": coup_code,
                "successful": coup_code == 2,
                "release_id": release.release_id,
                "provisional_release": release.provisional,
                "snapshot_checksum": snapshot_checksum,
                "logical_event_id": logical_event_id,
            },
        )
        output.append(
            TemporalRecord(
                record_id=f"powell_thyne:coup_attempt:{release.release_id}:{vintage_digest}",
                variable="event:coup_attempt",
                value=1,
                **common,
            )
        )
        if coup_code == 2:
            output.append(
                TemporalRecord(
                    record_id=f"powell_thyne:coup_success:{release.release_id}:{vintage_digest}",
                    variable="event:coup_success",
                    value=1,
                    **common,
                )
            )
    return output
