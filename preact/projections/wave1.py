"""Normalize initial Wave-1 providers into PREACT TemporalRecords."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence

from preact.history.connectors.ucdp import UCDPPage
from preact.history.connectors.unhcr import UNHCRPage
from preact.history.connectors.world_bank import WorldBankObservation
from preact.history.schema import EvidenceClass, Provenance, TemporalRecord


def _record_id(prefix: str, material: object) -> str:
    digest = sha256(repr(material).encode("utf-8")).hexdigest()
    return f"{prefix}:sha256:{digest}"


def world_bank_records(
    observations: Iterable[WorldBankObservation],
) -> list[TemporalRecord]:
    output: list[TemporalRecord] = []
    for obs in observations:
        valid_from = datetime(obs.year, 1, 1, tzinfo=timezone.utc)
        valid_to = datetime(obs.year + 1, 1, 1, tzinfo=timezone.utc)
        output.append(
            TemporalRecord(
                record_id=(
                    f"world_bank:{obs.country_iso3}:{obs.indicator}:"
                    f"{obs.year}:{obs.snapshot_checksum or 'nosnapshot'}"
                ),
                entity_id=f"iso3:{obs.country_iso3}",
                variable=f"world_bank:{obs.indicator}",
                value=obs.value,
                valid_from=valid_from,
                valid_to=valid_to,
                known_at=obs.retrieved_at,
                evidence_class=EvidenceClass.OBSERVATION,
                provenance=Provenance(
                    source="world_bank",
                    source_ref=f"{obs.country_iso3}:{obs.indicator}:{obs.year}",
                    retrieved_at=obs.retrieved_at,
                    transform="World Bank API observation -> annual bitemporal record",
                    notes=(
                        "Current-vintage API value. Strict replay before retrieval "
                        "is intentionally blocked unless a historical vintage is supplied."
                    ),
                ),
                attributes={
                    "snapshot_checksum": obs.snapshot_checksum,
                    "replay_eligible_before_retrieval": (
                        obs.replay_eligible_before_retrieval
                    ),
                },
            )
        )
    return output


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def unhcr_records(pages: Sequence[UNHCRPage]) -> list[TemporalRecord]:
    output: list[TemporalRecord] = []
    for page in pages:
        for row in page.rows:
            year = _int_or_none(row.get("year"))
            if year is None or year < 1:
                continue
            valid_from = datetime(year, 1, 1, tzinfo=timezone.utc)
            valid_to = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            asylum_iso = str(
                row.get("coa_iso")
                or row.get("coa")
                or row.get("country_asylum_iso")
                or ""
            ).strip().upper()
            origin_iso = str(
                row.get("coo_iso")
                or row.get("coo")
                or row.get("country_origin_iso")
                or ""
            ).strip().upper()
            primary_iso = asylum_iso or origin_iso or "UNK"

            population_fields = {
                key: _float_or_none(row.get(key))
                for key in (
                    "refugees",
                    "asylum_seekers",
                    "returned_refugees",
                    "idps",
                    "returned_idps",
                    "stateless",
                    "ooc",
                    "oip",
                )
                if key in row
            }
            material = (
                page.dataset,
                year,
                origin_iso,
                asylum_iso,
                tuple(sorted((str(k), str(v)) for k, v in row.items())),
            )
            output.append(
                TemporalRecord(
                    record_id=_record_id("unhcr", material),
                    entity_id=(
                        f"iso3:{primary_iso}"
                        if len(primary_iso) == 3
                        else f"unhcr_country:{primary_iso}"
                    ),
                    variable=f"unhcr:{page.dataset}",
                    value=population_fields or dict(row),
                    valid_from=valid_from,
                    valid_to=valid_to,
                    known_at=page.retrieved_at,
                    evidence_class=EvidenceClass.OBSERVATION,
                    provenance=Provenance(
                        source="unhcr",
                        source_ref=f"{page.dataset}:page:{page.page}",
                        retrieved_at=page.retrieved_at,
                        transform="UNHCR API row -> annual bitemporal record",
                        notes=(
                            "Current API snapshot; historical values are not backdated "
                            "for strict replay."
                        ),
                    ),
                    attributes={
                        "snapshot_checksum": page.snapshot_checksum,
                        "origin_iso": origin_iso or None,
                        "asylum_iso": asylum_iso or None,
                        "raw_row": dict(row),
                    },
                )
            )
    return output


def _parse_date(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def ucdp_records(
    pages: Sequence[UCDPPage],
    *,
    version_release_at: datetime | None = None,
) -> list[TemporalRecord]:
    """Normalize UCDP event-like rows conservatively.

    If a trustworthy release timestamp for the pinned UCDP version is supplied,
    it becomes known_at. Otherwise retrieval time is used, preventing accidental
    hindsight in replay.
    """

    output: list[TemporalRecord] = []
    for page in pages:
        known_at = version_release_at or page.retrieved_at
        if known_at.tzinfo is None:
            known_at = known_at.replace(tzinfo=timezone.utc)
        for row in page.rows:
            valid_from = (
                _parse_date(row.get("date_start"))
                or _parse_date(row.get("date_start_clean"))
                or _parse_date(row.get("date"))
            )
            if valid_from is None:
                year = _int_or_none(row.get("year"))
                if year is None:
                    continue
                valid_from = datetime(year, 1, 1, tzinfo=timezone.utc)

            country_id = str(
                row.get("country_id")
                or row.get("country")
                or row.get("country_name")
                or "unknown"
            ).strip()
            provider_id = str(row.get("id") or row.get("id_event") or "").strip()
            material = (
                page.resource,
                page.version,
                provider_id,
                tuple(sorted((str(k), str(v)) for k, v in row.items())),
            )
            output.append(
                TemporalRecord(
                    record_id=(
                        f"ucdp:{page.resource}:{page.version}:{provider_id}"
                        if provider_id
                        else _record_id("ucdp", material)
                    ),
                    entity_id=f"ucdp_country:{country_id}",
                    variable=f"ucdp:{page.resource}",
                    value=dict(row),
                    valid_from=valid_from,
                    known_at=known_at,
                    evidence_class=EvidenceClass.OBSERVATION,
                    provenance=Provenance(
                        source="ucdp",
                        source_ref=provider_id or f"page:{page.page}",
                        retrieved_at=page.retrieved_at,
                        dataset_version=page.version,
                        transform="version-pinned UCDP row -> bitemporal record",
                    ),
                    attributes={
                        "snapshot_checksum": page.snapshot_checksum,
                        "resource": page.resource,
                        "version": page.version,
                        "release_timestamp_supplied": version_release_at is not None,
                    },
                )
            )
    return output
