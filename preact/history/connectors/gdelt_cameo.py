"""GDELT/CAMEO country-code lookup acquisition and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pycountry

from .base import AcquiredDataset, BulkFileConnector

CAMEO_COUNTRY_URL = "https://gdeltproject.org/data/lookups/CAMEO.country.txt"


@dataclass(frozen=True)
class CAMEOCountryRow:
    code: str
    label: str
    iso3: str | None


class CAMEOCountryMap:
    """Resolve GDELT actor country codes to current ISO-3166 alpha-3 codes.

    Regional/supranational CAMEO codes remain unresolved by design. Historical
    entities that no longer map one-to-one to a current ISO country also remain
    unresolved unless the official label can be mapped unambiguously.
    """

    def __init__(self, rows: Iterable[CAMEOCountryRow]) -> None:
        self.rows = tuple(rows)
        self._by_code = {row.code: row for row in self.rows if row.code}

    def resolve(self, code: str) -> str | None:
        key = str(code or "").strip().upper()
        row = self._by_code.get(key)
        return row.iso3 if row is not None else None

    def as_dict(self) -> dict[str, str]:
        return {
            row.code: row.iso3
            for row in self.rows
            if row.code and row.iso3 is not None
        }


def _label_to_iso3(label: str) -> str | None:
    value = str(label or "").strip()
    if not value:
        return None

    aliases = {
        "EAST TIMOR": "Timor-Leste",
        "MACEDONIA": "North Macedonia",
        "SWAZILAND": "Eswatini",
        "IVORY COAST": "Côte d'Ivoire",
        "CAPE VERDE": "Cabo Verde",
        "FAEROE ISLANDS": "Faroe Islands",
        "OCCUPIED PALESTINIAN TERRITORY": "Palestine, State of",
        "REUNION": "Réunion",
        "RUNION": "Réunion",
    }
    candidate = aliases.get(value.upper(), value)

    try:
        return pycountry.countries.lookup(candidate).alpha_3
    except LookupError:
        try:
            matches = pycountry.countries.search_fuzzy(candidate)
        except LookupError:
            return None
        return matches[0].alpha_3 if len(matches) == 1 else None


def parse_cameo_country_lookup(payload: bytes) -> list[CAMEOCountryRow]:
    """Parse the official GDELT tab-delimited CAMEO country lookup."""

    text = payload.decode("utf-8-sig", errors="replace")
    rows: list[CAMEOCountryRow] = []
    seen: set[str] = set()
    for index, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        if index == 0 and line.upper().startswith("CODE\t"):
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        code = parts[0].strip().upper()
        label = parts[1].strip()
        if not code or code in seen:
            continue
        seen.add(code)

        iso3: str | None
        try:
            iso3 = pycountry.countries.get(alpha_3=code).alpha_3  # type: ignore[union-attr]
        except AttributeError:
            iso3 = None
        if iso3 is None:
            iso3 = _label_to_iso3(label)

        rows.append(CAMEOCountryRow(code=code, label=label, iso3=iso3))
    return rows


class GDELTCAMEOCountryConnector:
    """Acquire the official GDELT CAMEO country lookup as an immutable snapshot."""

    def __init__(self, bulk: BulkFileConnector) -> None:
        if bulk.source_id != "gdelt":
            raise ValueError("BulkFileConnector source_id must be 'gdelt'")
        self.bulk = bulk

    def acquire(self) -> AcquiredDataset:
        return self.bulk.fetch(
            CAMEO_COUNTRY_URL,
            source_release="GDELT CAMEO country lookup",
            licence_reference="https://www.gdeltproject.org/",
            replay_eligible_before_retrieval=False,
            notes=(
                "Current actor-code lookup snapshot. Historical replay must bind "
                "to a mapping snapshot retrieved no later than the replay cutoff."
            ),
        )


def build_cameo_country_map(payload: bytes) -> CAMEOCountryMap:
    return CAMEOCountryMap(parse_cameo_country_lookup(payload))


__all__ = [
    "CAMEO_COUNTRY_URL",
    "CAMEOCountryMap",
    "CAMEOCountryRow",
    "GDELTCAMEOCountryConnector",
    "build_cameo_country_map",
    "parse_cameo_country_lookup",
]
