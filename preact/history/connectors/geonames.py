"""GeoNames country-code crosswalk connector."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from .base import AcquiredDataset, BulkFileConnector

COUNTRY_INFO_URL = "https://download.geonames.org/export/dump/countryInfo.txt"

COUNTRY_INFO_COLUMNS = (
    "iso2",
    "iso3",
    "iso_numeric",
    "fips",
    "country",
    "capital",
    "area_km2",
    "population",
    "continent",
    "tld",
    "currency_code",
    "currency_name",
    "phone",
    "postal_code_format",
    "postal_code_regex",
    "languages",
    "geoname_id",
    "neighbours",
    "equivalent_fips",
)


@dataclass(frozen=True)
class CountryCodeRow:
    iso2: str
    iso3: str
    fips: str
    country: str
    geoname_id: str | None = None


class GeoNamesCountryInfoConnector:
    def __init__(self, bulk: BulkFileConnector) -> None:
        if bulk.source_id != "geonames":
            raise ValueError("BulkFileConnector source_id must be 'geonames'")
        self.bulk = bulk

    def acquire(self) -> AcquiredDataset:
        return self.bulk.fetch(
            COUNTRY_INFO_URL,
            source_release="daily countryInfo.txt",
            licence_reference="https://www.geonames.org/export/",
            replay_eligible_before_retrieval=False,
            notes=(
                "Current crosswalk snapshot. Useful for normalization; not a "
                "historical border/state-membership source."
            ),
        )

    @staticmethod
    def parse(payload: bytes) -> list[CountryCodeRow]:
        text = payload.decode("utf-8-sig", errors="replace")
        rows: list[CountryCodeRow] = []
        for raw_line in text.splitlines():
            if not raw_line or raw_line.startswith("#"):
                continue
            values = next(csv.reader([raw_line], delimiter="\t"))
            padded = values + [""] * max(0, len(COUNTRY_INFO_COLUMNS) - len(values))
            data = dict(zip(COUNTRY_INFO_COLUMNS, padded))
            iso2 = data["iso2"].strip().upper()
            iso3 = data["iso3"].strip().upper()
            fips = data["fips"].strip().upper()
            if not iso2:
                continue
            rows.append(
                CountryCodeRow(
                    iso2=iso2,
                    iso3=iso3,
                    fips=fips,
                    country=data["country"].strip(),
                    geoname_id=data["geoname_id"].strip() or None,
                )
            )
        return rows


class CountryCodeMap:
    def __init__(self, rows: list[CountryCodeRow]) -> None:
        self.rows = tuple(rows)
        self.by_fips = {row.fips: row for row in rows if row.fips}
        self.by_iso2 = {row.iso2: row for row in rows if row.iso2}
        self.by_iso3 = {row.iso3: row for row in rows if row.iso3}

    def fips_to_iso3(self) -> dict[str, str]:
        return {
            fips: row.iso3
            for fips, row in self.by_fips.items()
            if row.iso3
        }
