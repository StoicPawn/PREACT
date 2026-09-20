"""Operational Wave-1 ingestion runner.

The runner is intentionally source-aware: missing credentials or manual-release
sources are reported explicitly rather than replaced with synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preact.data_hub.gateway import SharedProviderGateway
from preact.history.connectors.base import BulkFileConnector
from preact.history.connectors.cow import COWStateSystemConnector
from preact.history.connectors.geonames import (
    CountryCodeMap,
    GeoNamesCountryInfoConnector,
)
from preact.history.connectors.ucdp import UCDPConnector
from preact.history.connectors.unhcr import UNHCRConnector
from preact.history.connectors.world_bank import WorldBankIndicatorConnector
from preact.history.snapshot_store import SourceSnapshotStore


@dataclass(frozen=True)
class SourceRun:
    source_id: str
    status: str
    rows: int = 0
    snapshots: int = 0
    message: str = ""
    metadata: dict[str, Any] | None = None


class Wave1Runner:
    def __init__(
        self,
        *,
        hub_root: str | Path = "data/shared_hub",
        gateway: SharedProviderGateway | None = None,
    ) -> None:
        self.hub_root = Path(hub_root)
        self.snapshot_store = SourceSnapshotStore(self.hub_root / "snapshots")
        self.gateway = gateway or SharedProviderGateway(self.hub_root)

    def run_geonames(self) -> SourceRun:
        connector = GeoNamesCountryInfoConnector(
            BulkFileConnector("geonames", self.snapshot_store)
        )
        acquired = connector.acquire()
        rows = connector.parse(acquired.payload)
        return SourceRun(
            "geonames",
            "success",
            rows=len(rows),
            snapshots=1,
            metadata={
                "snapshot_checksum": acquired.snapshot.checksum_sha256,
                "fips_iso3_pairs": len(CountryCodeMap(rows).fips_to_iso3()),
            },
        )

    def run_cow(self) -> SourceRun:
        connector = COWStateSystemConnector(
            BulkFileConnector("cow", self.snapshot_store)
        )
        acquired = connector.acquire()
        rows = connector.parse_rows(acquired.payload)
        entities = connector.to_entities(rows)
        return SourceRun(
            "cow",
            "success",
            rows=len(entities),
            snapshots=1,
            metadata={
                "release": "State System Membership v2024",
                "snapshot_checksum": acquired.snapshot.checksum_sha256,
                "strict_replay_eligible_before_retrieval": (
                    acquired.replay_eligible_before_retrieval
                ),
            },
        )

    def run_world_bank(
        self,
        *,
        country: str = "WLD",
        indicators: tuple[str, ...] = (
            "NY.GDP.MKTP.KD.ZG",
            "FP.CPI.TOTL.ZG",
            "SP.POP.TOTL",
        ),
        start_year: int = 1960,
        end_year: int | None = None,
    ) -> SourceRun:
        end_year = end_year or datetime.now(timezone.utc).year
        connector = WorldBankIndicatorConnector(self.gateway)
        rows = []
        for indicator in indicators:
            rows.extend(
                connector.fetch(
                    country=country,
                    indicator=indicator,
                    start_year=start_year,
                    end_year=end_year,
                )
            )
        return SourceRun(
            "world_bank",
            "success",
            rows=len(rows),
            snapshots=len(indicators),
            metadata={
                "country": country,
                "indicators": list(indicators),
                "start_year": start_year,
                "end_year": end_year,
            },
        )

    def run_ucdp(
        self,
        *,
        version: str,
        resource: str = "gedevents",
        filters: dict[str, Any] | None = None,
        max_pages: int | None = None,
    ) -> SourceRun:
        try:
            connector = UCDPConnector(self.gateway)
            pages = connector.fetch_pages(
                resource=resource,
                version=version,
                filters=filters,
                max_pages=max_pages,
            )
        except RuntimeError as exc:
            if "UCDP_API_TOKEN" in str(exc):
                return SourceRun(
                    "ucdp",
                    "requires_registration",
                    message=str(exc),
                    metadata={"version": version, "resource": resource},
                )
            raise
        return SourceRun(
            "ucdp",
            "success",
            rows=sum(len(page.rows) for page in pages),
            snapshots=len(pages),
            metadata={"version": version, "resource": resource},
        )

    def run_unhcr(
        self,
        *,
        year_from: int = 1951,
        year_to: int | None = None,
        max_pages: int | None = None,
    ) -> SourceRun:
        year_to = year_to or datetime.now(timezone.utc).year
        pages = UNHCRConnector(self.gateway).fetch_pages(
            year_from=year_from,
            year_to=year_to,
            include_all_origins=True,
            include_all_asylum=True,
            max_pages=max_pages,
        )
        return SourceRun(
            "unhcr",
            "success",
            rows=sum(len(page.rows) for page in pages),
            snapshots=len(pages),
            metadata={"year_from": year_from, "year_to": year_to},
        )

    @staticmethod
    def pending_manual_sources() -> list[SourceRun]:
        return [
            SourceRun(
                "vdem",
                "requires_registration",
                message="Download/version access must be configured before automated ingestion.",
            ),
            SourceRun(
                "sipri",
                "manual_release",
                message="Archive the exact published workbook/release; do not silently replace revisions.",
            ),
        ]

    def run_available(
        self,
        *,
        ucdp_version: str | None = None,
        lightweight: bool = False,
    ) -> list[SourceRun]:
        calls = [
            ("geonames", self.run_geonames),
            ("cow", self.run_cow),
        ]
        if not lightweight:
            calls.extend(
                [
                    ("world_bank", self.run_world_bank),
                    ("unhcr", self.run_unhcr),
                ]
            )

        results: list[SourceRun] = []
        for source_id, call in calls:
            try:
                results.append(call())
            except Exception as exc:
                results.append(
                    SourceRun(
                        source_id,
                        "failed",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )

        if ucdp_version:
            try:
                results.append(self.run_ucdp(version=ucdp_version))
            except Exception as exc:
                results.append(
                    SourceRun(
                        "ucdp",
                        "failed",
                        message=f"{type(exc).__name__}: {exc}",
                        metadata={"version": ucdp_version},
                    )
                )
        else:
            results.append(
                SourceRun(
                    "ucdp",
                    "requires_version",
                    message="Pass an explicit UCDP dataset version for reproducible ingestion.",
                )
            )

        results.extend(self.pending_manual_sources())
        return results
