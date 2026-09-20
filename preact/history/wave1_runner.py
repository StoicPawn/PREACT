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
from preact.history.connectors.cow_network import COWNetworkConnector
from preact.history.connectors.cshapes import CShapesConnector
from preact.history.connectors.maddison import Maddison2023Connector
from preact.history.connectors.sipri import SIPRIMilitaryExpenditureConnector
from preact.history.connectors.un_population import UNPopulationConnector
from preact.history.connectors.geonames import (
    CountryCodeMap,
    GeoNamesCountryInfoConnector,
)
from preact.history.connectors.ucdp import UCDPConnector
from preact.history.connectors.unhcr import UNHCRConnector
from preact.history.connectors.world_bank import WorldBankIndicatorConnector
from preact.history.snapshot_store import SourceSnapshotStore
from preact.history.ingestion_state import IngestionStateStore
from preact.history.warehouse import HistoricalWarehouse
from preact.history.graph_store import HistoricalGraphStore
from preact.history.wave1_pipeline import Wave1IngestionPipeline
from preact.projections.cow_network import (
    cow_alliance_relations,
    cow_contiguity_relations,
    cow_mid_relations,
    cow_nmc_records,
)


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
        history_db: str | Path = "data/history/preact_history.duckdb",
        state_db: str | Path = "data/history/ingestion_state.sqlite3",
        graph_db: str | Path = "data/history/preact_graph.duckdb",
        gateway: SharedProviderGateway | None = None,
        warehouse: HistoricalWarehouse | None = None,
        state: IngestionStateStore | None = None,
    ) -> None:
        self.hub_root = Path(hub_root)
        self.snapshot_store = SourceSnapshotStore(self.hub_root / "snapshots")
        self.gateway = gateway or SharedProviderGateway(self.hub_root)
        self.warehouse = warehouse or HistoricalWarehouse(history_db)
        self.state = state or IngestionStateStore(state_db)
        self.graph = HistoricalGraphStore(graph_db)
        self.pipeline = Wave1IngestionPipeline(
            gateway=self.gateway,
            warehouse=self.warehouse,
            state=self.state,
        )

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

    def run_cow_network(self) -> SourceRun:
        connector = COWNetworkConnector(
            BulkFileConnector("cow", self.snapshot_store)
        )
        acquired_alliances = connector.acquire_alliances()
        acquired_contiguity = connector.acquire_contiguity()
        acquired_mids = connector.acquire_dyadic_mids()
        acquired_nmc = connector.acquire_nmc()

        known_alliances = acquired_alliances.retrieved_at
        known_contiguity = acquired_contiguity.retrieved_at
        known_mids = acquired_mids.retrieved_at
        known_nmc = acquired_nmc.retrieved_at

        alliance_rows = connector.parse_alliances(acquired_alliances.payload)
        contiguity_rows = connector.parse_contiguity(acquired_contiguity.payload)
        mid_rows = connector.parse_dyadic_mids(acquired_mids.payload)
        nmc_rows = connector.parse_nmc(acquired_nmc.payload)

        relations = [
            *cow_alliance_relations(
                alliance_rows,
                known_at=known_alliances,
                retrieved_at=acquired_alliances.retrieved_at,
            ),
            *cow_contiguity_relations(
                contiguity_rows,
                known_at=known_contiguity,
                retrieved_at=acquired_contiguity.retrieved_at,
            ),
            *cow_mid_relations(
                mid_rows,
                known_at=known_mids,
                retrieved_at=acquired_mids.retrieved_at,
            ),
        ]
        inserted_relations = self.graph.insert(relations)
        nmc_records = cow_nmc_records(
            nmc_rows,
            known_at=known_nmc,
            retrieved_at=acquired_nmc.retrieved_at,
        )
        inserted_records = self.warehouse.insert_records(nmc_records)

        return SourceRun(
            "cow_network",
            "success",
            rows=len(relations) + len(nmc_records),
            snapshots=4,
            metadata={
                "relations_inserted": inserted_relations,
                "nmc_records_inserted": inserted_records,
                "alliances_rows": len(alliance_rows),
                "contiguity_rows": len(contiguity_rows),
                "mid_rows": len(mid_rows),
                "nmc_rows": len(nmc_rows),
                "releases": [
                    acquired_alliances.snapshot.source_release,
                    acquired_contiguity.snapshot.source_release,
                    acquired_mids.snapshot.source_release,
                    acquired_nmc.snapshot.source_release,
                ],
            },
        )

    def run_cshapes(self) -> SourceRun:
        connector = CShapesConnector(
            BulkFileConnector("cshapes", self.snapshot_store)
        )
        acquired = connector.acquire()
        rows = connector.parse_rows(acquired.payload)
        return SourceRun(
            "cshapes",
            "success",
            rows=len(rows),
            snapshots=1,
            metadata={
                "release": "CShapes 2.0",
                "snapshot_checksum": acquired.snapshot.checksum_sha256,
                "strict_replay_eligible_before_retrieval": (
                    acquired.replay_eligible_before_retrieval
                ),
            },
        )

    def run_maddison(self) -> SourceRun:
        connector = Maddison2023Connector(
            self.gateway,
            BulkFileConnector("maddison", self.snapshot_store),
        )
        acquired = connector.acquire()
        return SourceRun(
            "maddison",
            "success",
            rows=0,
            snapshots=1,
            metadata={
                "release": acquired.snapshot.source_release,
                "snapshot_checksum": acquired.snapshot.checksum_sha256,
                "strict_replay_eligible_before_retrieval": (
                    acquired.replay_eligible_before_retrieval
                ),
            },
        )

    def run_sipri(self) -> SourceRun:
        connector = SIPRIMilitaryExpenditureConnector(
            BulkFileConnector("sipri", self.snapshot_store)
        )
        acquired = connector.acquire()
        return SourceRun(
            "sipri",
            "success",
            rows=0,
            snapshots=1,
            metadata={
                "release": acquired.snapshot.source_release,
                "snapshot_checksum": acquired.snapshot.checksum_sha256,
                "strict_replay_eligible_before_retrieval": (
                    acquired.replay_eligible_before_retrieval
                ),
            },
        )

    def run_un_population(
        self,
        *,
        indicators: str = "49",
        locations: str = "900",
        start_year: int = 1950,
        end_year: int | None = None,
        max_pages: int | None = None,
    ) -> SourceRun:
        end_year = end_year or datetime.now(timezone.utc).year
        try:
            result = self.pipeline.ingest_un_population(
                indicators=indicators,
                locations=locations,
                start_year=start_year,
                end_year=end_year,
                max_pages=max_pages,
            )
        except RuntimeError as exc:
            if "UN_POPULATION_API_TOKEN" in str(exc):
                return SourceRun(
                    "un_wpp",
                    "requires_registration",
                    message=str(exc),
                    metadata={
                        "indicators": indicators,
                        "locations": locations,
                        "start_year": start_year,
                        "end_year": end_year,
                    },
                )
            raise
        return SourceRun(
            "un_wpp",
            "success",
            rows=int(result["rows_seen"]),
            snapshots=int(result["snapshots"]),
            metadata={
                **dict(result["details"]),
                "rows_inserted": int(result["rows_inserted"]),
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
        inserted = 0
        rows = 0
        snapshots = 0
        details: list[dict[str, Any]] = []
        for indicator in indicators:
            result = self.pipeline.ingest_world_bank(
                country=country,
                indicator=indicator,
                start_year=start_year,
                end_year=end_year,
            )
            inserted += int(result["rows_inserted"])
            rows += int(result["rows_seen"])
            snapshots += int(result["snapshots"])
            details.append(dict(result["details"]))
        return SourceRun(
            "world_bank",
            "success",
            rows=rows,
            snapshots=snapshots,
            metadata={
                "country": country,
                "indicators": list(indicators),
                "start_year": start_year,
                "end_year": end_year,
                "rows_inserted": inserted,
                "pipeline_runs": details,
            },
        )

    def run_ucdp(
        self,
        *,
        version: str,
        resource: str = "gedevents",
        filters: dict[str, Any] | None = None,
        max_pages: int | None = None,
        version_release_at: datetime | None = None,
    ) -> SourceRun:
        try:
            result = self.pipeline.ingest_ucdp(
                resource=resource,
                version=version,
                filters=filters,
                max_pages=max_pages,
                version_release_at=version_release_at,
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
            rows=int(result["rows_seen"]),
            snapshots=int(result["snapshots"]),
            metadata={
                **dict(result["details"]),
                "rows_inserted": int(result["rows_inserted"]),
            },
        )

    def run_unhcr(
        self,
        *,
        year_from: int = 1951,
        year_to: int | None = None,
        max_pages: int | None = None,
    ) -> SourceRun:
        year_to = year_to or datetime.now(timezone.utc).year
        result = self.pipeline.ingest_unhcr(
            year_from=year_from,
            year_to=year_to,
            include_all_origins=True,
            include_all_asylum=True,
            max_pages=max_pages,
        )
        return SourceRun(
            "unhcr",
            "success",
            rows=int(result["rows_seen"]),
            snapshots=int(result["snapshots"]),
            metadata={
                **dict(result["details"]),
                "rows_inserted": int(result["rows_inserted"]),
            },
        )

    @staticmethod
    def pending_manual_sources() -> list[SourceRun]:
        return [
            SourceRun(
                "vdem",
                "requires_registration",
                message=(
                    "V-Dem download form/registration must be configured before "
                    "automated ingestion. Always pin the dataset release version."
                ),
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
            ("cow_network", self.run_cow_network),
            ("cshapes", self.run_cshapes),
            ("maddison", self.run_maddison),
            ("sipri", self.run_sipri),
        ]
        if not lightweight:
            calls.extend(
                [
                    ("world_bank", self.run_world_bank),
                    ("unhcr", self.run_unhcr),
                    ("un_wpp", self.run_un_population),
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
