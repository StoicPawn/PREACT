"""End-to-end Wave-1 acquisition -> normalization -> warehouse pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from preact.data_hub.gateway import SharedProviderGateway
from preact.history.connectors.ucdp import UCDPConnector
from preact.history.connectors.unhcr import UNHCRConnector
from preact.history.connectors.world_bank import WorldBankIndicatorConnector
from preact.history.ingestion_state import IngestionStateStore, SourceRun
from preact.history.warehouse import HistoricalWarehouse
from preact.projections.wave1 import (
    ucdp_records,
    unhcr_records,
    world_bank_records,
)


class Wave1IngestionPipeline:
    def __init__(
        self,
        *,
        gateway: SharedProviderGateway,
        warehouse: HistoricalWarehouse,
        state: IngestionStateStore,
    ) -> None:
        self.gateway = gateway
        self.warehouse = warehouse
        self.state = state

    def _finish(
        self,
        *,
        source_id: str,
        started_at: datetime,
        rows_seen: int,
        rows_inserted: int,
        snapshots: int,
        details: Mapping[str, object],
        status: str = "ready",
    ) -> dict[str, object]:
        completed_at = datetime.now(timezone.utc)
        run = SourceRun(
            source_id=source_id,
            started_at=started_at,
            completed_at=completed_at,
            status=status,
            rows_seen=rows_seen,
            rows_inserted=rows_inserted,
            snapshots=snapshots,
            details=dict(details),
        )
        self.state.record(run)
        return {
            "source_id": source_id,
            "status": status,
            "rows_seen": rows_seen,
            "rows_inserted": rows_inserted,
            "snapshots": snapshots,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "details": dict(details),
        }

    def ingest_world_bank(
        self,
        *,
        country: str,
        indicator: str,
        start_year: int,
        end_year: int,
    ) -> dict[str, object]:
        started_at = datetime.now(timezone.utc)
        observations = WorldBankIndicatorConnector(self.gateway).fetch(
            country=country,
            indicator=indicator,
            start_year=start_year,
            end_year=end_year,
        )
        records = world_bank_records(observations)
        inserted = self.warehouse.insert_records(records)
        return self._finish(
            source_id="world_bank",
            started_at=started_at,
            rows_seen=len(observations),
            rows_inserted=inserted,
            snapshots=len({item.snapshot_checksum for item in observations if item.snapshot_checksum}),
            details={
                "country": country,
                "indicator": indicator,
                "start_year": start_year,
                "end_year": end_year,
            },
        )

    def ingest_unhcr(self, **kwargs: Any) -> dict[str, object]:
        started_at = datetime.now(timezone.utc)
        pages = UNHCRConnector(self.gateway).fetch_pages(**kwargs)
        records = unhcr_records(pages)
        inserted = self.warehouse.insert_records(records)
        return self._finish(
            source_id="unhcr",
            started_at=started_at,
            rows_seen=sum(len(page.rows) for page in pages),
            rows_inserted=inserted,
            snapshots=len({page.snapshot_checksum for page in pages if page.snapshot_checksum}),
            details={"pages": len(pages), **kwargs},
        )

    def ingest_ucdp(
        self,
        *,
        resource: str,
        version: str,
        version_release_at: datetime | None = None,
        **kwargs: Any,
    ) -> dict[str, object]:
        started_at = datetime.now(timezone.utc)
        pages = UCDPConnector(self.gateway).fetch_pages(
            resource=resource,
            version=version,
            **kwargs,
        )
        records = ucdp_records(
            pages,
            version_release_at=version_release_at,
        )
        inserted = self.warehouse.insert_records(records)
        return self._finish(
            source_id="ucdp",
            started_at=started_at,
            rows_seen=sum(len(page.rows) for page in pages),
            rows_inserted=inserted,
            snapshots=len({page.snapshot_checksum for page in pages if page.snapshot_checksum}),
            details={
                "resource": resource,
                "version": version,
                "pages": len(pages),
                "version_release_at": (
                    version_release_at.isoformat()
                    if version_release_at
                    else None
                ),
            },
        )
