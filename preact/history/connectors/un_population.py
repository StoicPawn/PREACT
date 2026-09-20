"""UN Population Division Data Portal API connector."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Any

from preact.data_hub.gateway import ProviderResponse, SharedProviderGateway


@dataclass(frozen=True)
class UNPopulationPage:
    indicators: str
    locations: str
    start_year: int
    end_year: int
    page: int
    rows: tuple[dict[str, Any], ...]
    retrieved_at: datetime
    snapshot_checksum: str | None


class UNPopulationConnector:
    BASE = "https://population.un.org/dataportalapi/api/v1"

    def __init__(
        self,
        gateway: SharedProviderGateway,
        *,
        token: str | None = None,
    ) -> None:
        self.gateway = gateway
        self.token = token or os.getenv("UN_POPULATION_API_TOKEN")

    def fetch_pages(
        self,
        *,
        indicators: str,
        locations: str,
        start_year: int,
        end_year: int,
        page_size: int = 100,
        max_pages: int | None = None,
        ttl_seconds: int = 86400,
    ) -> list[UNPopulationPage]:
        if not self.token:
            raise RuntimeError("UN_POPULATION_API_TOKEN is required")
        if end_year < start_year:
            raise ValueError("end_year must be >= start_year")

        safe_page_size = max(1, min(int(page_size), 100))
        pages: list[UNPopulationPage] = []
        page = 1
        while True:
            response = self.gateway.get_json(
                source_id="un_wpp",
                operation="data_portal",
                url=(
                    f"{self.BASE}/data/indicators/{indicators}/locations/{locations}"
                    f"/start/{int(start_year)}/end/{int(end_year)}"
                ),
                params={
                    "pageNumber": page,
                    "pageSize": safe_page_size,
                    "format": "json",
                },
                ttl_seconds=max(0, int(ttl_seconds)),
                # Official API documents 5 requests / 10 seconds / IP.
                minimum_interval_seconds=2.1,
                timeout_seconds=60.0,
                headers={"Authorization": f"Bearer {self.token}"},
            )
            payload = response.payload
            rows_raw = payload if isinstance(payload, list) else (
                payload.get("data", []) if isinstance(payload, dict) else []
            )
            rows = tuple(item for item in rows_raw if isinstance(item, dict))
            pages.append(
                UNPopulationPage(
                    indicators=indicators,
                    locations=locations,
                    start_year=int(start_year),
                    end_year=int(end_year),
                    page=page,
                    rows=rows,
                    retrieved_at=response.retrieved_at,
                    snapshot_checksum=response.snapshot_checksum,
                )
            )

            # The API max page size is 100. A short/empty page terminates the scan.
            if len(rows) < safe_page_size:
                break
            if max_pages is not None and page >= max(1, int(max_pages)):
                break
            page += 1
        return pages

    def fetch_all(self, **kwargs) -> list[dict[str, Any]]:
        return [row for page in self.fetch_pages(**kwargs) for row in page.rows]
