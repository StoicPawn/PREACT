"""Library of Congress Chronicling America connector (loc.gov API)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from preact.data_hub.gateway import SharedProviderGateway


@dataclass(frozen=True)
class ChroniclingPage:
    page: int
    rows: tuple[dict[str, Any], ...]
    retrieved_at: datetime
    snapshot_checksum: str | None
    next_url: str | None = None


class ChroniclingAmericaConnector:
    """Search historic newspaper content through the current loc.gov API."""

    BASE = "https://www.loc.gov/collections/chronicling-america/"

    def __init__(self, gateway: SharedProviderGateway) -> None:
        self.gateway = gateway

    def search_pages(
        self,
        *,
        query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        state: str | None = None,
        city: str | None = None,
        language: str | None = None,
        display_level: str = "page",
        operation: str = "AND",
        front_pages_only: bool = False,
        page_size: int = 100,
        max_pages: int | None = None,
        ttl_seconds: int = 86400,
    ) -> list[ChroniclingPage]:
        params: dict[str, Any] = {
            "fo": "json",
            "dl": display_level,
            "qs": query,
            "ops": operation,
            "c": max(1, min(int(page_size), 100)),
            "sp": 1,
            "at": "results,pagination",
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if state:
            params["location_state"] = state
        if city:
            params["location_city"] = city
        if language:
            params["fa"] = f"language:{language}"
        if front_pages_only:
            params["front_pages_only"] = "true"

        pages: list[ChroniclingPage] = []
        page = 1
        while True:
            params["sp"] = page
            response = self.gateway.get_json(
                source_id="chronicling_america",
                operation="loc_search",
                url=self.BASE,
                params=params,
                ttl_seconds=max(0, int(ttl_seconds)),
                minimum_interval_seconds=0.25,
                timeout_seconds=60.0,
            )
            payload = response.payload if isinstance(response.payload, dict) else {}
            results = payload.get("results", [])
            rows = tuple(item for item in results if isinstance(item, dict))
            pagination = payload.get("pagination", {})
            next_url = (
                pagination.get("next")
                if isinstance(pagination, dict)
                else None
            )
            pages.append(
                ChroniclingPage(
                    page=page,
                    rows=rows,
                    retrieved_at=response.retrieved_at,
                    snapshot_checksum=response.snapshot_checksum,
                    next_url=str(next_url) if next_url else None,
                )
            )
            if not next_url or not rows:
                break
            if max_pages is not None and page >= max(1, int(max_pages)):
                break
            page += 1
        return pages

    def search_all(self, **kwargs) -> list[dict[str, Any]]:
        return [row for page in self.search_pages(**kwargs) for row in page.rows]
