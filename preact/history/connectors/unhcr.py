"""UNHCR Refugee Statistics API connector."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from preact.data_hub.gateway import SharedProviderGateway


@dataclass(frozen=True)
class UNHCRPage:
    dataset: str
    page: int
    max_pages: int
    rows: tuple[dict[str, Any], ...]
    retrieved_at: datetime
    snapshot_checksum: str | None


class UNHCRConnector:
    BASE = "https://api.unhcr.org/population/v1"

    def __init__(self, gateway: SharedProviderGateway) -> None:
        self.gateway = gateway

    @staticmethod
    def _max_pages(payload: dict[str, Any], default: int) -> int:
        direct = payload.get("maxPages")
        if direct is not None:
            try:
                return max(1, int(direct))
            except (TypeError, ValueError):
                pass
        pagination = payload.get("pagination")
        if isinstance(pagination, dict):
            value = pagination.get("maxPages")
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                pass
        return max(1, int(default))

    def fetch_pages(
        self,
        *,
        dataset: str = "population",
        year_from: int | None = None,
        year_to: int | None = None,
        country_origin: str | None = None,
        country_asylum: str | None = None,
        include_all_origins: bool = False,
        include_all_asylum: bool = False,
        extra_params: Mapping[str, Any] | None = None,
        page_size: int = 10000,
        max_pages: int | None = None,
        ttl_seconds: int = 86400,
    ) -> list[UNHCRPage]:
        params: dict[str, Any] = {
            "limit": max(1, min(int(page_size), 10000)),
            "cf_type": "ISO",
        }
        if year_from is not None:
            params["yearFrom"] = int(year_from)
        if year_to is not None:
            params["yearTo"] = int(year_to)
        if country_origin:
            params["coo"] = country_origin.upper()
        if country_asylum:
            params["coa"] = country_asylum.upper()
        if include_all_origins:
            params["coo_all"] = "true"
        if include_all_asylum:
            params["coa_all"] = "true"
        params.update(dict(extra_params or {}))

        pages: list[UNHCRPage] = []
        page = 1
        while True:
            page_params = dict(params)
            page_params["page"] = page
            response = self.gateway.get_json(
                source_id="unhcr",
                operation=dataset,
                url=f"{self.BASE}/{dataset}/",
                params=page_params,
                ttl_seconds=max(0, int(ttl_seconds)),
                minimum_interval_seconds=0.05,
                timeout_seconds=60.0,
            )
            payload = response.payload if isinstance(response.payload, dict) else {}
            items = payload.get("items", [])
            rows = tuple(item for item in items if isinstance(item, dict))
            total_pages = self._max_pages(payload, page)
            pages.append(
                UNHCRPage(
                    dataset=dataset,
                    page=page,
                    max_pages=total_pages,
                    rows=rows,
                    retrieved_at=response.retrieved_at,
                    snapshot_checksum=response.snapshot_checksum,
                )
            )
            if page >= total_pages:
                break
            if max_pages is not None and page >= max(1, int(max_pages)):
                break
            if not rows:
                break
            page += 1
        return pages

    def fetch_all(self, **kwargs) -> list[dict[str, Any]]:
        return [row for page in self.fetch_pages(**kwargs) for row in page.rows]
