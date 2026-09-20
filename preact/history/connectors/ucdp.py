"""Version-pinned UCDP API connector.

UCDP guarantees that a versioned API URL remains stable. PREACT therefore
requires an explicit dataset version for every retrieval and snapshots every page.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Any, Mapping

from preact.data_hub.gateway import SharedProviderGateway


@dataclass(frozen=True)
class UCDPPage:
    resource: str
    version: str
    page: int
    total_pages: int
    total_count: int
    rows: tuple[dict[str, Any], ...]
    retrieved_at: datetime
    snapshot_checksum: str | None


class UCDPConnector:
    BASE = "https://ucdpapi.pcr.uu.se/api"

    def __init__(
        self,
        gateway: SharedProviderGateway,
        *,
        token: str | None = None,
    ) -> None:
        self.gateway = gateway
        self.token = token or os.getenv("UCDP_API_TOKEN")

    def fetch_pages(
        self,
        *,
        resource: str,
        version: str,
        filters: Mapping[str, Any] | None = None,
        page_size: int = 1000,
        max_pages: int | None = None,
        ttl_seconds: int = 86400,
    ) -> list[UCDPPage]:
        if not version.strip():
            raise ValueError("UCDP version is required for reproducible replay")
        if not self.token:
            raise RuntimeError("UCDP_API_TOKEN is required")
        safe_size = max(1, min(int(page_size), 1000))
        page = 1
        pages: list[UCDPPage] = []

        while True:
            params = {"pagesize": safe_size, "page": page}
            params.update(dict(filters or {}))
            response = self.gateway.get_json(
                source_id="ucdp",
                operation=f"{resource}:{version}",
                url=f"{self.BASE}/{resource}/{version}",
                params=params,
                ttl_seconds=max(0, int(ttl_seconds)),
                minimum_interval_seconds=0.05,
                timeout_seconds=60.0,
                headers={"x-ucdp-access-token": self.token},
            )
            payload = response.payload if isinstance(response.payload, dict) else {}
            rows_raw = payload.get("Result", [])
            rows = tuple(item for item in rows_raw if isinstance(item, dict))
            total_pages = int(payload.get("TotalPages") or page)
            total_count = int(payload.get("TotalCount") or len(rows))
            pages.append(
                UCDPPage(
                    resource=resource,
                    version=version,
                    page=page,
                    total_pages=total_pages,
                    total_count=total_count,
                    rows=rows,
                    retrieved_at=response.retrieved_at,
                    snapshot_checksum=response.snapshot_checksum,
                )
            )
            if page >= total_pages:
                break
            if max_pages is not None and page >= max(1, int(max_pages)):
                break
            page += 1
        return pages

    def fetch_all(self, **kwargs) -> list[dict[str, Any]]:
        pages = self.fetch_pages(**kwargs)
        return [row for page in pages for row in page.rows]
