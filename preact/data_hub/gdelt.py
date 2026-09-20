"""GDELT adapters for the shared provider gateway."""

from __future__ import annotations

from typing import Any

from .gateway import ProviderResponse, SharedProviderGateway

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def gdelt_doc_articles(
    gateway: SharedProviderGateway,
    *,
    query: str,
    timespan: str = "1d",
    max_records: int = 75,
    ttl_seconds: int = 900,
) -> ProviderResponse:
    """Return GDELT article-list results through the shared cache/snapshot layer."""

    safe_max = max(1, min(int(max_records), 250))
    response = gateway.get_json(
        source_id="gdelt",
        operation="doc_artlist",
        url=GDELT_DOC_URL,
        params={
            "query": query,
            "mode": "artlist",
            "maxrecords": safe_max,
            "timespan": timespan,
            "sort": "datedesc",
            "format": "json",
        },
        ttl_seconds=max(0, int(ttl_seconds)),
        minimum_interval_seconds=1.0,
        timeout_seconds=35.0,
    )
    payload = response.payload if isinstance(response.payload, dict) else {}
    articles = payload.get("articles", [])
    clean: list[dict[str, Any]] = [item for item in articles if isinstance(item, dict)]
    return ProviderResponse(
        source_id=response.source_id,
        operation=response.operation,
        payload={"articles": clean},
        retrieved_at=response.retrieved_at,
        cached=response.cached,
        request_fingerprint=response.request_fingerprint,
        snapshot_checksum=response.snapshot_checksum,
    )
