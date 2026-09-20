"""FastAPI surface for the shared data provider gateway."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query

from .gateway import SharedProviderGateway
from .gdelt import gdelt_doc_articles

ROOT = Path(os.getenv("SHARED_DATA_HUB_ROOT", "data/shared_hub"))
TOKEN = os.getenv("SHARED_DATA_HUB_TOKEN", "").strip()
gateway = SharedProviderGateway(ROOT)
app = FastAPI(title="PREACT Shared Data Hub", version="0.1.0")


def _authorize(authorization: str | None) -> None:
    if not TOKEN:
        return
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="invalid data-hub token")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "shared-data-hub"}


@app.get("/v1/gdelt/doc")
def gdelt_doc(
    q: str = Query(..., min_length=1, max_length=800),
    timespan: str = Query("1d", min_length=2, max_length=32),
    max_records: int = Query(75, ge=1, le=250),
    ttl_seconds: int = Query(900, ge=0, le=86400),
    authorization: str | None = Header(default=None),
) -> dict:
    _authorize(authorization)
    result = gdelt_doc_articles(
        gateway,
        query=q,
        timespan=timespan,
        max_records=max_records,
        ttl_seconds=ttl_seconds,
    )
    return {
        "source": result.source_id,
        "operation": result.operation,
        "cached": result.cached,
        "retrieved_at": result.retrieved_at.isoformat(),
        "request_fingerprint": result.request_fingerprint,
        "snapshot_checksum": result.snapshot_checksum,
        "articles": result.payload.get("articles", []),
    }
