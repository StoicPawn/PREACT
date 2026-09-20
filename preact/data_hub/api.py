"""FastAPI surface for the shared data provider gateway."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query

from .gateway import SharedProviderGateway
from .gdelt import gdelt_doc_articles
from .google_news import google_news_search
from preact.history.connectors.world_bank import WorldBankIndicatorConnector
from preact.history.source_catalog import SOURCE_BY_ID, SOURCES

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


@app.get("/v1/sources")
def sources(authorization: str | None = Header(default=None)) -> dict:
    _authorize(authorization)
    return {
        "sources": [
            {
                "source_id": source.source_id,
                "name": source.name,
                "domains": list(source.domains),
                "temporal_coverage": source.temporal_coverage,
                "update_cadence": source.update_cadence,
                "access": source.access,
                "replay_policy": source.replay_policy,
                "priority": source.priority,
                "url": source.url,
                "licence_note": source.licence_note,
                "notes": source.notes,
            }
            for source in SOURCES
        ]
    }


@app.get("/v1/sources/{source_id}")
def source_detail(
    source_id: str,
    authorization: str | None = Header(default=None),
) -> dict:
    _authorize(authorization)
    source = SOURCE_BY_ID.get(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="unknown source")
    return {
        "source_id": source.source_id,
        "name": source.name,
        "domains": list(source.domains),
        "temporal_coverage": source.temporal_coverage,
        "update_cadence": source.update_cadence,
        "access": source.access,
        "replay_policy": source.replay_policy,
        "priority": source.priority,
        "url": source.url,
        "licence_note": source.licence_note,
        "notes": source.notes,
    }


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





@app.get("/v1/google-news/rss")
def google_news_rss(
    q: str = Query(..., min_length=1, max_length=800),
    hl: str = Query("en-US", min_length=2, max_length=16),
    gl: str = Query("US", min_length=2, max_length=8),
    ceid: str = Query("US:en", min_length=3, max_length=16),
    max_records: int = Query(60, ge=1, le=200),
    ttl_seconds: int = Query(900, ge=0, le=86400),
    authorization: str | None = Header(default=None),
) -> dict:
    _authorize(authorization)
    result = google_news_search(
        gateway,
        query=q,
        hl=hl,
        gl=gl,
        ceid=ceid,
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

@app.get("/v1/world-bank/indicator")
def world_bank_indicator(
    country: str = Query(..., min_length=2, max_length=12),
    indicator: str = Query(..., min_length=2, max_length=64),
    start_year: int = Query(..., ge=1, le=2200),
    end_year: int = Query(..., ge=1, le=2200),
    ttl_seconds: int = Query(86400, ge=0, le=604800),
    authorization: str | None = Header(default=None),
) -> dict:
    _authorize(authorization)
    rows = WorldBankIndicatorConnector(gateway).fetch(
        country=country,
        indicator=indicator,
        start_year=start_year,
        end_year=end_year,
        ttl_seconds=ttl_seconds,
    )
    return {
        "source": "world_bank",
        "country": country,
        "indicator": indicator,
        "observations": [
            {
                "country_iso3": row.country_iso3,
                "indicator": row.indicator,
                "year": row.year,
                "value": row.value,
                "retrieved_at": row.retrieved_at.isoformat(),
                "snapshot_checksum": row.snapshot_checksum,
                "replay_eligible_before_retrieval": row.replay_eligible_before_retrieval,
            }
            for row in rows
        ],
    }
