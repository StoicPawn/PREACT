"""FastAPI application exposing PREACT backend services."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

try:  # pragma: no cover - runtime dependency management
    from pydantic import BaseModel
except ModuleNotFoundError:  # pragma: no cover - fallback when Pydantic is unavailable
    from ._pydantic_stub import install_pydantic_stub

    install_pydantic_stub()
    from pydantic import BaseModel  # type: ignore

from ..config import PREACTConfig
from ..data_ingestion.orchestrator import DataIngestionOrchestrator
from ..data_ingestion.sources import GDELTQuery, GDELTSource, IngestionResult

try:  # pragma: no cover - runtime dependency management
    from fastapi import FastAPI, HTTPException, Query
except ModuleNotFoundError:  # pragma: no cover - fallback when FastAPI is unavailable
    from ._fastapi_stub import install_fastapi_stub

    install_fastapi_stub()
    from fastapi import FastAPI, HTTPException, Query  # type: ignore


class IngestRequest(BaseModel):
    """Payload used to trigger ingestion runs."""

    lookback_days: int | None = None
    end: datetime | None = None
    persist: bool | None = None


def _normalise_metadata(metadata: Mapping[str, str]) -> Dict[str, Any]:
    normalised: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            normalised[key] = value
            continue
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "false"}:
                normalised[key] = lowered == "true"
                continue
            try:
                if "." in value:
                    normalised[key] = float(value)
                else:
                    normalised[key] = int(value)
                continue
            except ValueError:
                pass
        normalised[key] = value
    return normalised


def _summarise_ingestion(result: Any) -> Dict[str, Dict[str, int]]:
    bronze_summary: Dict[str, int] = {}
    if hasattr(result, "bronze"):
        bronze_items = getattr(result, "bronze").items()
        for name, ingestion in bronze_items:
            rows = getattr(ingestion, "metadata", {}).get("rows")
            try:
                bronze_summary[name] = int(rows)
            except (TypeError, ValueError):
                bronze_summary[name] = len(getattr(ingestion, "data", []))

    silver_summary: Dict[str, int] = {}
    if hasattr(result, "silver"):
        for name, frame in getattr(result, "silver").items():
            try:
                silver_summary[name] = len(frame)
            except TypeError:
                silver_summary[name] = 0

    gold_summary: Dict[str, int] = {}
    if hasattr(result, "gold"):
        for name, frame in getattr(result, "gold").items():
            try:
                gold_summary[name] = len(frame)
            except TypeError:
                gold_summary[name] = 0

    return {
        "bronze": bronze_summary,
        "silver": silver_summary,
        "gold": gold_summary,
    }


def _serialise_events(frame: pd.DataFrame, limit: int) -> Iterable[Dict[str, Any]]:
    if frame.empty:
        return []
    columns = [
        "event_id",
        "event_date",
        "country",
        "actor1",
        "actor2",
        "themes",
        "tone",
        "goldstein",
        "num_articles",
        "source_url",
        "latitude",
        "longitude",
    ]
    available = [column for column in columns if column in frame.columns]
    subset = frame.loc[:, available].head(limit)
    records = []
    for _, row in subset.iterrows():
        record: Dict[str, Any] = {}
        for column, value in row.items():
            if isinstance(value, pd.Timestamp):
                record[column] = value.to_pydatetime().isoformat()
            elif isinstance(value, datetime):
                record[column] = value.isoformat()
            elif pd.isna(value):
                record[column] = None
            else:
                record[column] = value
        records.append(record)
    return records


def _find_gdelt_source(orchestrator: DataIngestionOrchestrator) -> GDELTSource:
    for source in orchestrator.sources.values():
        if isinstance(source, GDELTSource):
            return source
    raise HTTPException(status_code=404, detail="GDELT source is not configured")


def create_app(
    config: PREACTConfig | None = None,
    orchestrator: DataIngestionOrchestrator | None = None,
) -> FastAPI:
    """Create a FastAPI application configured for the PREACT platform."""

    if orchestrator is None:
        if config is None:
            raise ValueError("Either a configuration or an orchestrator must be provided")
        orchestrator = DataIngestionOrchestrator(config)

    app = FastAPI(title="PREACT Platform API", version="0.1.0")
    app.state.orchestrator = orchestrator

    @app.get("/health")
    def health() -> Dict[str, Any]:
        sources = list(orchestrator.sources.keys())
        return {
            "status": "ok",
            "sources": sources,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @app.post("/ingest")
    def run_ingestion(request: IngestRequest) -> Dict[str, Any]:
        artifacts = orchestrator.run(
            lookback_days=request.lookback_days,
            end=request.end,
            persist=request.persist,
        )
        summary = _summarise_ingestion(artifacts)
        return {"status": "ok", "summary": summary}

    @app.get("/gdelt/events")
    def gdelt_events(
        country: str | None = Query(None, description="ISO country code filter"),
        theme: str | None = Query(None, description="GDELT theme identifier"),
        keyword: str | None = Query(None, description="Keyword to search"),
        tone_min: float | None = Query(None, description="Minimum tone filter"),
        tone_max: float | None = Query(None, description="Maximum tone filter"),
        limit: int = Query(100, ge=1, le=250),
        lookback_days: int = Query(7, ge=1, le=365),
    ) -> Dict[str, Any]:
        source = _find_gdelt_source(orchestrator)
        query = GDELTQuery(
            keywords=[keyword] if keyword else (),
            countries=[country] if country else (),
            themes=[theme] if theme else (),
            tone_min=tone_min,
            tone_max=tone_max,
        )
        result: IngestionResult = source.recent_events(
            days=lookback_days,
            query=query,
            limit=limit,
        )
        events = _serialise_events(result.data, limit)
        metadata = _normalise_metadata(result.metadata)
        return {"events": list(events), "metadata": metadata}

    return app
