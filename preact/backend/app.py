"""FastAPI application exposing PREACT backend services."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd

try:  # pragma: no cover - runtime dependency management
    from pydantic import BaseModel, Field, validator
except ModuleNotFoundError:  # pragma: no cover - fallback when Pydantic is unavailable
    from ._pydantic_stub import install_pydantic_stub

    install_pydantic_stub()
    from pydantic import BaseModel, Field, validator  # type: ignore

from ..analytics import StateGraph, build_state_graph
from ..config import PREACTConfig
from ..data_ingestion.orchestrator import DataIngestionOrchestrator
from ..data_ingestion.sources import GDELTQuery, GDELTSource, IngestionResult
from ..simulation.service import SimulationService
from ..simulation.storage import SimulationRepository
from ..simulation.templates import default_templates

try:  # pragma: no cover - runtime dependency management
    from fastapi import FastAPI, HTTPException, Query
except ModuleNotFoundError:  # pragma: no cover - fallback when FastAPI is unavailable
    from ._fastapi_stub import install_fastapi_stub

    install_fastapi_stub()
    from fastapi import FastAPI, HTTPException, Query  # type: ignore


class TaxBracketPayload(BaseModel):
    threshold: float = Field(..., gt=0)
    rate: float = Field(..., ge=0, le=1)


class PolicyPayload(BaseModel):
    brackets: list[TaxBracketPayload]
    base_deduction: float = Field(..., ge=0)
    child_subsidy: float = Field(..., ge=0)
    unemployment_benefit: float | None = Field(default=None, ge=0)

    @validator('brackets')
    def _validate_brackets(cls, value: list[TaxBracketPayload]) -> list[TaxBracketPayload]:
        if not value:
            raise ValueError('At least one tax bracket must be provided')
        return value


class ShockPayload(BaseModel):
    name: str
    intensity: float = Field(..., ge=0)
    start_tick: int = Field(0, ge=0)
    end_tick: int | None = Field(default=None, ge=0)

    @validator('end_tick')
    def _check_end_tick(cls, value: int | None, values: Dict[str, Any]) -> int | None:
        if value is not None and value < values.get('start_tick', 0):
            raise ValueError('end_tick must be greater or equal to start_tick')
        return value


class SimulationRunRequest(BaseModel):
    template: str
    horizon: int | None = Field(None, ge=1, le=36)
    seed: int = Field(42, ge=0)
    policy: PolicyPayload | None = None
    base_policy: PolicyPayload | None = None
    shock: ShockPayload | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


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


def _serialise_records(frame: pd.DataFrame, limit: int | None = None) -> Iterable[Dict[str, Any]]:
    if frame.empty:
        return []
    subset = frame
    if limit is not None:
        subset = subset.head(limit)
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


def _serialise_events(frame: pd.DataFrame, limit: int) -> Iterable[Dict[str, Any]]:
    columns = [
        "event_id",
        "event_date",
        "country",
        "actor1_country",
        "actor2_country",
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
    subset = frame.loc[:, available]
    return _serialise_records(subset, limit)


def _find_gdelt_source(orchestrator: DataIngestionOrchestrator) -> GDELTSource:
    for source in orchestrator.sources.values():
        if isinstance(source, GDELTSource):
            return source
    raise HTTPException(status_code=404, detail="GDELT source is not configured")


def create_app(
    config: PREACTConfig | None = None,
    orchestrator: DataIngestionOrchestrator | None = None,
    simulation_service: SimulationService | None = None,
) -> FastAPI:
    """Create a FastAPI application configured for the PREACT platform."""

    if orchestrator is None:
        if config is None:
            raise ValueError("Either a configuration or an orchestrator must be provided")
        orchestrator = DataIngestionOrchestrator(config)

    if simulation_service is None:
        if config is None:
            raise ValueError("Simulation service requires a configuration when not provided explicitly")
        storage_root = Path(config.storage.root_dir)
        repository = SimulationRepository(
            storage_root / "simulation.duckdb",
            export_dir=storage_root / "simulation_exports",
        )
        simulation_service = SimulationService(repository=repository, templates=default_templates())

    app = FastAPI(title="PREACT Platform API", version="0.1.0")
    app.state.orchestrator = orchestrator
    app.state.simulation_service = simulation_service

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

    service = simulation_service

    @app.get("/simulation/scenarios")
    def simulation_scenarios() -> Dict[str, Any]:
        return {"scenarios": service.list_scenarios()}

    @app.post("/simulation/run")
    def simulation_run(request: SimulationRunRequest) -> Dict[str, Any]:
        try:
            summary = service.run(
                template_name=request.template,
                horizon=request.horizon,
                policy_payload=request.policy.dict() if request.policy else None,
                base_policy_payload=request.base_policy.dict() if request.base_policy else None,
                shock_payload=request.shock.dict() if request.shock else None,
                seed=request.seed,
                metadata=request.metadata,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return summary.to_dict()

    @app.get("/simulation/results/{run_id}")
    def simulation_results(run_id: str) -> Dict[str, Any]:
        try:
            results = service.fetch(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"run_id": run_id, "scenario": results.scenario_name, 
                "kpis": results.kpis(), "timeline": results.timeline.to_dict(orient="records")}

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

    @app.get("/gdelt/state-graph")
    def gdelt_state_graph(
        lookback_days: int = Query(30, ge=1, le=365),
        limit: int = Query(250, ge=10, le=250),
        min_events: int = Query(1, ge=1),
        min_weight: float | None = Query(None, ge=0),
        include_self_loops: bool = Query(False),
        country: str | None = Query(None, description="ISO country code filter"),
        theme: str | None = Query(None, description="GDELT theme identifier"),
        keyword: str | None = Query(None, description="Keyword to search"),
        tone_min: float | None = Query(None, description="Minimum tone filter"),
        tone_max: float | None = Query(None, description="Maximum tone filter"),
    ) -> Dict[str, Any]:
        source = _find_gdelt_source(orchestrator)
        query = GDELTQuery(
            keywords=[keyword] if keyword else (),
            countries=[country] if country else (),
            themes=[theme] if theme else (),
            tone_min=tone_min,
            tone_max=tone_max,
        )
        result = source.recent_events(days=lookback_days, query=query, limit=limit)
        graph: StateGraph = build_state_graph(
            result.data,
            weight_column="num_articles",
            min_events=min_events,
            min_weight=min_weight,
            include_self_loops=include_self_loops,
        )
        edges = list(_serialise_records(graph.edges))
        nodes = list(_serialise_records(graph.nodes))
        metadata = _normalise_metadata(result.metadata)
        metadata["edges"] = len(edges)
        metadata["nodes"] = len(nodes)
        metadata["query"] = result.metadata.get("query", "")
        return {"edges": edges, "nodes": nodes, "metadata": metadata}

    return app
