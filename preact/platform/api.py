"""Research-grade API for Historical Atlas / Replay / Scenario platform."""

from __future__ import annotations

from datetime import datetime,timezone
import os

from fastapi import FastAPI,HTTPException,Query

from preact.history.coverage import build_coverage_report
from preact.history.document_store import HistoricalDocumentStore
from preact.history.graph_store import HistoricalGraphStore
from preact.history.warehouse import HistoricalWarehouse
from preact.platform.services import HistoricalAtlasService


HISTORY_DB=os.getenv("PREACT_HISTORY_DB","data/history/preact_history.duckdb")
GRAPH_DB=os.getenv("PREACT_GRAPH_DB","data/history/preact_graph.duckdb")
DOCUMENT_DB=os.getenv("PREACT_DOCUMENT_DB","data/history/preact_documents.duckdb")

warehouse=HistoricalWarehouse(HISTORY_DB)
graph=HistoricalGraphStore(GRAPH_DB)
documents=HistoricalDocumentStore(DOCUMENT_DB)
atlas=HistoricalAtlasService(warehouse,graph)
app=FastAPI(title="PREACT Historical-Geopolitical API",version="0.1.0")


def _parse(value:str|None,name:str)->datetime|None:
    if value is None:
        return None
    try:
        parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400,detail=f"invalid {name}") from exc
    if parsed.tzinfo is None:
        parsed=parsed.replace(tzinfo=timezone.utc)
    return parsed


@app.get("/health")
def health()->dict:
    return {"status":"ok","service":"historical-geopolitical-api"}


@app.get("/v1/sources/coverage")
def source_coverage()->dict:
    rows=build_coverage_report(warehouse,graph,documents)
    return {"sources":[{
        "source_id":r.source_id,
        "name":r.name,
        "status":r.status,
        "access":r.access,
        "replay_policy":r.replay_policy,
        "records":r.records,
        "relations":r.relations,
        "documents":r.documents,
        "earliest_valid":r.earliest_valid.isoformat() if r.earliest_valid else None,
        "latest_valid":r.latest_valid.isoformat() if r.latest_valid else None,
        "latest_retrieved":r.latest_retrieved.isoformat() if r.latest_retrieved else None,
    } for r in rows]}


@app.get("/v1/atlas/{entity_id}")
def atlas_state(
    entity_id:str,
    knowledge_cutoff:str=Query(...),
    valid_at:str|None=Query(None),
    variable:str|None=Query(None),
)->dict:
    cutoff=_parse(knowledge_cutoff,"knowledge_cutoff")
    world=_parse(valid_at,"valid_at") or cutoff
    state=atlas.state(
        entity_id=entity_id,
        knowledge_cutoff=cutoff,
        valid_at=world,
        variable=variable,
    )
    return {
        "entity_id":state.entity_id,
        "knowledge_cutoff":state.knowledge_cutoff.isoformat(),
        "valid_at":state.valid_at.isoformat(),
        "records":list(state.records),
        "relations":list(state.relations),
    }


@app.get("/v1/atlas/{entity_id}/documents")
def atlas_documents(
    entity_id:str,
    knowledge_cutoff:str=Query(...),
    q:str|None=Query(None),
    limit:int=Query(100,ge=1,le=1000),
)->dict:
    cutoff=_parse(knowledge_cutoff,"knowledge_cutoff")
    rows=documents.as_of(
        cutoff=cutoff,
        entity_id=entity_id,
        query=q,
        limit=limit,
    )
    return {"entity_id":entity_id,"knowledge_cutoff":cutoff.isoformat(),"documents":rows}
