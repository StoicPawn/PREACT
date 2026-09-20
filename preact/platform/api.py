"""Research-grade API for Historical Atlas / Replay / Scenario platform."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import os

from fastapi import FastAPI,HTTPException,Query
from pydantic import BaseModel,Field

from preact.history.coverage import build_coverage_report
from preact.history.document_store import HistoricalDocumentStore
from preact.history.graph_store import HistoricalGraphStore
from preact.history.warehouse import HistoricalWarehouse
from preact.history.schema import KnowledgeMode
from preact.history.source_plan import SOURCE_PLAN
from preact.feature_store.panel import build_relation_risk_panel
from preact.feature_store.temporal import entity_feature_frame
from preact.models.governance import evaluate_promotion
from preact.models.panel_risk import panel_walk_forward_backtest
from preact.models.scenario_dynamics import ScenarioShock
from preact.platform.services import HistoricalAtlasService,ScenarioLabService


class PanelReplayRequest(BaseModel):
    entity_ids:list[str]=Field(min_length=1)
    start:datetime
    end:datetime
    step_days:int=Field(default=365,ge=1,le=3650)
    feature_variables:list[str]=Field(default_factory=list)
    target_relation_type:str="militarized_interstate_dispute"
    horizon_days:int=Field(default=365,ge=1,le=3650)
    recent_graph_days:int=Field(default=365,ge=1,le=3650)
    knowledge_mode:KnowledgeMode=KnowledgeMode.STRICT_AS_KNOWN
    min_train_dates:int=Field(default=20,ge=5)
    test_dates_per_fold:int=Field(default=5,ge=1)


class ScenarioShockPayload(BaseModel):
    variable:str
    step:int=Field(ge=0)
    operation:str
    value:float


class DynamicsScenarioRequest(BaseModel):
    entity_id:str
    variables:list[str]=Field(min_length=1)
    start:datetime
    end:datetime
    history_step_days:int=Field(default=365,ge=1,le=3650)
    future_steps:int=Field(default=10,ge=1,le=200)
    runs:int=Field(default=1000,ge=10,le=50000)
    knowledge_mode:KnowledgeMode=KnowledgeMode.STRICT_AS_KNOWN
    alpha:float=Field(default=1.0,gt=0)
    seed:int=42
    shocks:list[ScenarioShockPayload]=Field(default_factory=list)


def _cutoffs(start:datetime,end:datetime,step_days:int)->list[datetime]:
    if start.tzinfo is None:
        start=start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end=end.replace(tzinfo=timezone.utc)
    if end<start:
        raise HTTPException(status_code=400,detail="end must be >= start")
    out=[]
    cursor=start
    delta=timedelta(days=step_days)
    while cursor<=end:
        out.append(cursor)
        cursor+=delta
    return out


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


@app.get("/v1/sources/plan")
def source_plan()->dict:
    return {"sources":[{
        "source_id":x.source_id,
        "wave":x.wave,
        "order":x.order,
        "role":x.role,
        "dependencies":list(x.dependencies),
        "required_for":list(x.required_for),
    } for x in SOURCE_PLAN]}


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
    knowledge_mode:KnowledgeMode=Query(KnowledgeMode.STRICT_AS_KNOWN),
)->dict:
    cutoff=_parse(knowledge_cutoff,"knowledge_cutoff")
    world=_parse(valid_at,"valid_at") or cutoff
    state=atlas.state(
        entity_id=entity_id,
        knowledge_cutoff=cutoff,
        valid_at=world,
        variable=variable,
        knowledge_mode=knowledge_mode,
    )
    return {
        "entity_id":state.entity_id,
        "knowledge_cutoff":state.knowledge_cutoff.isoformat(),
        "valid_at":state.valid_at.isoformat(),
        "knowledge_mode":knowledge_mode.value,
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



@app.post("/v1/replay/panel-relation")
def replay_panel_relation(request:PanelReplayRequest)->dict:
    dates=_cutoffs(request.start,request.end,request.step_days)
    dataset=build_relation_risk_panel(
        warehouse=warehouse,
        graph=graph,
        entity_ids=request.entity_ids,
        cutoffs=dates,
        feature_variables=request.feature_variables,
        target_relation_type=request.target_relation_type,
        horizon_days=request.horizon_days,
        graph_recent_days=request.recent_graph_days,
        knowledge_mode=request.knowledge_mode,
    )
    if dataset.features.empty or dataset.features.shape[1]==0:
        raise HTTPException(status_code=400,detail=(
            "no usable features at these cutoffs; try later cutoffs or explicitly "
            "select retrospective mode for modern historical reconstructions"
        ))
    result=panel_walk_forward_backtest(
        dataset.features,
        dataset.target,
        horizon_days=request.horizon_days,
        min_train_dates=request.min_train_dates,
        test_dates_per_fold=request.test_dates_per_fold,
    )
    decision=evaluate_promotion(result.metrics)
    preview=(
        result.predictions.tail(500).to_dict(orient="records")
        if not result.predictions.empty else []
    )
    return {
        "knowledge_mode":request.knowledge_mode.value,
        "target_relation_type":request.target_relation_type,
        "horizon_days":request.horizon_days,
        "entities":list(dataset.entities),
        "feature_columns":list(dataset.features.columns),
        "folds_used":result.folds_used,
        "folds_skipped":result.folds_skipped,
        "metrics":asdict(result.metrics),
        "promotion":asdict(decision),
        "prediction_preview":preview,
    }


@app.post("/v1/scenario/dynamics")
def scenario_dynamics(request:DynamicsScenarioRequest)->dict:
    dates=_cutoffs(request.start,request.end,request.history_step_days)
    frame=entity_feature_frame(
        warehouse,
        entity_id=request.entity_id,
        cutoffs=dates,
        variables=request.variables,
        knowledge_mode=request.knowledge_mode,
    )
    frame=frame.reindex(columns=request.variables).ffill().dropna()
    if len(frame)<10:
        raise HTTPException(status_code=400,detail="at least 10 complete historical states are required")
    initial={str(k):float(v) for k,v in frame.iloc[-1].items()}
    shocks=[
        ScenarioShock(
            variable=item.variable,
            step=item.step,
            operation=item.operation,
            value=item.value,
        )
        for item in request.shocks
    ]
    try:
        result=ScenarioLabService.simulate_dynamics(
            history=frame,
            initial_state=initial,
            steps=request.future_steps,
            runs=request.runs,
            shocks=shocks,
            seed=request.seed,
            alpha=request.alpha,
        )
    except (ValueError,KeyError,RuntimeError) as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc
    return {
        "entity_id":request.entity_id,
        "knowledge_mode":request.knowledge_mode.value,
        "model":"ridge_var1_monte_carlo",
        "interpretation":"conditional counterfactual dynamics; not an identified causal effect",
        "variables":list(result.variables),
        "runs":result.runs,
        "quantiles":[result.lower_quantile,0.5,result.upper_quantile],
        "lower":result.lower.reset_index().to_dict(orient="records"),
        "median":result.median.reset_index().to_dict(orient="records"),
        "upper":result.upper.reset_index().to_dict(orient="records"),
    }
