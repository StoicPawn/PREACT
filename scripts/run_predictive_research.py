"""Run a reproducible geopolitical predictive benchmark on PREACT data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pandas as pd

from preact.feature_store.panel import build_relation_risk_panel
from preact.feature_store.event_panel import build_event_risk_panel
from preact.history.graph_store import HistoricalGraphStore
from preact.history.schema import KnowledgeMode
from preact.history.warehouse import HistoricalWarehouse
from preact.models.benchmark_suite import ModelBenchmark, bootstrap_dependence_diagnostic, evaluate_prediction_frame, run_benchmark_suite
from preact.models.ensemble import sequential_oos_ensemble
from preact.models.ablation import DEFAULT_FAMILIES, run_family_ablation
from preact.models.placebo import run_placebo_benchmark
from preact.models.sliced_evaluation import temporal_slice_metrics
from preact.models.stress_tests import run_unseen_entity_stress
from preact.models.experiment_manifest import build_manifest
from preact.models.oos_lineage import benchmark_suite_feature_lineage
from preact.models.reporting import attach_prediction_feature_provenance, benchmark_diagnostics, temporal_fold_audits, validate_benchmark_suite_fold_audits, validate_prediction_fold_audits
from preact.models.research_governance import evaluate_research_promotion
from preact.models.sealed_holdout import (
    create_holdout_seal,
    development_view,
    save_holdout_seal_once,
    seal_fingerprint,
    validate_holdout_commitment,
)


def _serialize_models(models):
    """Keep dependence-aware uncertainty diagnostics in every nested report."""
    return {name: benchmark_diagnostics(model) for name, model in models.items()}


def _require_suite_fold_audits(suite, *, context: str):
    """Treat a produced suite with invalid OOS provenance as corruption, not unavailability."""
    try:
        return validate_benchmark_suite_fold_audits(suite)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"{context} produced an invalid OOS artifact: {exc}") from exc


def _require_suite_feature_lineage(suite, fingerprints, *, context: str):
    """Persist nested OOS lineage only after temporal and PIT provenance validation."""
    try:
        return benchmark_suite_feature_lineage(suite, fingerprints)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"{context} produced invalid point-in-time lineage: {exc}") from exc


DEFAULT_FEATURES = ("cow_nmc:cinc", "cow_nmc:milex", "cow_nmc:milper", "cow_nmc:tpop", "cow_nmc:upop", "cow_nmc:energy", "cow_nmc:irst", "cow_nmc:pec")


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _cutoffs(start: datetime, end: datetime, step_days: int) -> list[datetime]:
    out = []
    cursor = start
    step = timedelta(days=int(step_days))
    while cursor <= end:
        out.append(cursor)
        cursor += step
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-db", default="data/history/preact_history.duckdb")
    parser.add_argument("--graph-db", default="data/history/preact_graph.duckdb")
    parser.add_argument("--start", required=True); parser.add_argument("--end", required=True)
    parser.add_argument("--step-days", type=int, default=365); parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument("--target-relation", default="militarized_interstate_dispute")
    parser.add_argument("--target-event", default=None)
    parser.add_argument("--knowledge-mode", choices=[mode.value for mode in KnowledgeMode], default=KnowledgeMode.STRICT_AS_KNOWN.value)
    parser.add_argument("--entity", action="append", default=[]); parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--min-train-dates", type=int, default=20); parser.add_argument("--calibration-dates", type=int, default=5); parser.add_argument("--test-dates-per-fold", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000); parser.add_argument("--output-dir", default="data/experiments/predictive")
    parser.add_argument("--outcome-observed-through", default=None)
    parser.add_argument("--sealed-holdout-dir", default="data/experiments/holdouts")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-min-dates", type=int, default=5)
    parser.add_argument("--development-min-dates", type=int, default=20)
    parser.add_argument("--skip-geographic-stress", action="store_true"); parser.add_argument("--skip-placebo", action="store_true"); parser.add_argument("--run-ablations", action="store_true")
    args = parser.parse_args()

    history = HistoricalWarehouse(args.history_db); graph = HistoricalGraphStore(args.graph_db)
    target_name = args.target_event or args.target_relation; target_kind = "event" if args.target_event else "relation"
    entities = tuple(args.entity) if args.entity else tuple(history.list_entities(variable=target_name) if target_kind == "event" else graph.list_entities(relation_type=target_name))
    if not entities: raise SystemExit(f"No entities available for {target_kind} target {target_name}")
    start, end = _dt(args.start), _dt(args.end)
    if end < start: raise SystemExit("--end must be >= --start")
    cutoffs = _cutoffs(start, end, args.step_days); variables = tuple(args.feature) or DEFAULT_FEATURES; mode = KnowledgeMode(args.knowledge_mode)
    if mode is not KnowledgeMode.STRICT_AS_KNOWN:
        raise SystemExit("predictive research requires --knowledge-mode strict_as_known")
    observed = _dt(args.outcome_observed_through) if args.outcome_observed_through else None
    common = dict(warehouse=history, graph=graph, entity_ids=entities, cutoffs=cutoffs, feature_variables=variables, horizon_days=args.horizon_days, graph_recent_days=max(365,args.horizon_days), knowledge_mode=mode, outcome_observed_through=observed)
    dataset = build_event_risk_panel(target_variable=target_name, include_relation_history=True, include_temporal_dynamics=True, include_target_history=True, **common) if target_kind == "event" else build_relation_risk_panel(target_relation_type=target_name, include_event_history=True, include_temporal_dynamics=True, **common)
    if dataset.features.empty: raise SystemExit("No panel features produced")
    full_features = dataset.features.dropna(axis=1, how="all")
    if full_features.shape[1] == 0: raise SystemExit("No usable feature columns produced")

    safe_target_name = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in str(target_name)
    )
    holdout_path = (
        Path(args.sealed_holdout_dir)
        / f"{safe_target_name}_{int(args.horizon_days)}d.json"
    )
    candidate_seal = create_holdout_seal(
        full_features,
        dataset.target,
        target_name=target_name,
        horizon_days=args.horizon_days,
        holdout_fraction=args.holdout_fraction,
        min_holdout_dates=args.holdout_min_dates,
        min_development_dates=args.development_min_dates,
    )
    seal = save_holdout_seal_once(holdout_path, candidate_seal)
    validate_holdout_commitment(full_features, dataset.target, seal)
    features, research_target, research_fingerprints = development_view(
        full_features,
        dataset.target,
        dataset.feature_snapshot_fingerprints,
        seal,
    )

    suite = run_benchmark_suite(features, research_target, horizon_days=args.horizon_days, min_train_dates=args.min_train_dates, calibration_dates=args.calibration_dates, test_dates_per_fold=args.test_dates_per_fold, bootstrap_samples=args.bootstrap_samples)
    ensemble = sequential_oos_ensemble({name:model.predictions for name,model in suite.models.items()})
    ensemble_diagnostic = bootstrap_dependence_diagnostic(ensemble.predictions, samples=args.bootstrap_samples)
    ensemble_benchmark = ModelBenchmark("sequential_ensemble", ensemble.predictions, evaluate_prediction_frame(ensemble.predictions), ensemble_diagnostic.block, ensemble_diagnostic)
    benchmarks={**suite.models,"sequential_ensemble":ensemble_benchmark}
    decisions={name:evaluate_research_promotion(b,folds_used=len(suite.folds)) for name,b in benchmarks.items()}

    geographic_stress=None
    if not args.skip_geographic_stress:
        try:
            split, stress_suite=run_unseen_entity_stress(features,research_target,horizon_days=args.horizon_days,min_train_dates=args.min_train_dates,calibration_dates=args.calibration_dates,test_dates_per_fold=args.test_dates_per_fold,bootstrap_samples=max(100,args.bootstrap_samples//2))
        except (ValueError,TypeError) as exc:
            geographic_stress={"status":"not_run","reason":str(exc)}
        else:
            stress_audits=_require_suite_fold_audits(stress_suite, context="geographic stress")
            stress_lineage=_require_suite_feature_lineage(stress_suite, research_fingerprints, context="geographic stress")
            geographic_stress={"train_entities":list(split.train_entities),"test_entities":list(split.test_entities),"folds":stress_audits,"feature_lineage":stress_lineage,"models":_serialize_models(stress_suite.models)}
    placebo=None
    if not args.skip_placebo:
        p=run_placebo_benchmark(features,research_target,horizon_days=args.horizon_days,min_train_dates=args.min_train_dates,calibration_dates=args.calibration_dates,test_dates_per_fold=args.test_dates_per_fold,bootstrap_samples=max(100,args.bootstrap_samples//4))
        placebo_audits=_require_suite_fold_audits(p.benchmark, context="placebo")
        placebo_lineage=_require_suite_feature_lineage(p.benchmark, research_fingerprints, context="placebo")
        placebo={"max_brier_skill":p.max_brier_skill,"folds":placebo_audits,"feature_lineage":placebo_lineage,"models":_serialize_models(p.benchmark.models)}
    ablations={}
    if args.run_ablations:
        for family_name,prefixes in DEFAULT_FAMILIES.items():
            if not any(any(str(c).startswith(p) for p in prefixes) for c in features.columns): continue
            try:
                r=run_family_ablation(features,research_target,family_name=family_name,prefixes=prefixes,horizon_days=args.horizon_days,min_train_dates=args.min_train_dates,calibration_dates=args.calibration_dates,test_dates_per_fold=args.test_dates_per_fold,bootstrap_samples=max(100,args.bootstrap_samples//4))
            except (ValueError,TypeError) as exc:
                ablations[family_name]={"status":"not_run","reason":str(exc)}; continue
            ablation_audits=_require_suite_fold_audits(r.benchmark, context=f"ablation {family_name}")
            ablation_lineage=_require_suite_feature_lineage(r.benchmark, research_fingerprints, context=f"ablation {family_name}")
            ablations[family_name]={"removed_columns":list(r.removed_columns),"folds":ablation_audits,"feature_lineage":ablation_lineage,"models":_serialize_models(r.benchmark.models)}
    stability={name:[asdict(item) for item in temporal_slice_metrics(b.predictions)] for name,b in benchmarks.items()}
    manifest=build_manifest(features,research_target,target_name=target_name,horizon_days=args.horizon_days,knowledge_mode=mode.value,metadata={"history_db":args.history_db,"graph_db":args.graph_db,"step_days":args.step_days,"entities_requested":len(entities),"target_kind":target_kind,"target_name":target_name,"outcome_observed_through":observed.isoformat() if observed else None,"research_phase":"development","sealed_holdout_path":str(holdout_path),"sealed_holdout_fingerprint":seal_fingerprint(seal),"sealed_holdout_start":seal.holdout_start,"sealed_holdout_end":seal.holdout_end})
    output=Path(args.output_dir)/manifest.dataset_fingerprint[:16]; output.mkdir(parents=True,exist_ok=True)
    fold_audits=temporal_fold_audits(suite.folds)
    persisted_predictions={}
    for name,b in benchmarks.items():
        try:
            validate_prediction_fold_audits(b.predictions, fold_audits)
            persisted_predictions[name]=attach_prediction_feature_provenance(b.predictions, research_fingerprints)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"refusing to persist inconsistent OOS artifact for {name}: {exc}") from exc
    lineage={name:{"rows":len(frame),"unique_feature_snapshots":int(frame["feature_snapshot_fingerprint"].nunique())} for name,frame in persisted_predictions.items()}
    report={"manifest":manifest.to_dict(),"sealed_holdout":{"phase":"development","path":str(holdout_path),"fingerprint":seal_fingerprint(seal),"holdout_start":seal.holdout_start,"holdout_end":seal.holdout_end,"outcomes_revealed":False},"models":{name:{**benchmark_diagnostics(b),"promotion":asdict(decisions[name])} for name,b in benchmarks.items()},"folds":fold_audits,"feature_lineage":lineage,"ensemble_fold_weights":{str(k):dict(v) for k,v in ensemble.fold_weights.items()},"temporal_stability":stability,"geographic_stress":geographic_stress,"placebo":placebo,"ablations":ablations}
    (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,default=str),encoding="utf-8")
    for name,frame in persisted_predictions.items():
        if not frame.empty: frame.to_parquet(output/f"predictions_{name}.parquet",index=False)
    print(json.dumps({"experiment_dir":str(output),"fingerprint":manifest.dataset_fingerprint,"rows":manifest.rows,"events":manifest.events,"models":{name:{"brier_skill":b.metrics.brier_skill,"skill_ci_lower":b.brier_skill_interval.lower,"promotable":decisions[name].promotable,"reasons":list(decisions[name].reasons)} for name,b in benchmarks.items()},"placebo_max_brier_skill":placebo.get("max_brier_skill") if isinstance(placebo,dict) else None,"geographic_stress_status":geographic_stress.get("status","completed") if isinstance(geographic_stress,dict) else "skipped"},indent=2,default=str))


if __name__ == "__main__": main()
