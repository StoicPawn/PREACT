"""Run a reproducible geopolitical predictive benchmark on PREACT data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pandas as pd

from preact.feature_store.panel import build_relation_risk_panel
from preact.history.graph_store import HistoricalGraphStore
from preact.history.schema import KnowledgeMode
from preact.history.warehouse import HistoricalWarehouse
from preact.models.benchmark_suite import (
    ModelBenchmark,
    block_bootstrap_brier_skill,
    evaluate_prediction_frame,
    run_benchmark_suite,
)
from preact.models.ensemble import sequential_oos_ensemble
from preact.models.experiment_manifest import build_manifest
from preact.models.research_governance import evaluate_research_promotion


DEFAULT_FEATURES = (
    "cow_nmc:cinc",
    "cow_nmc:milex",
    "cow_nmc:milper",
    "cow_nmc:tpop",
    "cow_nmc:upop",
    "cow_nmc:energy",
    "cow_nmc:irst",
    "cow_nmc:pec",
)


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
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--step-days", type=int, default=365)
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument(
        "--target-relation",
        default="militarized_interstate_dispute",
    )
    parser.add_argument(
        "--knowledge-mode",
        choices=[mode.value for mode in KnowledgeMode],
        default=KnowledgeMode.RETROSPECTIVE.value,
    )
    parser.add_argument("--entity", action="append", default=[])
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--min-train-dates", type=int, default=20)
    parser.add_argument("--calibration-dates", type=int, default=5)
    parser.add_argument("--test-dates-per-fold", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="data/experiments/predictive")
    args = parser.parse_args()

    history = HistoricalWarehouse(args.history_db)
    graph = HistoricalGraphStore(args.graph_db)
    entities = tuple(args.entity) or tuple(
        graph.list_entities(relation_type=args.target_relation)
    )
    if not entities:
        raise SystemExit("No entities available for target relation")

    start, end = _dt(args.start), _dt(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")
    cutoffs = _cutoffs(start, end, args.step_days)
    variables = tuple(args.feature) or DEFAULT_FEATURES
    mode = KnowledgeMode(args.knowledge_mode)

    dataset = build_relation_risk_panel(
        warehouse=history,
        graph=graph,
        entity_ids=entities,
        cutoffs=cutoffs,
        feature_variables=variables,
        target_relation_type=args.target_relation,
        horizon_days=args.horizon_days,
        graph_recent_days=max(365, args.horizon_days),
        knowledge_mode=mode,
        include_event_history=True,
    )
    if dataset.features.empty:
        raise SystemExit("No panel features produced")

    # Drop completely absent columns; per-fold imputers handle remaining gaps.
    features = dataset.features.dropna(axis=1, how="all")
    if features.shape[1] == 0:
        raise SystemExit("No usable feature columns produced")

    suite = run_benchmark_suite(
        features,
        dataset.target,
        horizon_days=args.horizon_days,
        min_train_dates=args.min_train_dates,
        calibration_dates=args.calibration_dates,
        test_dates_per_fold=args.test_dates_per_fold,
        bootstrap_samples=args.bootstrap_samples,
    )

    ensemble = sequential_oos_ensemble(
        {name: model.predictions for name, model in suite.models.items()}
    )
    ensemble_metrics = evaluate_prediction_frame(ensemble.predictions)
    ensemble_interval = block_bootstrap_brier_skill(
        ensemble.predictions,
        samples=args.bootstrap_samples,
    )
    ensemble_benchmark = ModelBenchmark(
        "sequential_ensemble",
        ensemble.predictions,
        ensemble_metrics,
        ensemble_interval,
    )

    benchmarks = {**suite.models, "sequential_ensemble": ensemble_benchmark}
    decisions = {
        name: evaluate_research_promotion(
            benchmark,
            folds_used=len(suite.folds),
        )
        for name, benchmark in benchmarks.items()
    }

    manifest = build_manifest(
        features,
        dataset.target,
        target_name=args.target_relation,
        horizon_days=args.horizon_days,
        knowledge_mode=mode.value,
        metadata={
            "history_db": args.history_db,
            "graph_db": args.graph_db,
            "step_days": args.step_days,
            "entities_requested": len(entities),
        },
    )

    output = Path(args.output_dir) / manifest.dataset_fingerprint[:16]
    output.mkdir(parents=True, exist_ok=True)

    report = {
        "manifest": manifest.to_dict(),
        "models": {
            name: {
                "metrics": asdict(benchmark.metrics),
                "brier_skill_interval": asdict(
                    benchmark.brier_skill_interval
                ),
                "promotion": asdict(decisions[name]),
            }
            for name, benchmark in benchmarks.items()
        },
        "folds": [
            {
                "fold": fold.fold,
                "training_cutoff": fold.training_cutoff.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
            }
            for fold in suite.folds
        ],
        "ensemble_fold_weights": {
            str(k): dict(v) for k, v in ensemble.fold_weights.items()
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    for name, benchmark in benchmarks.items():
        if not benchmark.predictions.empty:
            benchmark.predictions.to_parquet(
                output / f"predictions_{name}.parquet",
                index=False,
            )

    print(json.dumps(
        {
            "experiment_dir": str(output),
            "fingerprint": manifest.dataset_fingerprint,
            "rows": manifest.rows,
            "events": manifest.events,
            "models": {
                name: {
                    "brier_skill": benchmark.metrics.brier_skill,
                    "skill_ci_lower": benchmark.brier_skill_interval.lower,
                    "promotable": decisions[name].promotable,
                    "reasons": list(decisions[name].reasons),
                }
                for name, benchmark in benchmarks.items()
            },
        },
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
