"""One-shot public-data benchmark used by GitHub Actions research CI."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from preact.feature_store.event_panel import build_event_risk_panel
from preact.feature_store.panel import build_relation_risk_panel
from preact.history.graph_store import HistoricalGraphStore
from preact.history.schema import KnowledgeMode
from preact.history.warehouse import HistoricalWarehouse
from preact.history.wave1_runner import Wave1Runner
from preact.models.benchmark_suite import (
    ModelBenchmark,
    block_bootstrap_brier_skill,
    evaluate_prediction_frame,
    run_benchmark_suite,
)
from preact.models.ensemble import sequential_oos_ensemble
from preact.models.placebo import run_placebo_benchmark
from preact.models.research_governance import evaluate_research_promotion
from preact.models.stress_tests import run_unseen_entity_stress


FEATURES = (
    "cow_nmc:cinc",
    "cow_nmc:milex",
    "cow_nmc:milper",
    "cow_nmc:tpop",
    "cow_nmc:upop",
    "cow_nmc:energy",
    "cow_nmc:irst",
    "cow_nmc:pec",
)


def annual_cutoffs(start: int, end: int) -> list[datetime]:
    return [
        datetime(year, 1, 1, tzinfo=timezone.utc)
        for year in range(start, end + 1)
    ]


def top_relation_entities(
    graph: HistoricalGraphStore,
    *,
    relation_type: str,
    limit: int,
    start_year: int,
    end_year: int,
) -> list[str]:
    start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    end = datetime(end_year + 1, 1, 1, tzinfo=timezone.utc)
    with graph.connect() as conn:
        rows = conn.execute(
            """
            SELECT entity_id, COUNT(*) AS events
            FROM (
                SELECT subject_entity_id AS entity_id
                FROM historical_relations
                WHERE relation_type = ? AND valid_from >= ? AND valid_from < ?
                UNION ALL
                SELECT object_entity_id AS entity_id
                FROM historical_relations
                WHERE relation_type = ? AND valid_from >= ? AND valid_from < ?
            )
            GROUP BY entity_id
            ORDER BY events DESC, entity_id
            LIMIT ?
            """,
            [relation_type, start, end, relation_type, start, end, int(limit)],
        ).fetchall()
    return [str(row[0]) for row in rows]


def top_event_entities(
    warehouse: HistoricalWarehouse,
    *,
    variable: str,
    limit: int,
) -> list[str]:
    with warehouse.connect() as conn:
        rows = conn.execute(
            """
            SELECT entity_id, COUNT(DISTINCT source || ':' || source_ref) AS events
            FROM temporal_records
            WHERE variable = ?
            GROUP BY entity_id
            ORDER BY events DESC, entity_id
            LIMIT ?
            """,
            [variable, int(limit)],
        ).fetchall()
    return [str(row[0]) for row in rows]


def evaluate_panel(features, target, *, horizon_days: int) -> dict:
    x = features.dropna(axis=1, how="all")
    suite = run_benchmark_suite(
        x,
        target,
        horizon_days=horizon_days,
        min_train_dates=20,
        calibration_dates=5,
        test_dates_per_fold=5,
        bootstrap_samples=200,
        random_state=42,
    )
    ensemble = sequential_oos_ensemble(
        {name: model.predictions for name, model in suite.models.items()}
    )
    ensemble_benchmark = ModelBenchmark(
        "sequential_ensemble",
        ensemble.predictions,
        evaluate_prediction_frame(ensemble.predictions),
        block_bootstrap_brier_skill(
            ensemble.predictions,
            samples=200,
            seed=42,
        ),
    )
    models = {**suite.models, "sequential_ensemble": ensemble_benchmark}

    stress = None
    try:
        split, stress_suite = run_unseen_entity_stress(
            x,
            target,
            horizon_days=horizon_days,
            holdout_fraction=0.20,
            min_train_dates=20,
            calibration_dates=5,
            test_dates_per_fold=5,
            bootstrap_samples=100,
            random_state=42,
        )
        stress = {
            "test_entities": list(split.test_entities),
            "models": {
                name: {
                    "metrics": asdict(model.metrics),
                    "skill_interval": asdict(model.brier_skill_interval),
                }
                for name, model in stress_suite.models.items()
            },
        }
    except Exception as exc:
        stress = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    placebo_result = run_placebo_benchmark(
        x,
        target,
        horizon_days=horizon_days,
        min_train_dates=20,
        calibration_dates=5,
        test_dates_per_fold=5,
        bootstrap_samples=100,
        seed=42,
    )

    return {
        "rows": int(len(x)),
        "events": int(target.reindex(x.index).fillna(0).sum()),
        "features": list(x.columns),
        "folds": len(suite.folds),
        "models": {
            name: {
                "metrics": asdict(model.metrics),
                "skill_interval": asdict(model.brier_skill_interval),
                "promotion": asdict(
                    evaluate_research_promotion(
                        model,
                        folds_used=len(suite.folds),
                    )
                ),
            }
            for name, model in models.items()
        },
        "geographic_stress": stress,
        "placebo_max_brier_skill": placebo_result.max_brier_skill,
    }


def main() -> None:
    root = Path("data/public_predictive_benchmark")
    root.mkdir(parents=True, exist_ok=True)
    history_path = root / "history.duckdb"
    graph_path = root / "graph.duckdb"
    state_path = root / "state.sqlite3"
    hub_root = root / "hub"

    runner = Wave1Runner(
        hub_root=hub_root,
        history_db=history_path,
        graph_db=graph_path,
        state_db=state_path,
    )
    source_runs = {}

    cow = runner.run_cow_network()
    source_runs["cow_network"] = {
        "status": cow.status,
        "rows": cow.rows,
        "snapshots": cow.snapshots,
        "metadata": cow.metadata,
    }

    history = HistoricalWarehouse(history_path)
    graph = HistoricalGraphStore(graph_path)

    relation = "militarized_interstate_dispute"
    relation_entities = top_relation_entities(
        graph,
        relation_type=relation,
        limit=50,
        start_year=1950,
        end_year=2010,
    )
    relation_panel = build_relation_risk_panel(
        warehouse=history,
        graph=graph,
        entity_ids=relation_entities,
        cutoffs=annual_cutoffs(1950, 2010),
        feature_variables=FEATURES,
        target_relation_type=relation,
        horizon_days=365,
        graph_recent_days=1825,
        knowledge_mode=KnowledgeMode.RETROSPECTIVE,
        include_event_history=True,
        include_temporal_dynamics=True,
    )
    mids_result = evaluate_panel(
        relation_panel.features,
        relation_panel.target,
        horizon_days=365,
    )

    coup_result = None
    try:
        coup = runner.run_powell_thyne()
        source_runs["powell_thyne_coups"] = {
            "status": coup.status,
            "rows": coup.rows,
            "snapshots": coup.snapshots,
            "metadata": coup.metadata,
        }
        coup_entities = top_event_entities(
            history,
            variable="event:coup_attempt",
            limit=50,
        )
        coup_panel = build_event_risk_panel(
            warehouse=history,
            graph=graph,
            entity_ids=coup_entities,
            cutoffs=annual_cutoffs(1950, 2020),
            feature_variables=FEATURES,
            target_variable="event:coup_attempt",
            horizon_days=365,
            graph_recent_days=1825,
            knowledge_mode=KnowledgeMode.RETROSPECTIVE,
        )
        coup_result = evaluate_panel(
            coup_panel.features,
            coup_panel.target,
            horizon_days=365,
        )
    except Exception as exc:
        source_runs["powell_thyne_coups"] = {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        coup_result = {
            "status": "not_run",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "retrospective_oos",
        "caveat": (
            "Covariates come from modern historical releases. This tests "
            "out-of-time predictive structure, not contemporaneous data availability."
        ),
        "sources": source_runs,
        "targets": {
            "militarized_interstate_dispute": mids_result,
            "coup_attempt": coup_result,
        },
    }
    path = root / "public_benchmark_report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
