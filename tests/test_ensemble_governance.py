import pandas as pd

from preact.models.benchmark_suite import (
    BenchmarkMetrics,
    ModelBenchmark,
    SkillInterval,
)
from preact.models.ensemble import sequential_oos_ensemble
from preact.models.research_governance import (
    ResearchPromotionPolicy,
    evaluate_research_promotion,
)


def _pred(model_shift: float):
    rows = []
    for fold in range(3):
        for i in range(4):
            actual = int(i == 0)
            rows.append(
                {
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=fold * 10 + i),
                    "entity_id": f"c{i}",
                    "fold": fold,
                    "actual": actual,
                    "probability": min(0.99, max(0.01, 0.2 + model_shift + 0.1 * actual)),
                    "baseline_probability": 0.25,
                }
            )
    return pd.DataFrame(rows)


def _well_calibrated_oos():
    rows = []
    for fold in range(4):
        for i in range(1250):
            rows.append({"fold": fold, "actual": int(i < 50), "probability": 0.04})
    return pd.DataFrame(rows)


def _strong_metrics():
    return BenchmarkMetrics(
        rows=5000,
        events=200,
        brier=0.08,
        hierarchical_baseline_brier=0.10,
        brier_skill=0.20,
        log_loss=0.3,
        roc_auc=0.8,
        average_precision=0.4,
        calibration_gap=0.01,
        worst_fold_brier_skill=0.05,
    )


def test_sequential_ensemble_only_uses_prior_fold_performance():
    result = sequential_oos_ensemble(
        {"a": _pred(0.0), "b": _pred(0.2)}
    )
    assert len(result.predictions) == 12
    assert result.fold_weights[0] == {"a": 0.5, "b": 0.5}
    assert result.fold_weights[1]["a"] > result.fold_weights[1]["b"]


def test_research_gate_requires_confident_positive_skill():
    metrics = _strong_metrics()
    good = ModelBenchmark(
        "m",
        _well_calibrated_oos(),
        metrics,
        SkillInterval(0.05, 0.2, 0.3, 1000),
    )
    decision = evaluate_research_promotion(good, folds_used=4)
    assert decision.promotable is True

    weak = ModelBenchmark(
        "m",
        _well_calibrated_oos(),
        metrics,
        SkillInterval(-0.02, 0.2, 0.3, 1000),
    )
    decision = evaluate_research_promotion(weak, folds_used=4)
    assert decision.promotable is False
    assert "skill_ci_positive" in decision.reasons


def test_research_gate_rejects_stale_metric_accounting():
    predictions = _well_calibrated_oos().iloc[:-1].copy()
    benchmark = ModelBenchmark(
        "m",
        predictions,
        _strong_metrics(),
        SkillInterval(0.05, 0.2, 0.3, 1000),
    )
    decision = evaluate_research_promotion(benchmark, folds_used=4)
    assert decision.promotable is False
    assert decision.checks["metric_accounting_consistent"] is False
    assert "metric_accounting_consistent" in decision.reasons


def test_research_gate_rejects_fold_count_metadata_drift():
    benchmark = ModelBenchmark(
        "m",
        _well_calibrated_oos(),
        _strong_metrics(),
        SkillInterval(0.05, 0.2, 0.3, 1000),
    )
    decision = evaluate_research_promotion(benchmark, folds_used=6)
    assert decision.promotable is False
    assert decision.checks["enough_folds"] is True
    assert decision.checks["fold_accounting_consistent"] is False
    assert "fold_accounting_consistent" in decision.reasons


def test_research_gate_rejects_hidden_temporal_calibration_drift():
    metrics = _strong_metrics()
    metrics = BenchmarkMetrics(**{**metrics.__dict__, "calibration_gap": 0.0})
    predictions = _well_calibrated_oos()
    predictions.loc[predictions["fold"] == 0, "probability"] = 0.14
    predictions.loc[predictions["fold"] == 1, "probability"] = 0.0
    benchmark = ModelBenchmark(
        "m",
        predictions,
        metrics,
        SkillInterval(0.05, 0.2, 0.3, 1000),
    )
    decision = evaluate_research_promotion(benchmark, folds_used=4)
    assert decision.promotable is False
    assert decision.checks["calibrated"] is True
    assert decision.checks["calibration_fold_stability"] is False
    assert "calibration_fold_stability" in decision.reasons


def test_research_gate_rejects_distributed_calibration_drift():
    predictions = _well_calibrated_oos()
    for fold, shift in enumerate((0.04, -0.04, 0.04, -0.04)):
        predictions.loc[predictions["fold"] == fold, "probability"] += shift
    benchmark = ModelBenchmark(
        "m",
        predictions,
        _strong_metrics(),
        SkillInterval(0.05, 0.2, 0.3, 1000),
    )
    decision = evaluate_research_promotion(benchmark, folds_used=4)
    assert decision.checks["calibration_fold_stability"] is True
    assert decision.checks["calibration_drift_dispersion"] is False
    assert "calibration_drift_dispersion" in decision.reasons


def test_research_gate_requires_event_evidence_in_every_calibration_fold():
    predictions = _well_calibrated_oos()
    predictions.loc[predictions["fold"] == 3, "actual"] = 0
    predictions.loc[predictions["fold"] == 3, "probability"] = 0.0
    metrics = BenchmarkMetrics(**{**_strong_metrics().__dict__, "events": 150})
    benchmark = ModelBenchmark(
        "m",
        predictions,
        metrics,
        SkillInterval(0.05, 0.2, 0.3, 1000),
    )
    decision = evaluate_research_promotion(benchmark, folds_used=4)
    assert decision.promotable is False
    assert decision.checks["metric_accounting_consistent"] is True
    assert decision.checks["calibration_fold_evidence"] is False
    assert "calibration_fold_evidence" in decision.reasons


def test_research_gate_requires_nonevent_evidence_in_every_calibration_fold():
    predictions = _well_calibrated_oos()
    # Calibration is not identifiable from a fold containing only positives,
    # just as it is not identifiable from an event-free fold.
    predictions.loc[predictions["fold"] == 3, "actual"] = 1
    predictions.loc[predictions["fold"] == 3, "probability"] = 1.0
    metrics = BenchmarkMetrics(**{**_strong_metrics().__dict__, "events": 1400})
    benchmark = ModelBenchmark(
        "m",
        predictions,
        metrics,
        SkillInterval(0.05, 0.2, 0.3, 1000),
    )
    decision = evaluate_research_promotion(benchmark, folds_used=4)
    assert decision.promotable is False
    assert decision.checks["metric_accounting_consistent"] is True
    assert decision.checks["calibration_fold_evidence"] is False
    assert "calibration_fold_evidence" in decision.reasons


def test_research_gate_rejects_local_ece_hidden_by_pooled_calibration():
    predictions = _well_calibrated_oos()
    fold_zero = predictions["fold"] == 0
    fold_zero_index = predictions.index[fold_zero]
    # Keep fold-level mean error modest while making reliability sharply wrong
    # inside fixed probability bins: positives receive p=0 and many negatives
    # receive p=0.12. Other folds remain well calibrated, so pooled ECE stays low.
    predictions.loc[fold_zero_index, "probability"] = 0.0
    predictions.loc[fold_zero_index[50:675], "probability"] = 0.12
    benchmark = ModelBenchmark(
        "m",
        predictions,
        _strong_metrics(),
        SkillInterval(0.05, 0.2, 0.3, 1000),
    )
    decision = evaluate_research_promotion(benchmark, folds_used=4)
    assert decision.checks["calibration_fold_stability"] is True
    assert decision.checks["calibration_ece"] is True
    assert decision.checks["calibration_worst_fold_ece"] is False
    assert "calibration_worst_fold_ece" in decision.reasons
