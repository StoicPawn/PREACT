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


def test_sequential_ensemble_only_uses_prior_fold_performance():
    result = sequential_oos_ensemble(
        {"a": _pred(0.0), "b": _pred(0.2)}
    )
    assert len(result.predictions) == 12
    assert result.fold_weights[0] == {"a": 0.5, "b": 0.5}
    assert result.fold_weights[1]["a"] > result.fold_weights[1]["b"]


def test_research_gate_requires_confident_positive_skill():
    metrics = BenchmarkMetrics(
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
    good = ModelBenchmark(
        "m",
        pd.DataFrame(),
        metrics,
        SkillInterval(0.05, 0.2, 0.3, 1000),
    )
    decision = evaluate_research_promotion(good, folds_used=6)
    assert decision.promotable is True

    weak = ModelBenchmark(
        "m",
        pd.DataFrame(),
        metrics,
        SkillInterval(-0.02, 0.2, 0.3, 1000),
    )
    decision = evaluate_research_promotion(weak, folds_used=6)
    assert decision.promotable is False
    assert "skill_ci_positive" in decision.reasons
