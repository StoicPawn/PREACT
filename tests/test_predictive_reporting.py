import pandas as pd

from preact.models.benchmark_suite import (
    BenchmarkMetrics,
    DependenceDiagnostic,
    ModelBenchmark,
    SkillInterval,
)
from preact.models.reporting import benchmark_diagnostics, temporal_fold_audits
from preact.models.temporal_cv import TemporalFold


def test_benchmark_diagnostics_preserves_dependence_aware_and_iid_intervals():
    metrics = BenchmarkMetrics(100, 5, 0.08, 0.10, 0.20, 0.3, 0.7, 0.2, 0.01, 0.05)
    block = SkillInterval(0.02, 0.18, 0.31, 500)
    iid = SkillInterval(0.08, 0.19, 0.27, 500)
    diagnostic = DependenceDiagnostic(block=block, iid=iid, width_ratio=1.526315789)
    benchmark = ModelBenchmark("candidate", None, metrics, block, diagnostic)

    payload = benchmark_diagnostics(benchmark)

    assert payload["brier_skill_interval"] == payload["dependence_diagnostic"]["block"]
    assert payload["dependence_diagnostic"]["iid"]["lower"] == 0.08
    assert payload["dependence_diagnostic"]["width_ratio"] > 1.5
    assert payload["calibration_drift"] is None


def test_benchmark_diagnostics_handles_legacy_benchmark_without_diagnostic():
    metrics = BenchmarkMetrics(0, 0, None, None, None, None, None, None, None, None)
    interval = SkillInterval(None, None, None, 0)
    benchmark = ModelBenchmark("legacy", None, metrics, interval)

    payload = benchmark_diagnostics(benchmark)

    assert payload["dependence_diagnostic"] is None
    assert payload["calibration_drift"] is None


def test_benchmark_diagnostics_serializes_temporal_calibration_drift_from_oos_predictions():
    metrics = BenchmarkMetrics(8, 4, 0.1, 0.12, 0.1, 0.4, 0.6, 0.4, -0.1, -0.2)
    interval = SkillInterval(-0.1, 0.1, 0.3, 100)
    predictions = pd.DataFrame(
        {
            "fold": [0, 0, 0, 0, 1, 1, 1, 1],
            "actual": [0, 0, 1, 1, 0, 0, 1, 1],
            "probability": [0.2, 0.2, 0.8, 0.8, 0.0, 0.0, 0.6, 0.6],
        }
    )
    benchmark = ModelBenchmark("candidate", predictions, metrics, interval)

    payload = benchmark_diagnostics(benchmark)

    drift = payload["calibration_drift"]
    assert drift["folds"] == 2
    assert drift["worst_abs_fold_gap"] == 0.2
    assert drift["fold_gap_std"] == 0.1
    assert drift["expected_calibration_error"] >= abs(drift["weighted_gap"])


def test_temporal_fold_audits_preserve_full_embargo_contract():
    fit = tuple(pd.to_datetime(["2020-01-01", "2020-02-01"]))
    calibration = tuple(pd.to_datetime(["2020-04-01", "2020-05-01"]))
    test = tuple(pd.to_datetime(["2020-07-01", "2020-08-01"]))
    fold = TemporalFold(
        fold=3,
        fit_dates=fit,
        calibration_dates=calibration,
        test_dates=test,
        fit_cutoff=pd.Timestamp("2020-03-01"),
        calibration_start=pd.Timestamp("2020-04-01"),
        training_cutoff=pd.Timestamp("2020-06-01"),
        test_start=pd.Timestamp("2020-07-01"),
        test_end=pd.Timestamp("2020-08-01"),
    )

    payload = temporal_fold_audits([fold])

    assert payload == [fold.audit_record()]
    assert payload[0]["fit_cutoff"] == "2020-03-01T00:00:00"
    assert payload[0]["training_cutoff"] == "2020-06-01T00:00:00"
    assert payload[0]["fit_dates"] == 2
    assert payload[0]["calibration_dates"] == 2
    assert payload[0]["test_dates"] == 2