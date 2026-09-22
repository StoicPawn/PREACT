import pandas as pd
import pytest

from preact.models.benchmark_suite import BenchmarkMetrics, DependenceDiagnostic, ModelBenchmark, SkillInterval
from preact.models.reporting import attach_prediction_feature_provenance, benchmark_diagnostics, temporal_fold_audits, validate_prediction_fold_audits
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
    predictions = pd.DataFrame({"fold": [0, 0, 0, 0, 1, 1, 1, 1], "actual": [0, 0, 1, 1, 0, 0, 1, 1], "probability": [0.2, 0.2, 0.8, 0.8, 0.0, 0.0, 0.6, 0.6]})
    benchmark = ModelBenchmark("candidate", predictions, metrics, interval)
    drift = benchmark_diagnostics(benchmark)["calibration_drift"]
    assert drift["folds"] == 2
    assert drift["worst_abs_fold_gap"] == 0.2
    assert drift["fold_gap_std"] == 0.1
    assert drift["expected_calibration_error"] >= abs(drift["weighted_gap"])


def _audit_fold() -> TemporalFold:
    return TemporalFold(
        fold=3,
        fit_dates=tuple(pd.to_datetime(["2020-01-01", "2020-02-01"])),
        calibration_dates=tuple(pd.to_datetime(["2020-04-01", "2020-05-01"])),
        test_dates=tuple(pd.to_datetime(["2020-07-01", "2020-08-01"])),
        fit_cutoff=pd.Timestamp("2020-03-01"), calibration_start=pd.Timestamp("2020-04-01"),
        training_cutoff=pd.Timestamp("2020-06-01"), test_start=pd.Timestamp("2020-07-01"), test_end=pd.Timestamp("2020-08-01"),
    )


def test_temporal_fold_audits_preserve_full_embargo_contract():
    fold = _audit_fold()
    payload = temporal_fold_audits([fold])
    assert payload == [fold.audit_record()]
    assert payload[0]["fit_cutoff"] == "2020-03-01T00:00:00"
    assert payload[0]["training_cutoff"] == "2020-06-01T00:00:00"
    assert payload[0]["fit_dates"] == 2
    assert payload[0]["calibration_dates"] == 2
    assert payload[0]["test_dates"] == 2
    assert payload[0]["test_date_values"] == ["2020-07-01T00:00:00", "2020-08-01T00:00:00"]


def test_prediction_fold_audit_validation_accepts_matching_oos_artifact():
    predictions = pd.DataFrame({"fold": [3, 3, 3], "date": ["2020-07-01", "2020-07-01", "2020-08-01"]})
    validate_prediction_fold_audits(predictions, temporal_fold_audits([_audit_fold()]))


def test_prediction_fold_audit_validation_rejects_window_or_embargo_mismatch():
    audits = temporal_fold_audits([_audit_fold()])
    predictions = pd.DataFrame({"fold": [3, 3], "date": ["2020-07-01", "2020-09-01"]})
    with pytest.raises(ValueError, match="exact audited test dates"):
        validate_prediction_fold_audits(predictions, audits)
    broken = [dict(audits[0], training_cutoff="2020-05-01T00:00:00")]
    valid_predictions = pd.DataFrame({"fold": [3, 3], "date": ["2020-07-01", "2020-08-01"]})
    with pytest.raises(ValueError, match="embargo ordering"):
        validate_prediction_fold_audits(valid_predictions, broken)


def test_prediction_fold_audit_rejects_interior_date_substitution_with_same_endpoints_and_count():
    fold = TemporalFold(
        fold=4,
        fit_dates=tuple(pd.to_datetime(["2020-01-01", "2020-02-01"])),
        calibration_dates=tuple(pd.to_datetime(["2020-04-01", "2020-05-01"])),
        test_dates=tuple(pd.to_datetime(["2020-07-01", "2020-08-15", "2020-10-01"])),
        fit_cutoff=pd.Timestamp("2020-03-01"), calibration_start=pd.Timestamp("2020-04-01"),
        training_cutoff=pd.Timestamp("2020-06-01"), test_start=pd.Timestamp("2020-07-01"), test_end=pd.Timestamp("2020-10-01"),
    )
    substituted = pd.DataFrame({"fold": [4, 4, 4], "date": ["2020-07-01", "2020-09-01", "2020-10-01"]})
    with pytest.raises(ValueError, match="exact audited test dates"):
        validate_prediction_fold_audits(substituted, temporal_fold_audits([fold]))


def test_oos_predictions_receive_exact_point_in_time_feature_fingerprint():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2020-07-01"), "A"), (pd.Timestamp("2020-08-01"), "A")],
        names=["date", "entity_id"],
    )
    fingerprints = pd.Series(["a" * 64, "b" * 64], index=idx, name="feature_snapshot_fingerprint")
    predictions = pd.DataFrame({"date": ["2020-07-01", "2020-08-01"], "entity_id": ["A", "A"], "fold": [3, 3]})
    enriched = attach_prediction_feature_provenance(predictions, fingerprints)
    assert enriched["feature_snapshot_fingerprint"].tolist() == ["a" * 64, "b" * 64]


def test_oos_feature_provenance_fails_closed_on_missing_or_invalid_vintage():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-07-01"), "A")], names=["date", "entity_id"])
    predictions = pd.DataFrame({"date": ["2020-07-01", "2020-08-01"], "entity_id": ["A", "A"]})
    with pytest.raises(ValueError, match="missing point-in-time feature provenance"):
        attach_prediction_feature_provenance(predictions, pd.Series(["a" * 64], index=idx))
    with pytest.raises(ValueError, match="invalid feature snapshot fingerprint"):
        attach_prediction_feature_provenance(predictions.iloc[:1], pd.Series(["not-a-sha"], index=idx))
