"""Stable serialization helpers for predictive research diagnostics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping

import pandas as pd

from .benchmark_suite import BenchmarkSuiteResult, ModelBenchmark
from .calibration_diagnostics import temporal_calibration_diagnostics
from .temporal_cv import TemporalFold


def temporal_fold_audits(folds: Iterable[TemporalFold]) -> list[dict[str, object]]:
    """Serialize temporal folds without duplicating the split contract."""
    return [fold.audit_record() for fold in folds]


def attach_prediction_feature_provenance(
    predictions: pd.DataFrame,
    feature_snapshot_fingerprints: pd.Series,
) -> pd.DataFrame:
    """Attach exact point-in-time feature vintage fingerprints to OOS rows.

    Fails closed when an OOS entity/date cannot be mapped uniquely to the
    feature snapshot from which the model input was materialized.
    """
    required = {"date", "entity_id"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing feature provenance keys: {sorted(missing)}")
    if not isinstance(feature_snapshot_fingerprints.index, pd.MultiIndex):
        raise TypeError("feature snapshot fingerprints must use a panel MultiIndex")
    if feature_snapshot_fingerprints.index.has_duplicates:
        raise ValueError("feature snapshot provenance contains duplicate panel keys")
    if feature_snapshot_fingerprints.isna().any():
        raise ValueError("feature snapshot provenance contains missing fingerprints")

    provenance = feature_snapshot_fingerprints.rename("feature_snapshot_fingerprint").reset_index()
    if not {"date", "entity_id"}.issubset(provenance.columns):
        raise ValueError("feature snapshot provenance index must contain date and entity_id")
    provenance["date"] = pd.to_datetime(provenance["date"], errors="raise", utc=True)
    provenance["entity_id"] = provenance["entity_id"].astype(str)

    out = predictions.drop(columns=["feature_snapshot_fingerprint"], errors="ignore").copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise", utc=True)
    out["entity_id"] = out["entity_id"].astype(str)
    out = out.merge(provenance, on=["date", "entity_id"], how="left", validate="many_to_one", sort=False)
    if out["feature_snapshot_fingerprint"].isna().any():
        raise ValueError("missing point-in-time feature provenance for OOS predictions")
    if not out["feature_snapshot_fingerprint"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("invalid feature snapshot fingerprint in OOS predictions")
    return out


def validate_prediction_fold_audits(
    predictions: pd.DataFrame,
    audits: Iterable[Mapping[str, object]],
) -> None:
    """Reject OOS artifacts inconsistent with their persisted temporal contract."""
    required_prediction_columns = {"fold", "date"}
    missing = required_prediction_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing temporal audit columns: {sorted(missing)}")

    audit_rows = list(audits)
    if not audit_rows:
        raise ValueError("temporal fold audits must not be empty")
    audit_by_fold: dict[int, Mapping[str, object]] = {}
    for audit in audit_rows:
        try:
            fold = int(audit["fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("temporal fold audit has invalid fold id") from exc
        if fold in audit_by_fold:
            raise ValueError(f"duplicate temporal fold audit: {fold}")
        audit_by_fold[fold] = audit

    fold_values = pd.to_numeric(predictions["fold"], errors="coerce")
    if fold_values.isna().any() or (fold_values % 1 != 0).any():
        raise ValueError("predictions contain invalid fold ids")
    observed_folds = {int(value) for value in fold_values.unique()}
    if observed_folds != set(audit_by_fold):
        raise ValueError("prediction folds do not match temporal fold audits")

    dates = pd.to_datetime(predictions["date"], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("predictions contain invalid dates")
    checked = predictions.assign(_audit_fold=fold_values.astype(int), _audit_date=dates)

    for fold, audit in audit_by_fold.items():
        try:
            fit_end = pd.to_datetime(audit["fit_end"], utc=True)
            fit_cutoff = pd.to_datetime(audit["fit_cutoff"], utc=True)
            calibration_start = pd.to_datetime(audit["calibration_start"], utc=True)
            calibration_end = pd.to_datetime(audit["calibration_end"], utc=True)
            training_cutoff = pd.to_datetime(audit["training_cutoff"], utc=True)
            test_start = pd.to_datetime(audit["test_start"], utc=True)
            test_end = pd.to_datetime(audit["test_end"], utc=True)
            expected_test_dates = int(audit["test_dates"])
            expected_values = pd.to_datetime(audit["test_date_values"], errors="raise", utc=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"temporal fold audit {fold} is incomplete") from exc
        boundaries = [fit_end, fit_cutoff, calibration_start, calibration_end, training_cutoff, test_start, test_end]
        if any(pd.isna(value) for value in boundaries):
            raise ValueError(f"temporal fold audit {fold} contains invalid boundaries")
        if not (fit_end < fit_cutoff <= calibration_start <= calibration_end < training_cutoff <= test_start <= test_end):
            raise ValueError(f"temporal fold audit {fold} violates embargo ordering")
        expected_set = set(expected_values)
        if len(expected_values) != expected_test_dates or len(expected_set) != expected_test_dates:
            raise ValueError(f"temporal fold audit {fold} has inconsistent exact test dates")
        if min(expected_set) != test_start or max(expected_set) != test_end:
            raise ValueError(f"temporal fold audit {fold} exact dates do not match test window")

        fold_dates = checked.loc[checked["_audit_fold"] == fold, "_audit_date"]
        observed_dates = set(fold_dates.unique())
        if observed_dates != expected_set:
            raise ValueError(f"prediction dates do not match exact audited test dates for fold {fold}")


def validate_benchmark_suite_fold_audits(
    suite: BenchmarkSuiteResult,
) -> list[dict[str, object]]:
    """Validate every model in a suite against the suite's temporal split contract."""
    audits = temporal_fold_audits(suite.folds)
    for name, benchmark in suite.models.items():
        try:
            validate_prediction_fold_audits(benchmark.predictions, audits)
        except ValueError as exc:
            raise ValueError(f"benchmark {name} violates temporal fold audit: {exc}") from exc
    return audits


def benchmark_diagnostics(benchmark: ModelBenchmark) -> dict[str, Any]:
    """Serialize benchmark uncertainty and OOS calibration diagnostics."""
    payload: dict[str, Any] = {
        "name": benchmark.name,
        "metrics": asdict(benchmark.metrics),
        "brier_skill_interval": asdict(benchmark.brier_skill_interval),
    }
    diagnostic = benchmark.dependence_diagnostic
    payload["dependence_diagnostic"] = (
        None
        if diagnostic is None
        else {
            "block": asdict(diagnostic.block),
            "iid": asdict(diagnostic.iid),
            "width_ratio": diagnostic.width_ratio,
            "iid_understates_uncertainty": (
                diagnostic.width_ratio is not None and diagnostic.width_ratio > 1.0
            ),
        }
    )
    predictions = benchmark.predictions
    if predictions is None:
        payload["calibration_drift"] = None
    else:
        payload["calibration_drift"] = asdict(temporal_calibration_diagnostics(predictions))
    return payload
