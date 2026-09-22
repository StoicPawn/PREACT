"""Stable serialization helpers for predictive research diagnostics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping

import pandas as pd

from .benchmark_suite import BenchmarkSuiteResult, ModelBenchmark
from .calibration_diagnostics import temporal_calibration_diagnostics
from .temporal_cv import TemporalFold


def temporal_fold_audits(folds: Iterable[TemporalFold]) -> list[dict[str, object]]:
    """Serialize temporal folds without duplicating the split contract.

    Keeping report serialization delegated to ``TemporalFold.audit_record``
    prevents experiment artifacts from drifting away from the canonical
    anti-leakage definition as new embargo boundaries are added.
    """
    return [fold.audit_record() for fold in folds]


def validate_prediction_fold_audits(
    predictions: pd.DataFrame,
    audits: Iterable[Mapping[str, object]],
) -> None:
    """Reject OOS artifacts inconsistent with their persisted temporal contract.

    The report and parquet predictions are separate files, so serialization alone
    cannot guarantee that they describe the same evaluation windows. Validate
    fold identity, test-window boundaries, unique test-date counts, and both
    embargo inequalities before an experiment is persisted or consumed.
    """
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
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"temporal fold audit {fold} is incomplete") from exc
        boundaries = [fit_end, fit_cutoff, calibration_start, calibration_end, training_cutoff, test_start, test_end]
        if any(pd.isna(value) for value in boundaries):
            raise ValueError(f"temporal fold audit {fold} contains invalid boundaries")
        if not (fit_end < fit_cutoff <= calibration_start <= calibration_end < training_cutoff <= test_start <= test_end):
            raise ValueError(f"temporal fold audit {fold} violates embargo ordering")

        fold_dates = checked.loc[checked["_audit_fold"] == fold, "_audit_date"]
        if fold_dates.empty or fold_dates.min() != test_start or fold_dates.max() != test_end:
            raise ValueError(f"prediction dates do not match test window for fold {fold}")
        if int(fold_dates.nunique()) != expected_test_dates:
            raise ValueError(f"prediction test-date count does not match audit for fold {fold}")


def validate_benchmark_suite_fold_audits(
    suite: BenchmarkSuiteResult,
) -> list[dict[str, object]]:
    """Validate every model in a suite against the suite's temporal split contract.

    Stress, placebo and ablation suites are first-class OOS evidence too. Keeping
    this check at suite level prevents nested diagnostics from bypassing the same
    fail-closed temporal integrity checks used by the primary benchmark.
    """
    audits = temporal_fold_audits(suite.folds)
    for name, benchmark in suite.models.items():
        try:
            validate_prediction_fold_audits(benchmark.predictions, audits)
        except ValueError as exc:
            raise ValueError(f"benchmark {name} violates temporal fold audit: {exc}") from exc
    return audits


def benchmark_diagnostics(benchmark: ModelBenchmark) -> dict[str, Any]:
    """Serialize benchmark uncertainty and OOS calibration diagnostics.

    The dependence-aware block interval remains the primary uncertainty measure;
    the IID interval and width ratio are emitted only as diagnostics so reports
    can expose when serial dependence materially changes uncertainty. Calibration
    drift is computed only from the benchmark's already-OOS predictions: it is a
    diagnostic and never refits or tunes a model on evaluation observations.
    """
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