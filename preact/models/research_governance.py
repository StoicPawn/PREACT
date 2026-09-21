"""Stricter promotion rules for geopolitical predictive research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from preact.models.calibration_diagnostics import temporal_calibration_diagnostics


@dataclass(frozen=True)
class ResearchPromotionPolicy:
    min_oos_rows: int = 1000
    min_oos_events: int = 50
    min_brier_skill: float = 0.02
    min_skill_ci_lower: float = 0.0
    min_worst_fold_skill: float = -0.10
    max_abs_calibration_gap: float = 0.03
    max_worst_fold_calibration_gap: float = 0.05
    max_fold_calibration_gap_std: float = 0.03
    max_expected_calibration_error: float = 0.05
    min_folds: int = 4
    min_calibration_rows_per_fold: int = 50
    min_calibration_events_per_fold: int = 1
    min_calibration_nonevents_per_fold: int = 1


@dataclass(frozen=True)
class ResearchPromotionDecision:
    promotable: bool
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]


def evaluate_research_promotion(
    benchmark,
    *,
    folds_used: int,
    policy: ResearchPromotionPolicy = ResearchPromotionPolicy(),
) -> ResearchPromotionDecision:
    metrics = benchmark.metrics
    interval = benchmark.brier_skill_interval
    predictions = benchmark.predictions
    calibration = temporal_calibration_diagnostics(predictions)

    # Promotion evidence must be self-consistent with the immutable OOS
    # prediction artifact. This prevents stale metric summaries from silently
    # surviving a change in censoring, fold construction, or prediction rows.
    observed_rows = len(predictions)
    observed_events = None
    if "actual" in predictions.columns:
        observed_events = int(predictions["actual"].sum())
    metric_accounting_consistent = (
        int(metrics.rows) == observed_rows
        and observed_events is not None
        and int(metrics.events) == observed_events
    )

    # Calibration gates are only meaningful when every temporal slice contains
    # enough genuinely OOS evidence from both outcome classes. In rare-event
    # settings a tiny, event-free, or all-event fold can otherwise look
    # deceptively stable/calibrated.
    fold_evidence_ok = False
    observed_folds = 0
    if {"fold", "actual"}.issubset(predictions.columns):
        fold_evidence = predictions.groupby("fold", sort=False)["actual"].agg(
            rows="size", events="sum"
        )
        observed_folds = len(fold_evidence)
        nonevents = fold_evidence["rows"] - fold_evidence["events"]
        fold_evidence_ok = (
            observed_folds >= policy.min_folds
            and bool(
                (fold_evidence["rows"] >= policy.min_calibration_rows_per_fold).all()
            )
            and bool(
                (fold_evidence["events"] >= policy.min_calibration_events_per_fold).all()
            )
            and bool(
                (nonevents >= policy.min_calibration_nonevents_per_fold).all()
            )
        )

    checks = {
        "metric_accounting_consistent": metric_accounting_consistent,
        "enough_rows": int(metrics.rows) >= policy.min_oos_rows,
        "enough_events": int(metrics.events) >= policy.min_oos_events,
        "enough_folds": int(folds_used) >= policy.min_folds,
        # Do not trust caller metadata independently of the OOS prediction
        # artifact: stale/manually supplied fold counts could otherwise make a
        # benchmark appear to have more temporal validation than it contains.
        "fold_accounting_consistent": observed_folds == int(folds_used),
        "calibration_fold_evidence": fold_evidence_ok,
        "positive_skill": (
            metrics.brier_skill is not None
            and float(metrics.brier_skill) >= policy.min_brier_skill
        ),
        "skill_ci_positive": (
            interval.lower is not None
            and float(interval.lower) > policy.min_skill_ci_lower
        ),
        "fold_stability": (
            metrics.worst_fold_brier_skill is not None
            and float(metrics.worst_fold_brier_skill)
            >= policy.min_worst_fold_skill
        ),
        "calibrated": (
            metrics.calibration_gap is not None
            and abs(float(metrics.calibration_gap))
            <= policy.max_abs_calibration_gap
        ),
        # Aggregate calibration can hide folds with large errors of opposite sign.
        # Promotion therefore requires stability on the already-OOS predictions.
        "calibration_fold_stability": (
            calibration.folds >= policy.min_folds
            and calibration.worst_abs_fold_gap is not None
            and calibration.worst_abs_fold_gap
            <= policy.max_worst_fold_calibration_gap
        ),
        # A worst-fold bound alone can still admit repeated, material drift. The
        # dispersion gate detects instability across the full OOS fold history.
        "calibration_drift_dispersion": (
            calibration.folds >= policy.min_folds
            and calibration.fold_gap_std is not None
            and calibration.fold_gap_std
            <= policy.max_fold_calibration_gap_std
        ),
        "calibration_ece": (
            calibration.expected_calibration_error is not None
            and calibration.expected_calibration_error
            <= policy.max_expected_calibration_error
        ),
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return ResearchPromotionDecision(
        promotable=all(checks.values()),
        checks=checks,
        reasons=reasons,
    )
