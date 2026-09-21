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
    calibration = temporal_calibration_diagnostics(benchmark.predictions)
    checks = {
        "enough_rows": int(metrics.rows) >= policy.min_oos_rows,
        "enough_events": int(metrics.events) >= policy.min_oos_events,
        "enough_folds": int(folds_used) >= policy.min_folds,
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
