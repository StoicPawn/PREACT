"""Leakage-safe diagnostics for calibration drift in OOS prediction frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationDriftDiagnostics:
    """Summarise calibration stability across already-OOS temporal folds."""

    folds: int
    weighted_gap: float | None
    worst_abs_fold_gap: float | None
    fold_gap_std: float | None
    expected_calibration_error: float | None
    worst_fold_expected_calibration_error: float | None


def _fixed_bin_ece(y: np.ndarray, p: np.ndarray, *, bins: int) -> float:
    """Return ECE on fixed [0, 1] bins so slices remain comparable."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Include p == 1 in the final bin and keep fixed bins for reproducibility.
    assignments = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    ece = 0.0
    for bin_id in range(bins):
        mask = assignments == bin_id
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean() - y[mask].mean()))
    return float(ece)


def temporal_calibration_diagnostics(
    predictions: pd.DataFrame,
    *,
    bins: int = 10,
) -> CalibrationDriftDiagnostics:
    """Measure calibration drift without refitting on evaluation observations.

    ``predictions`` must contain OOS ``actual``, ``probability`` and ``fold``
    columns. ECE bins are fixed on [0, 1], rather than estimated from the
    evaluation sample, so diagnostics remain comparable across experiments.
    """
    required = {"actual", "probability", "fold"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"missing calibration columns: {sorted(missing)}")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if predictions.empty:
        return CalibrationDriftDiagnostics(0, None, None, None, None, None)

    # pandas.groupby drops NA keys by default. A missing fold label would thus
    # keep the row in global ECE/gap while silently excluding it from temporal
    # drift diagnostics, creating inconsistent evidence for promotion gates.
    if predictions["fold"].isna().any():
        raise ValueError("fold must be present for every OOS prediction")

    y = predictions["actual"].to_numpy(dtype=float)
    p = predictions["probability"].to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("actual and probability must be finite")
    # Calibration metrics assume Bernoulli outcomes. Silently accepting counts,
    # soft labels, or corrupted target encodings can produce plausible-looking
    # gaps/ECE and allow an invalid OOS artifact into promotion governance.
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("actual must contain binary outcomes in {0, 1}")
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("probability must lie in [0, 1]")

    gaps: list[float] = []
    fold_eces: list[float] = []
    for _, group in predictions.groupby("fold", sort=True):
        fold_y = group["actual"].to_numpy(dtype=float)
        fold_p = group["probability"].to_numpy(dtype=float)
        gaps.append(float(fold_p.mean() - fold_y.mean()))
        fold_eces.append(_fixed_bin_ece(fold_y, fold_p, bins=bins))

    ece = _fixed_bin_ece(y, p, bins=bins)
    weighted_gap = float(p.mean() - y.mean())
    worst = float(max(abs(gap) for gap in gaps))
    std = float(np.std(gaps, ddof=0)) if len(gaps) > 1 else 0.0
    return CalibrationDriftDiagnostics(
        folds=len(gaps),
        weighted_gap=weighted_gap,
        worst_abs_fold_gap=worst,
        fold_gap_std=std,
        expected_calibration_error=ece,
        worst_fold_expected_calibration_error=float(max(fold_eces)),
    )
