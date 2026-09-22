"""Temporal stability slices for OOS geopolitical forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import brier_score_loss


@dataclass(frozen=True)
class SliceMetric:
    label: str
    rows: int
    events: int
    brier_skill: float | None
    calibration_gap: float | None


def temporal_slice_metrics(
    predictions: pd.DataFrame,
    *,
    years_per_slice: int = 10,
) -> tuple[SliceMetric, ...]:
    if predictions.empty:
        return ()
    if years_per_slice < 1:
        raise ValueError("years_per_slice must be positive")
    frame = predictions.copy()
    years = pd.to_datetime(frame["date"]).dt.year.astype(int)
    base = (years // years_per_slice) * years_per_slice
    frame["_slice"] = base.astype(str) + "-" + (base + years_per_slice - 1).astype(str)
    output = []
    for label, group in frame.groupby("_slice", sort=True):
        y = group["actual"].astype(int)
        p = group["probability"].astype(float)
        b = group["baseline_probability"].astype(float)
        model_brier = float(brier_score_loss(y, p))
        base_brier = float(brier_score_loss(y, b))
        output.append(
            SliceMetric(
                label=str(label),
                rows=int(len(group)),
                events=int(y.sum()),
                brier_skill=(
                    float(1.0 - model_brier / base_brier)
                    if base_brier > 0
                    else None
                ),
                calibration_gap=float(p.mean() - y.mean()),
            )
        )
    return tuple(output)
