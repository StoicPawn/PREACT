"""Leakage-resistant probabilistic baseline for historical replay.

This module is deliberately separate from PREACT's legacy training path. It is the
reference evaluation protocol for historical claims: chronological folds, target-
horizon purge, chronological calibration and untouched out-of-sample scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class ReplayBacktestMetrics:
    rows: int
    events: int
    brier: float | None
    baseline_brier: float | None
    brier_skill: float | None
    log_loss: float | None
    roc_auc: float | None
    average_precision: float | None
    mean_probability: float | None
    event_rate: float | None
    calibration_gap: float | None


@dataclass(frozen=True)
class ReplayBacktestResult:
    predictions: pd.DataFrame
    metrics: ReplayBacktestMetrics
    folds_used: int
    folds_skipped: int


def _as_datetime_index(frame: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("features must use a DatetimeIndex")
    x = frame.sort_index().copy()
    y = target.reindex(x.index)
    valid = y.notna()
    x = x.loc[valid]
    y = y.loc[valid].astype(int)
    return x, y


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def _smoothed_rate(target: pd.Series) -> float:
    # Jeffreys smoothing avoids exact 0/1 forecasts in rare-event folds.
    return float((target.sum() + 0.5) / (len(target) + 1.0))


def _constant_predictions(rate: float, n: int) -> np.ndarray:
    return np.full(n, float(np.clip(rate, 1e-6, 1 - 1e-6)))


def _fit_and_predict_fold(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    horizon_days: int,
    calibration_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, float] | None:
    if x_train.empty or x_test.empty:
        return None

    baseline_rate = _smoothed_rate(y_train)
    unique_dates = pd.Index(x_train.index.unique()).sort_values()
    if len(unique_dates) < 8:
        return _constant_predictions(baseline_rate, len(x_test)), baseline_rate

    split_pos = min(
        len(unique_dates) - 1,
        max(2, int(len(unique_dates) * (1.0 - calibration_fraction))),
    )
    calibration_start = pd.Timestamp(unique_dates[split_pos])
    purge_delta = pd.Timedelta(days=max(0, int(horizon_days)))

    base_mask = x_train.index < (calibration_start - purge_delta)
    calibration_mask = x_train.index >= calibration_start
    x_base = x_train.loc[base_mask]
    y_base = y_train.loc[base_mask]
    x_cal = x_train.loc[calibration_mask]
    y_cal = y_train.loc[calibration_mask]

    if len(x_base) < 20 or y_base.nunique() < 2:
        return _constant_predictions(baseline_rate, len(x_test)), baseline_rate

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        random_state=random_state,
    )
    model.fit(x_base, y_base)

    test_raw = model.predict_proba(x_test)[:, 1]
    if len(x_cal) < 10 or y_cal.nunique() < 2:
        return np.clip(test_raw, 1e-6, 1 - 1e-6), baseline_rate

    cal_raw = model.predict_proba(x_cal)[:, 1]
    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    calibrator.fit(_logit(cal_raw), y_cal)
    calibrated = calibrator.predict_proba(_logit(test_raw))[:, 1]
    return np.clip(calibrated, 1e-6, 1 - 1e-6), baseline_rate


def _metrics(predictions: pd.DataFrame) -> ReplayBacktestMetrics:
    if predictions.empty:
        return ReplayBacktestMetrics(
            rows=0,
            events=0,
            brier=None,
            baseline_brier=None,
            brier_skill=None,
            log_loss=None,
            roc_auc=None,
            average_precision=None,
            mean_probability=None,
            event_rate=None,
            calibration_gap=None,
        )

    y = predictions["actual"].astype(int)
    p = predictions["probability"].astype(float)
    baseline = predictions["baseline_probability"].astype(float)
    brier = float(brier_score_loss(y, p))
    baseline_brier = float(brier_score_loss(y, baseline))
    brier_skill = (
        float(1.0 - brier / baseline_brier)
        if baseline_brier > 0
        else None
    )
    ll = float(log_loss(y, p, labels=[0, 1]))
    roc = float(roc_auc_score(y, p)) if y.nunique() > 1 else None
    ap = float(average_precision_score(y, p)) if y.nunique() > 1 else None
    mean_probability = float(p.mean())
    event_rate = float(y.mean())
    return ReplayBacktestMetrics(
        rows=int(len(predictions)),
        events=int(y.sum()),
        brier=brier,
        baseline_brier=baseline_brier,
        brier_skill=brier_skill,
        log_loss=ll,
        roc_auc=roc,
        average_precision=ap,
        mean_probability=mean_probability,
        event_rate=event_rate,
        calibration_gap=float(mean_probability - event_rate),
    )


def purged_walk_forward_backtest(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    horizon_days: int,
    n_splits: int = 5,
    calibration_fraction: float = 0.20,
    random_state: int = 42,
) -> ReplayBacktestResult:
    """Evaluate binary risk forecasts without target-horizon overlap.

    Training observations are removed when their label horizon reaches into the
    test period. The tail of each remaining training fold is reserved for
    chronological probability calibration and is never used to fit the base model.
    """

    if horizon_days < 0:
        raise ValueError("horizon_days must be non-negative")
    if not 0.05 <= calibration_fraction <= 0.40:
        raise ValueError("calibration_fraction must be between 0.05 and 0.40")

    x, y = _as_datetime_index(features, target)
    if len(x) < max(30, n_splits + 5):
        return ReplayBacktestResult(
            predictions=pd.DataFrame(),
            metrics=_metrics(pd.DataFrame()),
            folds_used=0,
            folds_skipped=0,
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict[str, Any]] = []
    folds_used = 0
    folds_skipped = 0
    purge_delta = pd.Timedelta(days=int(horizon_days))

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x), start=1):
        x_candidate = x.iloc[train_idx]
        y_candidate = y.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_test = y.iloc[test_idx]
        if x_test.empty:
            folds_skipped += 1
            continue

        test_start = pd.Timestamp(x_test.index.min())
        keep = x_candidate.index < (test_start - purge_delta)
        x_train = x_candidate.loc[keep]
        y_train = y_candidate.loc[keep]
        if len(x_train) < 20:
            folds_skipped += 1
            continue

        result = _fit_and_predict_fold(
            x_train,
            y_train,
            x_test,
            horizon_days=horizon_days,
            calibration_fraction=calibration_fraction,
            random_state=random_state + fold,
        )
        if result is None:
            folds_skipped += 1
            continue

        probabilities, baseline_rate = result
        for timestamp, actual, probability in zip(
            x_test.index,
            y_test.to_numpy(),
            probabilities,
        ):
            rows.append(
                {
                    "date": pd.Timestamp(timestamp),
                    "fold": fold,
                    "actual": int(actual),
                    "probability": float(probability),
                    "baseline_probability": float(baseline_rate),
                    "test_start": test_start,
                    "training_cutoff": test_start - purge_delta,
                }
            )
        folds_used += 1

    predictions = pd.DataFrame(rows)
    if not predictions.empty:
        predictions = predictions.set_index("date").sort_index()
    return ReplayBacktestResult(
        predictions=predictions,
        metrics=_metrics(predictions),
        folds_used=folds_used,
        folds_skipped=folds_skipped,
    )
