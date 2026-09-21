"""Reusable purged chronological folds for geopolitical panel research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    fold: int
    fit_dates: tuple[pd.Timestamp, ...]
    calibration_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]
    fit_cutoff: pd.Timestamp
    calibration_start: pd.Timestamp
    training_cutoff: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def purged_panel_folds(
    index: pd.MultiIndex,
    *,
    horizon_days: int,
    min_train_dates: int = 20,
    calibration_dates: int = 5,
    test_dates_per_fold: int = 5,
) -> list[TemporalFold]:
    """Create expanding-window folds with target-horizon purges.

    Splits are performed on unique calendar dates, never individual rows, so all
    countries/polities for a date remain in the same fold.  The target horizon
    is purged not only before each test window but also between the fit and
    calibration windows.  The resulting fold records both embargo boundaries,
    making the full fit -> calibration -> test separation auditable downstream
    instead of forcing experiment artifacts to reconstruct it from row dates.
    This matters whenever calibration predictions are used for model selection,
    weighting or probability calibration: otherwise the latest fit labels can
    contain outcomes occurring inside the calibration period.
    """

    if not isinstance(index, pd.MultiIndex) or "date" not in index.names:
        raise TypeError("index must be a MultiIndex containing 'date'")
    # PREACT evaluates future-event risk. A zero-day horizon is not a valid
    # forecasting problem and, more importantly, collapses the temporal embargo
    # that protects fit/calibration/test labels from sharing outcome time.
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if min_train_dates < 5:
        raise ValueError("min_train_dates must be >= 5")
    if calibration_dates < 1:
        raise ValueError("calibration_dates must be >= 1")
    if test_dates_per_fold < 1:
        raise ValueError("test_dates_per_fold must be >= 1")

    raw_dates = index.get_level_values("date")
    if raw_dates.isna().any():
        raise ValueError("date level must not contain missing timestamps")
    try:
        normalized_dates = pd.to_datetime(raw_dates, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("date level must contain valid timestamps") from exc
    if normalized_dates.isna().any():
        raise ValueError("date level must not contain missing timestamps")

    dates = pd.Index(normalized_dates.unique()).sort_values()
    purge = pd.Timedelta(days=int(horizon_days))
    folds: list[TemporalFold] = []
    cursor = min_train_dates
    fold_id = 0

    while cursor < len(dates):
        test_dates = dates[cursor : cursor + test_dates_per_fold]
        if len(test_dates) == 0:
            break
        test_start = pd.Timestamp(test_dates[0])
        test_end = pd.Timestamp(test_dates[-1])
        training_cutoff = test_start - purge
        eligible = dates[dates < training_cutoff]

        if len(eligible) >= min_train_dates:
            n_cal = min(calibration_dates, max(1, len(eligible) // 4))
            cal = eligible[-n_cal:]
            calibration_start = pd.Timestamp(cal[0])
            fit_cutoff = calibration_start - purge
            fit = eligible[eligible < fit_cutoff]
            if len(fit) >= max(5, min_train_dates - n_cal):
                folds.append(
                    TemporalFold(
                        fold=fold_id,
                        fit_dates=tuple(pd.Timestamp(x) for x in fit),
                        calibration_dates=tuple(pd.Timestamp(x) for x in cal),
                        test_dates=tuple(pd.Timestamp(x) for x in test_dates),
                        fit_cutoff=pd.Timestamp(fit_cutoff),
                        calibration_start=calibration_start,
                        training_cutoff=pd.Timestamp(training_cutoff),
                        test_start=test_start,
                        test_end=test_end,
                    )
                )
                fold_id += 1

        cursor += test_dates_per_fold
    return folds
