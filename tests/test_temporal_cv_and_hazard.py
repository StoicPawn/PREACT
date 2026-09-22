import numpy as np
import pandas as pd
import pytest

from preact.models.hazard import ComplementaryLogLogHazard
from preact.models.temporal_cv import purged_panel_folds


def test_purged_panel_folds_keep_horizon_between_fit_and_test():
    dates = pd.date_range("2000-01-01", periods=60, freq="30D")
    index = pd.MultiIndex.from_product(
        [dates, ["a", "b", "c"]], names=["date", "entity_id"]
    )
    folds = purged_panel_folds(
        index,
        horizon_days=90,
        min_train_dates=20,
        calibration_dates=4,
        test_dates_per_fold=3,
    )
    assert folds
    horizon = pd.Timedelta(days=90)
    for fold in folds:
        assert max(fold.calibration_dates) < fold.training_cutoff
        assert fold.training_cutoff == fold.test_start - horizon
        assert fold.calibration_start == min(fold.calibration_dates)
        assert fold.fit_cutoff == fold.calibration_start - horizon
        assert max(fold.fit_dates) < fold.fit_cutoff
        assert max(fold.fit_dates) < min(fold.calibration_dates) - horizon


def test_temporal_fold_audit_record_persists_both_embargoes_and_windows():
    dates = pd.date_range("2000-01-01", periods=60, freq="30D")
    index = pd.MultiIndex.from_product([dates, ["a", "b"]], names=["date", "entity_id"])
    fold = purged_panel_folds(
        index, horizon_days=90, min_train_dates=20, calibration_dates=4, test_dates_per_fold=3
    )[0]
    record = fold.audit_record()
    assert record["fit_cutoff"] == fold.fit_cutoff.isoformat()
    assert record["training_cutoff"] == fold.training_cutoff.isoformat()
    assert record["fit_end"] == max(fold.fit_dates).isoformat()
    assert record["calibration_start"] == min(fold.calibration_dates).isoformat()
    assert record["calibration_end"] == max(fold.calibration_dates).isoformat()
    assert record["test_start"] == min(fold.test_dates).isoformat()
    assert record["test_end"] == max(fold.test_dates).isoformat()
    assert record["fit_dates"] == len(fold.fit_dates)
    assert record["calibration_dates"] == len(fold.calibration_dates)
    assert record["test_dates"] == len(fold.test_dates)


def test_purged_panel_folds_calibration_embargo_handles_irregular_dates():
    regular = pd.date_range("2000-01-01", periods=75, freq="30D")
    dates = regular.delete([17, 18, 33, 51])
    index = pd.MultiIndex.from_product(
        [dates, ["a", "b"]], names=["date", "entity_id"]
    )
    folds = purged_panel_folds(
        index,
        horizon_days=75,
        min_train_dates=20,
        calibration_dates=5,
        test_dates_per_fold=4,
    )
    assert folds
    horizon = pd.Timedelta(days=75)
    for fold in folds:
        assert fold.fit_cutoff == fold.calibration_start - horizon
        assert fold.training_cutoff == fold.test_start - horizon
        assert max(fold.fit_dates) + horizon < min(fold.calibration_dates)
        assert max(fold.calibration_dates) + horizon < fold.test_start


def test_purged_panel_folds_reject_missing_dates():
    index = pd.MultiIndex.from_arrays(
        [[pd.Timestamp("2000-01-01"), pd.NaT, pd.Timestamp("2000-03-01")], ["a", "a", "a"]],
        names=["date", "entity_id"],
    )
    with pytest.raises(ValueError, match="missing timestamps"):
        purged_panel_folds(index, horizon_days=30)


def test_purged_panel_folds_reject_zero_horizon():
    dates = pd.date_range("2000-01-01", periods=30, freq="30D")
    index = pd.MultiIndex.from_product([dates, ["a", "b"]], names=["date", "entity_id"])
    with pytest.raises(ValueError, match="horizon_days must be positive"):
        purged_panel_folds(index, horizon_days=0)


def test_cloglog_hazard_learns_monotone_signal():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(500, 1))
    true_p = 1.0 - np.exp(-np.exp(-2.0 + 1.5 * x[:, 0]))
    y = rng.binomial(1, true_p)
    model = ComplementaryLogLogHazard(l2=0.1).fit(x, y)
    low = model.predict_proba([[-2.0]])[0, 1]
    high = model.predict_proba([[2.0]])[0, 1]
    assert 0 < low < high < 1
    assert model.coef_[0, 0] > 0
