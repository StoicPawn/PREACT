import numpy as np
import pandas as pd

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
    for fold in folds:
        assert max(fold.fit_dates + fold.calibration_dates) < fold.training_cutoff
        assert fold.training_cutoff <= fold.test_start - pd.Timedelta(days=90)


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
