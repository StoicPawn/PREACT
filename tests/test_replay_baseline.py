import numpy as np
import pandas as pd

from preact.models.replay_baseline import purged_walk_forward_backtest


def test_purged_backtest_keeps_training_horizon_before_test() -> None:
    index = pd.date_range("2020-01-01", periods=180, freq="D")
    x = pd.DataFrame(
        {
            "trend": np.linspace(-1, 1, len(index)),
            "cycle": np.sin(np.arange(len(index)) / 10.0),
        },
        index=index,
    )
    y = pd.Series(
        ((x["trend"] + 0.5 * x["cycle"]) > 0.35).astype(int),
        index=index,
    )

    result = purged_walk_forward_backtest(
        x,
        y,
        horizon_days=14,
        n_splits=4,
    )

    assert result.folds_used > 0
    assert not result.predictions.empty
    assert (
        result.predictions["training_cutoff"]
        <= result.predictions["test_start"] - pd.Timedelta(days=14)
    ).all()
    assert 0.0 <= result.metrics.brier <= 1.0
    assert result.metrics.baseline_brier is not None


def test_purged_backtest_handles_rare_constant_training_periods() -> None:
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    x = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)
    y = pd.Series(0, index=index)
    y.iloc[-10:] = 1

    result = purged_walk_forward_backtest(
        x,
        y,
        horizon_days=7,
        n_splits=3,
    )

    assert result.folds_used > 0
    assert result.predictions["probability"].between(0, 1).all()
    assert result.metrics.rows == len(result.predictions)
