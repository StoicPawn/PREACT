import numpy as np
import pandas as pd
import pytest

from preact.models.temporal_bootstrap import (
    default_block_length,
    moving_block_date_samples,
)


def test_default_block_length_never_degenerates_to_iid_when_possible():
    assert default_block_length(2) == 2
    assert default_block_length(25) == 5
    assert default_block_length(100) == 10


def test_moving_blocks_keep_contiguous_dates_and_sample_size():
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    replicates = list(
        moving_block_date_samples(dates, samples=8, seed=7, block_length=3)
    )
    assert len(replicates) == 8
    for replicate in replicates:
        assert len(replicate) == len(dates)
        # Compare timedeltas rather than raw ``asi8`` integers: pandas may use
        # microsecond or nanosecond datetime storage depending on its version.
        for start in range(0, 9, 3):
            block = pd.DatetimeIndex(replicate[start : start + 3])
            assert np.all(np.diff(block.to_numpy()) == np.timedelta64(1, "D"))


def test_moving_blocks_sort_irregular_dates_before_resampling():
    dates = pd.to_datetime(["2020-03-01", "2020-01-01", "2020-01-10", "2020-02-01"])
    replicate = next(
        moving_block_date_samples(dates, samples=1, seed=2, block_length=2)
    )
    observed = pd.DatetimeIndex(sorted(dates))
    valid_pairs = {(observed[i], observed[i + 1]) for i in range(len(observed) - 1)}
    assert (pd.Timestamp(replicate[0]), pd.Timestamp(replicate[1])) in valid_pairs
    assert (pd.Timestamp(replicate[2]), pd.Timestamp(replicate[3])) in valid_pairs


def test_block_length_rejects_iid_and_impossible_blocks():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    with pytest.raises(ValueError, match="preserve temporal dependence"):
        list(moving_block_date_samples(dates, samples=1, block_length=1))
    with pytest.raises(ValueError, match="cannot exceed"):
        list(moving_block_date_samples(dates, samples=1, block_length=5))
