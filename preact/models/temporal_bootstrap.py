"""Temporal resampling utilities for dependent out-of-sample predictions.

The predictive panel is serially dependent across forecast dates.  IID date
bootstrap therefore produces confidence intervals that can be materially too
narrow.  This module provides a moving-block bootstrap over *ordered unique
forecast dates* while keeping all entities observed on a sampled date together.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd


def default_block_length(n_dates: int) -> int:
    """Conservative data-driven default when no dependence scale is supplied.

    ``sqrt(n)`` is deliberately simple and auditable.  It grows with the OOS
    history rather than silently reverting to IID resampling.
    """
    if n_dates < 1:
        raise ValueError("n_dates must be positive")
    return max(2, min(n_dates, int(math.ceil(math.sqrt(n_dates)))))


def moving_block_date_samples(
    dates: pd.Series | pd.Index | np.ndarray,
    *,
    samples: int,
    seed: int = 42,
    block_length: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield moving-block bootstrap samples of chronological unique dates.

    Blocks are contiguous and sampled with replacement.  They never wrap from
    the end of the observed history back to its beginning, avoiding a false
    adjacency at the sample boundary.  The final block is truncated so every
    replicate contains exactly ``n_dates`` date positions.
    """
    if samples < 1:
        return
    ordered = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates).unique())).sort_values()
    n_dates = len(ordered)
    if n_dates < 2:
        return
    length = default_block_length(n_dates) if block_length is None else int(block_length)
    if length < 2:
        raise ValueError("block_length must be >= 2 to preserve temporal dependence")
    if length > n_dates:
        raise ValueError("block_length cannot exceed the number of unique dates")

    rng = np.random.default_rng(seed)
    starts = np.arange(0, n_dates - length + 1)
    blocks_per_sample = int(math.ceil(n_dates / length))
    values = ordered.to_numpy()
    for _ in range(int(samples)):
        chosen = rng.choice(starts, size=blocks_per_sample, replace=True)
        replicate = np.concatenate([values[start : start + length] for start in chosen])
        yield replicate[:n_dates]
