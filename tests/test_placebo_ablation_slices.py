import numpy as np
import pandas as pd

from preact.models.ablation import columns_in_family
from preact.models.placebo import within_date_permutation
from preact.models.sliced_evaluation import temporal_slice_metrics


def test_placebo_preserves_event_count_within_each_date():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=4), ["a", "b", "c", "d"]],
        names=["date", "entity_id"],
    )
    y = pd.Series([1,0,0,0, 0,1,1,0, 0,0,0,0, 1,1,0,0], index=idx)
    p = within_date_permutation(y, seed=2)
    before = y.groupby(level="date").sum()
    after = p.groupby(level="date").sum()
    assert before.equals(after)
    assert int(p.sum()) == int(y.sum())


def test_ablation_family_detection():
    columns = ["cow_nmc:cinc", "history:count_365d:x", "world_bank:gdp", "other"]
    assert columns_in_family(columns, ("history:", "graph:")) == (
        "history:count_365d:x",
    )


def test_temporal_slice_metrics_detects_periods():
    df = pd.DataFrame({
        "date":[pd.Timestamp("1995-01-01"),pd.Timestamp("2005-01-01")],
        "actual":[0,1],
        "probability":[0.1,0.8],
        "baseline_probability":[0.3,0.3],
    })
    slices = temporal_slice_metrics(df, years_per_slice=10)
    assert [x.label for x in slices] == ["1990-1999","2000-2009"]
