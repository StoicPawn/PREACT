from datetime import datetime, timezone

import numpy as np
import pandas as pd

from preact.feature_store.graph_targets import binary_relation_target
from preact.history.graph_store import HistoricalGraphStore
from preact.history.relations import HistoricalRelation
from preact.models.benchmark_suite import run_benchmark_suite

UTC = timezone.utc


def test_relation_target_marks_incomplete_future_horizon_as_censored(tmp_path):
    graph = HistoricalGraphStore(tmp_path / "graph.duckdb")
    graph.insert(
        [
            HistoricalRelation(
                relation_id="future-mid",
                relation_type="militarized_interstate_dispute",
                subject_entity_id="a",
                object_entity_id="b",
                valid_from=datetime(2021, 1, 10, tzinfo=UTC),
                known_at=datetime(2021, 1, 10, tzinfo=UTC),
                source="test",
                source_ref="future-mid",
                retrieved_at=datetime(2021, 1, 10, tzinfo=UTC),
            )
        ]
    )

    target = binary_relation_target(
        graph,
        entity_id="a",
        cutoffs=[datetime(2020, 12, 15, tzinfo=UTC)],
        relation_type="militarized_interstate_dispute",
        horizon_days=30,
        outcome_observed_through=datetime(2020, 12, 31, tzinfo=UTC),
    )

    # A future event exists in storage, but the observation window was not yet
    # complete at the declared data boundary. It must be unknown, not negative
    # and not positive, otherwise historical backtests gain look-ahead bias.
    assert target.isna().iloc[0]


def test_benchmark_excludes_censored_rows_before_temporal_folds():
    dates = pd.date_range("2000-01-01", periods=65, freq="30D")
    entities = [f"c{i}" for i in range(6)]
    index = pd.MultiIndex.from_product(
        [dates, entities], names=["date", "entity_id"]
    )
    rng = np.random.default_rng(19)
    trend = np.repeat(np.linspace(-1.0, 1.0, len(dates)), len(entities))
    features = pd.DataFrame(
        {"trend": trend, "noise": rng.normal(size=len(index))}, index=index
    )
    probability = 1.0 / (1.0 + np.exp(-(-2.0 + trend)))
    target = pd.Series(rng.binomial(1, probability), index=index, dtype="Int64")

    censored_dates = dates[-4:]
    target.loc[target.index.get_level_values("date").isin(censored_dates)] = pd.NA

    result = run_benchmark_suite(
        features,
        target,
        horizon_days=30,
        min_train_dates=20,
        calibration_dates=4,
        test_dates_per_fold=4,
        bootstrap_samples=20,
    )

    for model in result.models.values():
        if model.predictions.empty:
            continue
        predicted_dates = pd.to_datetime(model.predictions["date"])
        assert not predicted_dates.isin(censored_dates).any()
        assert model.predictions["actual"].notna().all()
