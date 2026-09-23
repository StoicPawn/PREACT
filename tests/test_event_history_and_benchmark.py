from datetime import datetime, timezone

import numpy as np
import pandas as pd

from preact.feature_store.event_history import event_history_features
from preact.history.graph_store import HistoricalGraphStore
from preact.history.relations import HistoricalRelation
from preact.models.benchmark_suite import run_benchmark_suite

UTC = timezone.utc


def _relation(rid, kind, start):
    return HistoricalRelation(
        relation_id=rid,
        relation_type=kind,
        subject_entity_id="a",
        object_entity_id="b",
        valid_from=start,
        known_at=start,
        source="test",
        source_ref=rid,
        retrieved_at=start,
    )


def test_event_history_features_measure_recency_and_network(tmp_path):
    graph = HistoricalGraphStore(tmp_path / "graph.duckdb")
    graph.insert(
        [
            _relation(
                "d1",
                "militarized_interstate_dispute",
                datetime(2020, 1, 1, tzinfo=UTC),
            ),
            _relation(
                "a1",
                "formal_alliance",
                datetime(2019, 1, 1, tzinfo=UTC),
            ),
        ]
    )
    values = event_history_features(
        graph,
        entity_id="a",
        cutoff=datetime(2020, 6, 1, tzinfo=UTC),
    )
    assert 100 < values[
        "history:days_since_last:militarized_interstate_dispute"
    ] < 200
    assert values["history:ever:formal_alliance"] == 1.0
    assert values["history:active_counterparties:formal_alliance"] == 1.0


def test_benchmark_suite_runs_models_on_identical_oos_rows():
    dates = pd.date_range("2000-01-01", periods=70, freq="30D")
    entities = [f"c{i}" for i in range(8)]
    index = pd.MultiIndex.from_product(
        [dates, entities], names=["date", "entity_id"]
    )
    t = np.repeat(np.linspace(-1.5, 1.5, len(dates)), len(entities))
    entity_effect = np.tile(np.linspace(-0.5, 0.5, len(entities)), len(dates))
    rng = np.random.default_rng(3)
    x = pd.DataFrame(
        {
            "trend": t,
            "entity_signal": entity_effect,
            "noise": rng.normal(size=len(index)),
        },
        index=index,
    )
    logits = -2.5 + 1.2 * t + 0.6 * entity_effect
    p = 1 / (1 + np.exp(-logits))
    y = pd.Series(rng.binomial(1, p), index=index)

    horizon_days = 60
    result = run_benchmark_suite(
        x,
        y,
        horizon_days=horizon_days,
        min_train_dates=25,
        calibration_dates=4,
        test_dates_per_fold=4,
        bootstrap_samples=50,
    )

    assert set(result.models) == {
        "logistic_l2",
        "cloglog_hazard",
        "hist_gradient_boosting",
        "extra_trees",
    }
    lengths = {len(model.predictions) for model in result.models.values()}
    assert len(lengths) == 1
    assert next(iter(lengths)) > 0
    assert all(model.metrics.rows > 0 for model in result.models.values())

    # Every candidate must be scored on exactly the same OOS observations.  A
    # length-only check can miss model-specific row substitutions or duplicate
    # forecasts and would make benchmark comparisons invalid.
    reference_keys = None
    for model in result.models.values():
        frame = model.predictions
        assert not frame.duplicated(["date", "entity_id"]).any()
        keys = set(zip(frame["date"], frame["entity_id"], strict=True))
        if reference_keys is None:
            reference_keys = keys
        else:
            assert keys == reference_keys

        # The latest label admitted to training/calibration must be separated
        # from every scored forecast by at least the target horizon.  Keep this
        # invariant at the benchmark boundary so future CV refactors cannot
        # silently reintroduce temporal target leakage.
        forecast_dates = pd.to_datetime(frame["date"])
        cutoffs = pd.to_datetime(frame["training_cutoff"])
        assert (cutoffs <= forecast_dates - pd.Timedelta(days=horizon_days)).all()
