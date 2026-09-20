import numpy as np
import pandas as pd

from preact.models.stress_tests import (
    deterministic_entity_holdout,
    run_unseen_entity_stress,
)


def test_entity_holdout_is_stable_and_disjoint():
    entities = [f"c{i}" for i in range(20)]
    a = deterministic_entity_holdout(entities, fraction=0.2, salt="x")
    b = deterministic_entity_holdout(entities, fraction=0.2, salt="x")
    assert a == b
    assert set(a.train_entities).isdisjoint(a.test_entities)
    assert set(a.train_entities) | set(a.test_entities) == set(entities)


def test_unseen_entity_stress_never_trains_on_holdout_entities():
    dates = pd.date_range("2000-01-01", periods=55, freq="30D")
    entities = [f"c{i}" for i in range(12)]
    idx = pd.MultiIndex.from_product([dates, entities], names=["date", "entity_id"])
    rng = np.random.default_rng(8)
    x = pd.DataFrame(
        {
            "x": rng.normal(size=len(idx)),
            "trend": np.repeat(np.linspace(-1, 1, len(dates)), len(entities)),
        },
        index=idx,
    )
    p = 1 / (1 + np.exp(-(-2 + 1.2 * x["trend"] + 0.4 * x["x"])))
    y = pd.Series(rng.binomial(1, p), index=idx)

    split, result = run_unseen_entity_stress(
        x,
        y,
        horizon_days=30,
        holdout_fraction=0.25,
        min_train_dates=20,
        calibration_dates=4,
        test_dates_per_fold=4,
        bootstrap_samples=20,
    )

    assert result.models
    for model in result.models.values():
        if not model.predictions.empty:
            assert set(model.predictions["entity_id"]).issubset(
                set(split.test_entities)
            )
