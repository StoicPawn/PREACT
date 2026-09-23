import numpy as np
import pandas as pd

from preact.models.relational_hazard import PREACTRelationalHazardMixture


def test_relational_hazard_uses_nested_temporal_gate_and_returns_probabilities():
    dates = pd.date_range("2000-01-01", periods=36, freq="90D")
    entities = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product(
        [dates, entities],
        names=["date", "entity_id"],
    )
    rng = np.random.default_rng(11)
    local = rng.normal(size=len(index))
    world = rng.normal(size=len(index))
    trend = np.repeat(np.linspace(-1.0, 1.0, len(dates)), len(entities))
    logits = -1.5 + 0.7 * local + 0.9 * world + 0.6 * trend
    probability = 1.0 / (1.0 + np.exp(-logits))
    y = pd.Series(rng.binomial(1, probability), index=index)

    X = pd.DataFrame(
        {
            "local:economy": local,
            "local:trend": trend,
            "world_context:neighbor_recent_365d:total": world,
            "world_context:system_decay_365d": np.abs(world) + 0.5,
        },
        index=index,
    )

    model = PREACTRelationalHazardMixture(
        horizon_days=60,
        gate_validation_dates=4,
        weight_grid_size=11,
        random_state=7,
    )
    model.fit(X, y)

    predicted = model.predict_proba(X.iloc[-20:])

    assert predicted.shape == (20, 2)
    assert np.allclose(predicted.sum(axis=1), 1.0)
    assert ((predicted[:, 1] > 0.0) & (predicted[:, 1] < 1.0)).all()
    assert 0.0 <= model.context_weight_ <= 1.0
    assert model.context_feature_count_ == 2
    assert model.gate_status_ in {
        "nested_temporal_validation",
        "fixed_insufficient_inner_outcomes",
        "fixed_no_valid_inner_split",
    }
    if model.gate_status_ == "nested_temporal_validation":
        assert model.inner_training_cutoff_ < model.internal_validation_start_
        assert (
            model.internal_validation_start_ - model.inner_training_cutoff_
            >= pd.Timedelta(days=60)
        )
