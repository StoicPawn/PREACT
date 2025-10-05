from __future__ import annotations

import pandas as pd

from preact.config import ModelConfig
from preact.models.neural import NeuralNetworkEngine
from preact.models.predictor import PredictiveEngine


def _sample_features() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2023-01-01", periods=8, freq="D")
    return {
        "events": pd.DataFrame(
            {
                "count": [5, 6, 8, 7, 9, 11, 12, 13],
                "severity": [0.1, 0.12, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4],
            },
            index=index,
        ),
        "economic": pd.DataFrame(
            {
                "gdp": [1.0, 1.05, 1.07, 1.1, 1.15, 1.17, 1.2, 1.25],
                "inflation": [0.04, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.09],
            },
            index=index,
        ),
    }


def _sample_target() -> pd.Series:
    return pd.Series(
        [0, 0, 0, 0, 1, 1, 1, 1],
        index=pd.date_range("2023-01-01", periods=8, freq="D"),
        name="event",
    )


def test_predictive_engine_persistence(tmp_path):
    config = ModelConfig(
        name="thirty_day",
        target="event",
        horizon_days=30,
        features=["events", "economic"],
        hyperparameters={"n_estimators": 10, "max_depth": 2},
    )
    engine = PredictiveEngine([config])
    features = _sample_features()
    target = _sample_target()

    models = engine.train(features, target)
    engine.save_models(models, tmp_path)

    assert engine.has_persisted_models(tmp_path)

    loaded = engine.load_models(tmp_path)
    original = engine.predict(models, features, target)[0].probabilities
    restored = engine.predict(loaded, features, target)[0].probabilities

    pd.testing.assert_series_equal(original, restored)


def test_neural_engine_persistence(tmp_path):
    config = ModelConfig(
        name="atrocity_nn",
        target="event",
        horizon_days=60,
        features=["events", "economic"],
        hyperparameters={"hidden_layer_sizes": (4,), "max_iter": 50},
    )
    engine = NeuralNetworkEngine([config], random_state=5)
    features = _sample_features()
    target = _sample_target()

    models = engine.train(features, target)
    engine.save_models(models, tmp_path)

    assert engine.has_persisted_models(tmp_path)

    restored = engine.load_models(tmp_path)
    assert config.name in engine.training_history

    original = engine.predict(models, features, target)[0].probabilities
    replay = engine.predict(restored, features, target)[0].probabilities

    pd.testing.assert_series_equal(original, replay)
