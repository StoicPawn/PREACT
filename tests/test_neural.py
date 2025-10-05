import pandas as pd

from preact.config import ModelConfig
from preact.models.neural import NeuralNetworkEngine


def test_neural_engine_trains_and_predicts():
    features = {
        "events": pd.DataFrame(
            {
                "count": [5, 6, 7, 9, 11, 13],
                "severity": [0.2, 0.25, 0.3, 0.4, 0.45, 0.5],
            },
            index=pd.date_range("2023-01-01", periods=6, freq="D"),
        ),
        "economic": pd.DataFrame(
            {
                "gdp": [1.0, 1.1, 1.2, 1.25, 1.3, 1.35],
                "inflation": [0.05, 0.05, 0.06, 0.07, 0.08, 0.09],
            },
            index=pd.date_range("2023-01-01", periods=6, freq="D"),
        ),
    }
    target = pd.Series([0, 0, 0, 1, 1, 1], index=pd.date_range("2023-01-01", periods=6, freq="D"))

    config = ModelConfig(
        name="atrocity_nn",
        target="atrocity",
        horizon_days=30,
        features=["events", "economic"],
        hyperparameters={"hidden_layer_sizes": (8,), "max_iter": 50},
    )
    engine = NeuralNetworkEngine([config], random_state=7)

    models = engine.train(features, target)
    assert config.name in models
    assert config.name in engine.training_history
    history = engine.training_history[config.name]
    assert history.loss_curve, "Loss curve should capture training progression"

    outputs = engine.predict(models, features, target)
    assert len(outputs) == 1
    output = outputs[0]
    assert output.name == config.name
    assert all(0 <= prob <= 1 for prob in output.probabilities)
    assert set(["brier", "precision", "recall"]).issubset(output.diagnostics)

