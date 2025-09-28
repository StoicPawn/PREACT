import pandas as pd

from preact.models.bayesian import BayesianExplainer


def test_bayesian_explainer_returns_evidence():
    features = pd.DataFrame(
        {
            "events__feature": [0.1, 0.2, 0.4, 0.5],
            "economic__feature": [1.0, 1.1, 1.2, 1.3],
        },
        index=pd.date_range("2023-01-01", periods=4, freq="D"),
    )
    target = pd.Series([0, 0, 1, 1], index=features.index)

    explainer = BayesianExplainer(prior=0.2, smoothing=1.0, bins=3)
    evidence = explainer.explain(features, target)

    assert not evidence.empty
    assert set(["feature", "posterior", "likelihood_ratio"]).issubset(evidence.columns)
    assert evidence.iloc[0]["posterior"] >= 0
