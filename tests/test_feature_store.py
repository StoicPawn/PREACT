from datetime import datetime

import pandas as pd

from preact.config import FeatureConfig
from preact.feature_store.builder import aggregate_events, build_feature_store


def test_aggregate_events_counts_events():
    df = pd.DataFrame(
        {
            "event_date": ["2023-01-01", "2023-01-01", "2023-01-02"],
            "country": ["A", "A", "B"],
        }
    )
    config = FeatureConfig(
        name="events_test",
        inputs=["dummy"],
        aggregation="count",
        window_days=3,
    )
    result = aggregate_events(df, config)
    assert result.loc["2023-01-01", "A"] == 2
    assert result.loc["2023-01-02", "B"] == 1


def test_build_feature_store_combines_tables():
    ingestion = {
        "GDELT": pd.DataFrame(
            {
                "event_date": ["2023-01-01"],
                "country": ["A"],
            }
        ),
        "Synthetic_Economic": pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "food_price_index": [100],
            }
        ),
    }
    features = [
        FeatureConfig(
            name="events_protests",
            inputs=["GDELT"],
            aggregation="count",
            window_days=7,
        ),
        FeatureConfig(
            name="economic_pressure",
            inputs=["Synthetic_Economic"],
            aggregation="mean",
            window_days=14,
        ),
    ]
    store = build_feature_store(ingestion, features)
    assert "events__events_protests" in store.tables
    assert "economic__economic_pressure" in store.tables

