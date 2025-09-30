from datetime import datetime

import pandas as pd

from preact.config import FeatureConfig
from preact.feature_store.builder import (
    aggregate_events,
    aggregate_humanitarian,
    build_feature_store,
)


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
        "Economic_Indicators": pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "inflation_rate": [5.0],
                "gdp_growth": [3.2],
            }
        ),
        "UNHCR": pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02"],
                "country": ["A", "A"],
                "displaced_population": [100, 150],
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
            inputs=["Economic_Indicators"],
            aggregation="mean",
            window_days=14,
        ),
        FeatureConfig(
            name="humanitarian_displacement",
            inputs=["UNHCR"],
            aggregation="mean",
            window_days=7,
        ),
    ]
    store = build_feature_store(ingestion, features)
    assert "events__events_protests" in store.tables
    assert "economic__economic_pressure" in store.tables
    assert "humanitarian__humanitarian_displacement" in store.tables


def test_aggregate_humanitarian_timeseries():
    df = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "country": ["A", "B"],
            "displaced_population": [100, 150],
        }
    )
    config = FeatureConfig(
        name="humanitarian_displacement",
        inputs=["UNHCR"],
        aggregation="mean",
        window_days=3,
    )
    result = aggregate_humanitarian(df, config)
    assert result.loc["2023-01-01", "A"] == 100
    assert result.loc["2023-01-02", "B"] == 150

