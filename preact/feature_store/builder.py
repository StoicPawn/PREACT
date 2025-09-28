"""Feature engineering pipeline for PREACT."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Mapping

import pandas as pd

from ..config import FeatureConfig


@dataclass
class FeatureStore:
    """Container for feature tables keyed by entity identifiers."""

    tables: Dict[str, pd.DataFrame]

    def latest(self) -> Dict[str, pd.Series]:
        return {name: df.iloc[-1] for name, df in self.tables.items() if not df.empty}


def aggregate_events(events: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Aggregate event-based datasets into daily time series."""

    if "event_date" in events.columns:
        events["event_date"] = pd.to_datetime(events["event_date"])
    elif "date" in events.columns:
        events["event_date"] = pd.to_datetime(events["date"])
    else:
        raise ValueError("Expected an event_date or date column")

    events["country"] = events.get("country", "GLOBAL")
    grouped = (
        events.groupby(["country", pd.Grouper(key="event_date", freq="D")])
        .size()
        .reset_index(name=config.name)
    )
    pivot = grouped.pivot_table(
        index="event_date", columns="country", values=config.name, fill_value=0
    )
    return pivot.rolling(window=config.window_days, min_periods=1).sum()


def combine_features(
    event_features: Mapping[str, pd.DataFrame],
    economic_features: Mapping[str, pd.DataFrame],
) -> FeatureStore:
    """Combine separate feature sources into a unified feature store."""

    tables: Dict[str, pd.DataFrame] = {}
    for key, df in event_features.items():
        tables[f"events__{key}"] = df
    for key, df in economic_features.items():
        tables[f"economic__{key}"] = df

    aligned: Dict[str, pd.DataFrame] = {}
    for name, table in tables.items():
        aligned[name] = table.sort_index().fillna(method="ffill").fillna(0)
    return FeatureStore(tables=aligned)


def build_feature_store(
    ingestion_results: Mapping[str, pd.DataFrame],
    feature_configs: Iterable[FeatureConfig],
) -> FeatureStore:
    """Build the feature store from raw ingestion outputs."""

    event_features: Dict[str, pd.DataFrame] = {}
    economic_features: Dict[str, pd.DataFrame] = {}
    for cfg in feature_configs:
        if cfg.name.startswith("events_"):
            df = aggregate_events(ingestion_results[cfg.inputs[0]], cfg)
            event_features[cfg.name] = df
        elif cfg.name.startswith("economic_"):
            df = ingestion_results[cfg.inputs[0]].set_index("date").rolling(
                window=cfg.window_days, min_periods=1
            ).mean()
            economic_features[cfg.name] = df
        else:
            raise ValueError(f"Unsupported feature type: {cfg.name}")
    return combine_features(event_features=event_features, economic_features=economic_features)

