"""Utilities for constructing PREACT's feature store."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import pandas as pd

from ..config import FeatureConfig


@dataclass
class FeatureStore:
    """Container for feature tables keyed by entity identifiers."""

    tables: Dict[str, pd.DataFrame]

    def latest(self) -> Dict[str, pd.Series]:
        """Return the latest observation for each feature table."""

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


def combine_features(feature_groups: Mapping[str, Mapping[str, pd.DataFrame]]) -> FeatureStore:
    """Combine feature groups into a unified, aligned feature store."""

    tables: Dict[str, pd.DataFrame] = {}
    for domain, features in feature_groups.items():
        for key, df in features.items():
            tables[f"{domain}__{key}"] = df

    aligned: Dict[str, pd.DataFrame] = {}
    for name, table in tables.items():
        aligned[name] = table.sort_index().ffill().fillna(0)
    return FeatureStore(tables=aligned)


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        return df
    if "event_date" in df.columns:
        df["date"] = pd.to_datetime(df["event_date"])
        return df
    raise ValueError("Expected a date or event_date column for feature aggregation")


def _pivot_country_series(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if "country" not in df.columns:
        df = df.copy()
        df["country"] = "GLOBAL"
    pivot = df.pivot_table(index="date", columns="country", values=value_column, aggfunc="sum")
    return pivot.sort_index()


def aggregate_humanitarian(data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Aggregate humanitarian indicators into rolling windows."""

    df = _ensure_date_column(data.copy())
    value_col = None
    for candidate in (
        "displaced_population",
        "access_constraint_score",
        "value",
        "count",
        "indicator",
    ):
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        raise ValueError("Humanitarian source lacks a numeric value column")
    pivot = _pivot_country_series(df, value_col)
    return pivot.rolling(window=config.window_days, min_periods=1).mean()


def build_feature_store(
    ingestion_results: Mapping[str, pd.DataFrame],
    feature_configs: Iterable[FeatureConfig],
) -> FeatureStore:
    """Build the feature store from raw ingestion outputs."""

    grouped_features: Dict[str, Dict[str, pd.DataFrame]] = {}
    for cfg in feature_configs:
        domain, _, _ = cfg.name.partition("_")
        grouped_features.setdefault(domain, {})
        if domain == "events":
            df = aggregate_events(ingestion_results[cfg.inputs[0]], cfg)
            grouped_features[domain][cfg.name] = df
        elif domain == "economic":
            base = ingestion_results[cfg.inputs[0]].copy()
            if base.empty:
                empty_index = pd.DatetimeIndex([], name="date")
                grouped_features[domain][cfg.name] = pd.DataFrame(index=empty_index)
                continue
            base["date"] = pd.to_datetime(base["date"])
            value_cols = [col for col in base.columns if col != "date"]
            if not value_cols:
                raise ValueError("Economic source lacks numeric columns")
            for column in value_cols:
                base[column] = pd.to_numeric(base[column], errors="coerce")
            base = base.groupby("date", as_index=True)[value_cols].mean().sort_index()
            full_index = pd.date_range(start=base.index.min(), end=base.index.max(), freq="D")
            base = base.reindex(full_index).ffill().dropna(how="all")
            base.index.name = "date"
            base = base.rolling(window=cfg.window_days, min_periods=1).mean()
            grouped_features[domain][cfg.name] = base
        elif domain == "humanitarian":
            df = aggregate_humanitarian(ingestion_results[cfg.inputs[0]], cfg)
            grouped_features[domain][cfg.name] = df
        else:
            raise ValueError(f"Unsupported feature type: {cfg.name}")
    return combine_features(feature_groups=grouped_features)


__all__ = [
    "FeatureStore",
    "aggregate_events",
    "aggregate_humanitarian",
    "build_feature_store",
    "combine_features",
]
