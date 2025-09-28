"""Data ingestion interfaces for the PREACT system."""
from __future__ import annotations

import abc
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import pandas as pd
import requests

from ..config import DataSourceConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Container for storing ingestion outputs."""

    data: pd.DataFrame
    metadata: Dict[str, str]

    def to_parquet(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_parquet(path)
        with path.with_suffix(".meta.json").open("w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, indent=2)


class DataSource(abc.ABC):
    """Abstract base class for all data sources."""

    config: DataSourceConfig

    def __init__(self, config: DataSourceConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def fetch(self, start: datetime, end: datetime) -> IngestionResult:
        """Fetch data for the given time window."""


class HTTPJSONSource(DataSource):
    """Base implementation for JSON APIs."""

    date_param: str = "start"
    end_param: str = "end"

    def build_params(self, start: datetime, end: datetime) -> MutableMapping[str, str]:
        params: MutableMapping[str, str] = {
            self.date_param: start.strftime("%Y-%m-%d"),
            self.end_param: end.strftime("%Y-%m-%d"),
        }
        if self.config.params:
            params.update(self.config.params)
        return params

    def fetch(self, start: datetime, end: datetime) -> IngestionResult:
        params = self.build_params(start, end)
        LOGGER.info("Fetching %s from %s", self.config.name, self.config.endpoint)
        response = requests.get(self.config.endpoint, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        data = pd.json_normalize(payload.get("results", payload))
        metadata = {
            "source": self.config.name,
            "endpoint": self.config.endpoint,
            "retrieved_at": datetime.utcnow().isoformat(),
            "start": params[self.date_param],
            "end": params[self.end_param],
        }
        return IngestionResult(data=data, metadata=metadata)


class GDELTSource(HTTPJSONSource):
    """Wrapper for the GDELT GKG events feed."""

    date_param = "startdatetime"
    end_param = "enddatetime"

    def build_params(self, start: datetime, end: datetime) -> MutableMapping[str, str]:
        params = super().build_params(start, end)
        params.setdefault("format", "json")
        return params


class ACLEDSource(HTTPJSONSource):
    """Wrapper around the ACLED API."""

    date_param = "event_date"
    end_param = "event_date_end"

    def build_params(self, start: datetime, end: datetime) -> MutableMapping[str, str]:
        params = super().build_params(start, end)
        params.setdefault("event_type", "Violence against civilians")
        params.setdefault("limit", "5000")
        return params


class SyntheticEconomicSource(DataSource):
    """Fallback synthetic generator for economic indicators."""

    def fetch(self, start: datetime, end: datetime) -> IngestionResult:
        date_range = pd.date_range(start=start, end=end, freq="D")
        df = pd.DataFrame(
            {
                "date": date_range,
                "food_price_index": 100 + pd.Series(range(len(date_range))).mod(13),
                "energy_price_index": 85 + pd.Series(range(len(date_range))).mod(17),
                "fx_volatility": 5 + pd.Series(range(len(date_range))).mod(11) / 10,
            }
        )
        metadata = {
            "source": self.config.name,
            "endpoint": self.config.endpoint,
            "retrieved_at": datetime.utcnow().isoformat(),
            "rows": str(len(df)),
            "synthetic": "true",
        }
        return IngestionResult(data=df, metadata=metadata)


def build_sources(configs: Iterable[DataSourceConfig]) -> Dict[str, DataSource]:
    """Instantiate data sources from configuration definitions."""

    registry: Dict[str, type[DataSource]] = {
        "gdelt": GDELTSource,
        "acled": ACLEDSource,
        "synthetic_economic": SyntheticEconomicSource,
    }
    sources: Dict[str, DataSource] = {}
    for cfg in configs:
        key = cfg.name.lower()
        if key not in registry:
            raise ValueError(f"Unsupported data source: {cfg.name}")
        sources[cfg.name] = registry[key](cfg)
    return sources


def fetch_all(
    sources: Mapping[str, DataSource],
    lookback_days: int = 30,
    end: Optional[datetime] = None,
) -> Dict[str, IngestionResult]:
    """Fetch data from all configured sources."""

    if end is None:
        end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    results: Dict[str, IngestionResult] = {}
    for name, source in sources.items():
        try:
            results[name] = source.fetch(start=start, end=end)
        except Exception as err:  # pragma: no cover - safety catch
            LOGGER.exception("Failed to fetch data from %s: %s", name, err)
    return results

