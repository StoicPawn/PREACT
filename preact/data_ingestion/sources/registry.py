"""Registry utilities for data source instantiation."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Iterable, Mapping, Optional

from ...config import DataSourceConfig
from .acled import ACLEDSource
from .base import DataSource, IngestionResult
from .economic import EconomicIndicatorSource
from .gdelt import GDELTSource
from .hdx import HDXSource
from .unhcr import UNHCRSource

LOGGER = logging.getLogger(__name__)

SOURCE_REGISTRY: Dict[str, type[DataSource]] = {
    "gdelt": GDELTSource,
    "acled": ACLEDSource,
    "economic_indicators": EconomicIndicatorSource,
    "unhcr": UNHCRSource,
    "hdx": HDXSource,
}


def build_sources(configs: Iterable[DataSourceConfig]) -> Dict[str, DataSource]:
    """Instantiate data sources from configuration definitions."""

    sources: Dict[str, DataSource] = {}
    for cfg in configs:
        key = cfg.name.lower()
        if key not in SOURCE_REGISTRY:
            raise ValueError(f"Unsupported data source: {cfg.name}")
        sources[cfg.name] = SOURCE_REGISTRY[key](cfg)
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
