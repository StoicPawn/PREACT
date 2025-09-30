"""UNHCR data source implementation."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import MutableMapping

import pandas as pd
import requests

from ...config import DataSourceConfig
from .base import HTTPJSONSource, IngestionResult

LOGGER = logging.getLogger(__name__)


class UNHCRSource(HTTPJSONSource):
    """Connector for UNHCR population statistics with synthetic fallback."""

    date_param = "startDate"
    end_param = "endDate"

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)

    def build_params(self, start: datetime, end: datetime) -> MutableMapping[str, str]:
        params = super().build_params(start, end)
        params.setdefault("format", "json")
        params.setdefault("population_group", "refugees")
        return params

    def _fallback(self, start: datetime, end: datetime) -> pd.DataFrame:
        date_range = pd.date_range(start=start, end=end, freq="D")
        countries = ["Burkina Faso", "Mali", "Niger"]
        records = []
        for idx, date in enumerate(date_range):
            country = countries[idx % len(countries)]
            records.append(
                {
                    "date": date,
                    "country": country,
                    "displaced_population": 5000 + (idx % 7) * 250,
                }
            )
        return pd.DataFrame.from_records(records)

    def _normalise(self, payload: object, start: datetime, end: datetime) -> pd.DataFrame:
        data = pd.json_normalize(payload.get("data", payload.get("results", payload)))
        if data.empty:
            return self._fallback(start, end)
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
        elif "record_date" in data.columns:
            data["date"] = pd.to_datetime(data["record_date"])
        else:
            data["date"] = pd.date_range(start=start, periods=len(data), freq="D")
        if "country" not in data.columns:
            for candidate in ("coo_name", "country_origin", "origin_country_name"):
                if candidate in data.columns:
                    data["country"] = data[candidate]
                    break
        if "country" not in data.columns:
            data["country"] = "GLOBAL"
        value_col = None
        for candidate in ("individuals", "value", "population", "people", "count"):
            if candidate in data.columns:
                value_col = candidate
                break
        if value_col is None:
            data["displaced_population"] = 0
            value_col = "displaced_population"
        data = data.rename(columns={value_col: "displaced_population"})
        tidy = data[["date", "country", "displaced_population"]]
        return tidy.sort_values("date")

    def fetch(self, start: datetime, end: datetime) -> IngestionResult:
        params = self.build_params(start, end)
        params, headers = self._apply_auth(params)
        metadata = self._build_metadata(params)
        try:
            response = requests.get(
                self.config.endpoint,
                params=params,
                headers=headers or None,
                timeout=30,
            )
            response.raise_for_status()
            tidy = self._normalise(response.json(), start, end)
            metadata["fallback"] = "false"
            metadata["rows"] = str(len(tidy))
        except Exception as err:  # pragma: no cover - network safety
            LOGGER.warning("UNHCR fetch failed, using fallback: %s", err)
            tidy = self._fallback(start, end)
            metadata["fallback"] = "true"
            metadata["error"] = str(err)
            metadata["rows"] = str(len(tidy))
        return IngestionResult(data=tidy, metadata=metadata)
