"""Economic indicator data source."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, Iterable

import pandas as pd
import requests

from ...config import DataSourceConfig
from .base import DataSource, IngestionResult

LOGGER = logging.getLogger(__name__)


class EconomicIndicatorSource(DataSource):
    """Retrieve macro-economic indicators from external APIs with fallback support."""

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        params = self.config.params or {}
        raw_indicators = params.get("indicators", "")
        indicators = [item.strip() for item in raw_indicators.split(",") if item.strip()]
        if not indicators:
            indicators = ["FP.CPI.TOTL.ZG", "NY.GDP.MKTP.KD.ZG"]
        self.indicators = indicators
        self.country = params.get("country", "WLD")
        self.per_page = params.get("per_page", "2000")
        aliases: Dict[str, str] = {}
        alias_param = params.get("aliases")
        if alias_param:
            try:
                aliases = json.loads(alias_param)
            except json.JSONDecodeError:
                LOGGER.warning(
                    "Invalid aliases configuration for %s: %s", self.config.name, alias_param
                )
        self.aliases = aliases

    def _indicator_column_name(self, indicator: str) -> str:
        if indicator in self.aliases:
            return self.aliases[indicator]
        slug = indicator.lower().replace(".", "_")
        return slug

    def _fetch_indicator(
        self, indicator: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        endpoint = self.config.endpoint.format(country=self.country, indicator=indicator)
        params = {
            "format": "json",
            "per_page": self.per_page,
            "date": f"{start.year - 5}:{end.year}",
        }
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        records: list[dict]
        if isinstance(payload, list):
            records = payload[1] if len(payload) > 1 and payload[1] else []
        else:
            records = payload.get("data", payload.get("results", []))  # type: ignore[assignment]
        frame = pd.DataFrame.from_records(records)
        column_name = self._indicator_column_name(indicator)
        if frame.empty or "value" not in frame.columns:
            return pd.DataFrame(columns=["date", column_name])
        frame["date"] = pd.to_datetime(frame["date"].astype(str) + "-01-01", errors="coerce")
        frame[column_name] = pd.to_numeric(frame["value"], errors="coerce")
        tidy = frame[["date", column_name]].dropna(subset=["date", column_name])
        tidy = tidy.groupby("date", as_index=False)[column_name].mean()
        return tidy

    def _merge_frames(self, frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
        frames_list = [frame for frame in frames if not frame.empty]
        if not frames_list:
            return pd.DataFrame(columns=["date"])
        merged = frames_list[0]
        for frame in frames_list[1:]:
            merged = pd.merge(merged, frame, on="date", how="outer")
        return merged.sort_values("date")

    def _indicator_names(self) -> Iterable[str]:
        if self.indicators:
            return [self._indicator_column_name(ind) for ind in self.indicators]
        return ["economic_indicator"]

    def _fallback(self, start: datetime, end: datetime) -> pd.DataFrame:
        date_range = pd.date_range(start=start, end=end, freq="D")
        data: Dict[str, pd.Series] = {"date": pd.Series(date_range)}
        base_series = pd.Series(range(len(date_range)), dtype=float)
        for idx, name in enumerate(self._indicator_names()):
            scale = (idx + 1) * 0.5
            data[name] = 95 + (base_series * scale).mod(10)
        return pd.DataFrame(data)

    def fetch(self, start: datetime, end: datetime) -> IngestionResult:
        metadata = {
            "source": self.config.name,
            "endpoint": self.config.endpoint,
            "retrieved_at": datetime.utcnow().isoformat(),
            "country": self.country,
            "indicators": ",".join(self.indicators),
        }
        try:
            frames = [self._fetch_indicator(indicator, start, end) for indicator in self.indicators]
            combined = self._merge_frames(frames)
            if combined.empty:
                raise ValueError("No economic indicator data retrieved")
            combined = combined.drop_duplicates(subset=["date"]).dropna(how="all")
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.set_index("date").sort_index()
            full_index = pd.date_range(start=combined.index.min(), end=end, freq="D")
            combined = combined.reindex(full_index).ffill().dropna(how="all")
            combined = combined.loc[start:end]
            combined.index.name = "date"
            tidy = combined.reset_index()
            metadata["fallback"] = "false"
        except Exception as err:  # pragma: no cover - network/intermittent safety
            LOGGER.warning("Economic indicator fetch failed, using fallback: %s", err)
            tidy = self._fallback(start, end)
            metadata["fallback"] = "true"
            metadata["error"] = str(err)
        metadata["rows"] = str(len(tidy))
        return IngestionResult(data=tidy, metadata=metadata)
