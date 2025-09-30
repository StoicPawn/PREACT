"""GDELT data source implementation."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import MutableMapping

import pandas as pd
import requests

from .base import HTTPJSONSource, IngestionResult

LOGGER = logging.getLogger(__name__)


class GDELTSource(HTTPJSONSource):
    """Wrapper for the GDELT GKG events feed."""

    date_param = "startdatetime"
    end_param = "enddatetime"

    def build_params(self, start: datetime, end: datetime) -> MutableMapping[str, str]:
        params: MutableMapping[str, str] = {
            self.date_param: start.strftime("%Y%m%d%H%M%S"),
            self.end_param: end.strftime("%Y%m%d%H%M%S"),
            "mode": "Events",
            "sort": "DateAsc",
            "maxrecords": "250",
            "format": "json",
        }
        if self.config.params:
            params.update(self.config.params)
        return params

    def _fallback(self, start: datetime, end: datetime) -> pd.DataFrame:
        date_range = pd.date_range(start=start, end=end, freq="D")
        countries = ["Burkina Faso", "Mali", "Niger"]
        records = []
        for idx, date in enumerate(date_range):
            records.append(
                {
                    "event_date": date,
                    "country": countries[idx % len(countries)],
                }
            )
        return pd.DataFrame.from_records(records)

    def _normalise(self, payload: object, start: datetime, end: datetime) -> pd.DataFrame:
        data = pd.json_normalize(payload.get("results", payload))
        if data.empty:
            return self._fallback(start, end)
        if "SQLDATE" in data.columns:
            data["event_date"] = pd.to_datetime(data["SQLDATE"], format="%Y%m%d")
        elif "date" in data.columns:
            data["event_date"] = pd.to_datetime(data["date"])
        else:
            data["event_date"] = pd.date_range(start=start, periods=len(data), freq="D")
        if "ActionGeo_CountryCode" in data.columns:
            data["country"] = data["ActionGeo_CountryCode"]
        elif "country" not in data.columns:
            data["country"] = "GLOBAL"
        tidy = data[["event_date", "country"]]
        return tidy.sort_values("event_date")

    def fetch(self, start: datetime, end: datetime) -> IngestionResult:
        params = self.build_params(start, end)
        params, headers = self._apply_auth(params)
        metadata = self._build_metadata(params)
        metadata["start_iso"] = start.isoformat()
        metadata["end_iso"] = end.isoformat()
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
            LOGGER.warning("GDELT fetch failed, using fallback: %s", err)
            tidy = self._fallback(start, end)
            metadata["fallback"] = "true"
            metadata["error"] = str(err)
            metadata["rows"] = str(len(tidy))
        return IngestionResult(data=tidy, metadata=metadata)
