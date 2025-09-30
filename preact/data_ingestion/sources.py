"""Data ingestion interfaces for the PREACT system."""
from __future__ import annotations

import abc
import json
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

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

    def _build_metadata(self, params: Mapping[str, str]) -> Dict[str, str]:
        return {
            "source": self.config.name,
            "endpoint": self.config.endpoint,
            "retrieved_at": datetime.utcnow().isoformat(),
            "start": params.get(self.date_param, ""),
            "end": params.get(self.end_param, ""),
        }

    def fetch(self, start: datetime, end: datetime) -> IngestionResult:
        params = self.build_params(start, end)
        params, headers = self._apply_auth(params)
        LOGGER.info("Fetching %s from %s", self.config.name, self.config.endpoint)
        response = requests.get(
            self.config.endpoint, params=params, headers=headers or None, timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        data = pd.json_normalize(payload.get("results", payload))
        metadata = self._build_metadata(params)
        return IngestionResult(data=data, metadata=metadata)

    def _apply_auth(
        self, params: MutableMapping[str, str]
    ) -> tuple[MutableMapping[str, str], Dict[str, str] | None]:
        headers: Dict[str, str] | None = None
        if self.config.headers:
            headers = dict(self.config.headers)

        key_value: Optional[str] = None
        if self.config.key_env_var:
            key_value = os.environ.get(self.config.key_env_var)
            if not key_value and self.config.requires_key:
                raise RuntimeError(
                    "API key required but environment variable "
                    f"{self.config.key_env_var} is not set"
                )
        elif self.config.requires_key:
            raise RuntimeError(
                "Data source requires an API key but no key_env_var was provided"
            )

        if key_value:
            if self.config.key_param:
                params.setdefault(self.config.key_param, key_value)
            else:
                inserted = False
                if headers is None:
                    headers = {}
                updated_headers: Dict[str, str] = {}
                for name, value in headers.items():
                    if "{key}" in value:
                        updated_headers[name] = value.format(key=key_value)
                        inserted = True
                    else:
                        updated_headers[name] = value
                headers = updated_headers
                if not inserted:
                    headers.setdefault("Authorization", key_value)

        return params, headers


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


class ACLEDSource(HTTPJSONSource):
    """Wrapper around the ACLED API."""

    date_param = "event_date"
    end_param = "event_date_end"
    token_endpoint = "https://acleddata.com/oauth/token"

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        self._access_token: str | None = None
        self._token_expiry: datetime | None = None

    def build_params(self, start: datetime, end: datetime) -> MutableMapping[str, str]:
        params = super().build_params(start, end)
        params.setdefault("event_type", "Violence against civilians")
        params.setdefault("limit", "5000")
        return params

    def _apply_auth(
        self, params: MutableMapping[str, str]
    ) -> tuple[MutableMapping[str, str], Dict[str, str] | None]:
        headers: Dict[str, str] | None = None
        if self.config.headers:
            headers = dict(self.config.headers)

        token = self._get_access_token()

        if headers is None:
            headers = {"Authorization": f"Bearer {token}"}
        else:
            inserted = False
            updated_headers: Dict[str, str] = {}
            for name, value in headers.items():
                if "{key}" in value:
                    updated_headers[name] = value.format(key=token)
                    inserted = True
                else:
                    updated_headers[name] = value
            if not inserted:
                updated_headers.setdefault("Authorization", f"Bearer {token}")
            headers = updated_headers

        return params, headers

    def _get_access_token(self) -> str:
        now = datetime.utcnow()
        if self._access_token and self._token_expiry and now < self._token_expiry:
            return self._access_token

        username = os.environ.get("ACLED_USERNAME")
        password = os.environ.get("ACLED_PASSWORD")
        if not username or not password:
            raise RuntimeError(
                "ACLED credentials are not configured. "
                "Set ACLED_USERNAME and ACLED_PASSWORD environment variables."
            )

        payload: Dict[str, str] = {
            "grant_type": "password",
            "username": username,
            "password": password,
        }

        client_id = os.environ.get("ACLED_CLIENT_ID")
        if client_id:
            payload["client_id"] = client_id

        client_secret = os.environ.get("ACLED_CLIENT_SECRET")
        if client_secret:
            payload["client_secret"] = client_secret

        response = requests.post(self.token_endpoint, data=payload, timeout=30)
        response.raise_for_status()
        token_payload = response.json()

        token = token_payload.get("access_token")
        if not token:
            raise RuntimeError("ACLED OAuth response did not include an access_token")

        expires_in = token_payload.get("expires_in")
        try:
            expires_seconds = int(expires_in)
        except (TypeError, ValueError):
            expires_seconds = 3600

        # Refresh the token slightly before it expires to avoid race conditions.
        self._token_expiry = now + timedelta(seconds=max(expires_seconds - 60, 60))
        self._access_token = token
        return token

    def _fallback(self, start: datetime, end: datetime) -> pd.DataFrame:
        date_range = pd.date_range(start=start, end=end, freq="D")
        countries = ["Burkina Faso", "Mali", "Niger"]
        records = []
        for idx, date in enumerate(date_range):
            records.append(
                {
                    "event_date": date,
                    "country": countries[idx % len(countries)],
                    "fatalities": (idx % 3),
                }
            )
        return pd.DataFrame.from_records(records)

    def _normalise(self, payload: object, start: datetime, end: datetime) -> pd.DataFrame:
        data = pd.json_normalize(payload.get("data", payload.get("results", payload)))
        if data.empty:
            return self._fallback(start, end)
        if "event_date" in data.columns:
            data["event_date"] = pd.to_datetime(data["event_date"])
        elif "date" in data.columns:
            data["event_date"] = pd.to_datetime(data["date"])
        else:
            data["event_date"] = pd.date_range(start=start, periods=len(data), freq="D")
        if "country" not in data.columns:
            for candidate in ("country", "country_name", "admin1"):
                if candidate in data.columns:
                    data["country"] = data[candidate]
                    break
        if "country" not in data.columns:
            data["country"] = "GLOBAL"
        tidy = data[["event_date", "country"]]
        return tidy.sort_values("event_date")

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
            LOGGER.warning("ACLED fetch failed, using fallback: %s", err)
            tidy = self._fallback(start, end)
            metadata["fallback"] = "true"
            metadata["error"] = str(err)
            metadata["rows"] = str(len(tidy))
        return IngestionResult(data=tidy, metadata=metadata)


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
        records: list[MutableMapping[str, object]]
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


class UNHCRSource(HTTPJSONSource):
    """Connector for UNHCR population statistics with synthetic fallback."""

    date_param = "startDate"
    end_param = "endDate"

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


class HDXSource(HTTPJSONSource):
    """Connector for HDX humanitarian operations with graceful degradation."""

    date_param = "startDate"
    end_param = "endDate"

    def build_params(self, start: datetime, end: datetime) -> MutableMapping[str, str]:
        params = super().build_params(start, end)
        params.setdefault("format", "json")
        params.setdefault("indicator", "access")
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
                    "access_constraint_score": 0.3 + (idx % 5) * 0.1,
                }
            )
        return pd.DataFrame.from_records(records)

    def _normalise(self, payload: object, start: datetime, end: datetime) -> pd.DataFrame:
        data = pd.json_normalize(payload.get("data", payload.get("results", payload)))
        if data.empty:
            return self._fallback(start, end)
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
        elif "report_date" in data.columns:
            data["date"] = pd.to_datetime(data["report_date"])
        else:
            data["date"] = pd.date_range(start=start, periods=len(data), freq="D")
        if "country" not in data.columns:
            for candidate in ("admin1_name", "country_name", "iso3"):
                if candidate in data.columns:
                    data["country"] = data[candidate]
                    break
        if "country" not in data.columns:
            data["country"] = "GLOBAL"
        value_col = None
        for candidate in ("access", "value", "score", "constraint"):
            if candidate in data.columns:
                value_col = candidate
                break
        if value_col is None:
            data["access_constraint_score"] = 0.0
            value_col = "access_constraint_score"
        data = data.rename(columns={value_col: "access_constraint_score"})
        tidy = data[["date", "country", "access_constraint_score"]]
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
            LOGGER.warning("HDX fetch failed, using fallback: %s", err)
            tidy = self._fallback(start, end)
            metadata["fallback"] = "true"
            metadata["error"] = str(err)
            metadata["rows"] = str(len(tidy))
        return IngestionResult(data=tidy, metadata=metadata)


def build_sources(configs: Iterable[DataSourceConfig]) -> Dict[str, DataSource]:
    """Instantiate data sources from configuration definitions."""

    registry: Dict[str, type[DataSource]] = {
        "gdelt": GDELTSource,
        "acled": ACLEDSource,
        "economic_indicators": EconomicIndicatorSource,
        "unhcr": UNHCRSource,
        "hdx": HDXSource,
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

