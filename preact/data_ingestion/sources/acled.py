"""ACLED data source implementation."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, MutableMapping

import pandas as pd
import requests

from ...config import DataSourceConfig
from .base import HTTPJSONSource, IngestionResult

LOGGER = logging.getLogger(__name__)


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
