"""Base classes and utilities shared by data sources."""
from __future__ import annotations

import abc
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import pandas as pd
import requests

from ...config import DataSourceConfig

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
