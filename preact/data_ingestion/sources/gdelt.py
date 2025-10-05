"""Rich client for the GDELT events API."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, MutableMapping, Sequence

import pandas as pd
import requests

from .base import HTTPJSONSource, IngestionResult
from ...config import DataSourceConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GDELTQuery:
    """Structured description of a GDELT API search query."""

    keywords: Sequence[str] = field(default_factory=tuple)
    countries: Sequence[str] = field(default_factory=tuple)
    themes: Sequence[str] = field(default_factory=tuple)
    sources: Sequence[str] = field(default_factory=tuple)
    tone_min: float | None = None
    tone_max: float | None = None
    extra_terms: Sequence[str] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        """Return ``True`` if no filters are defined."""

        return (
            not self.keywords
            and not self.countries
            and not self.themes
            and not self.sources
            and self.tone_min is None
            and self.tone_max is None
            and not self.extra_terms
        )

    @staticmethod
    def _wrap_clause(values: Iterable[str], operator: str = "OR") -> str:
        quoted = [value.strip() for value in values if value]
        if not quoted:
            return ""
        if operator.upper() == "OR":
            joined = " OR ".join(quoted)
        else:
            joined = f" {operator} ".join(quoted)
        return f"({joined})"

    def _theme_clause(self) -> str:
        return self._wrap_clause((f"theme:{theme}" for theme in self.themes))

    def _country_clause(self) -> str:
        return self._wrap_clause(
            (f"sourceCountry:{country.upper()}" for country in self.countries)
        )

    def _source_clause(self) -> str:
        return self._wrap_clause(
            (f"sourceCollection:{source}" for source in self.sources)
        )

    def _tone_clause(self) -> str:
        clauses: list[str] = []
        if self.tone_min is not None:
            clauses.append(f"tone>{self.tone_min}")
        if self.tone_max is not None:
            clauses.append(f"tone<{self.tone_max}")
        return self._wrap_clause(clauses, operator="AND") if clauses else ""

    def to_query(self, base: str | None = None) -> str:
        """Return a query string compatible with the GDELT API."""

        clauses: list[str] = []
        if base:
            clauses.append(base)
        if self.keywords:
            clauses.append(
                self._wrap_clause((f'"{keyword}"' for keyword in self.keywords))
            )
        clause_builders = (
            self._country_clause,
            self._theme_clause,
            self._source_clause,
            self._tone_clause,
        )
        for builder in clause_builders:
            clause = builder()
            if clause:
                clauses.append(clause)
        if self.extra_terms:
            clauses.extend(self.extra_terms)
        return " AND ".join(clause for clause in clauses if clause)


class GDELTSource(HTTPJSONSource):
    """Wrapper for the GDELT events feed with rich query support."""

    date_param = "startdatetime"
    end_param = "enddatetime"

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        params = dict(config.params or {})
        self.default_query = params.pop("query", None)
        self.default_mode = params.pop("mode", "Events")
        self.default_sort = params.pop("sort", "DateDesc")
        self.response_format = params.pop("format", "json")
        self.max_records = int(params.pop("maxrecords", 250))
        self.extra_params = params

    def build_params(self, start: datetime, end: datetime) -> MutableMapping[str, str]:
        params: MutableMapping[str, str] = {
            self.date_param: start.strftime("%Y%m%d%H%M%S"),
            self.end_param: end.strftime("%Y%m%d%H%M%S"),
            "mode": self.default_mode,
            "sort": self.default_sort,
            "maxrecords": str(self.max_records),
            "format": self.response_format,
        }
        params.update(self.extra_params)
        return params

    def _fallback(self, start: datetime, end: datetime) -> pd.DataFrame:
        date_range = pd.date_range(start=start, end=end, freq="D")
        countries = ["BFA", "MLI", "NER"]
        records = []
        for idx, date in enumerate(date_range):
            records.append(
                {
                    "event_id": f"synthetic-{date:%Y%m%d}-{idx}",
                    "event_date": date,
                    "country": countries[idx % len(countries)],
                    "actor1": "Synthetic Actor",
                    "actor2": "Synthetic Counterpart",
                    "themes": "SYNTHETIC",
                    "source_url": "",
                    "tone": 0.0,
                    "goldstein": 0.0,
                    "num_articles": 1,
                    "latitude": None,
                    "longitude": None,
                }
            )
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _coalesce_columns(data: pd.DataFrame, *candidates: str, default: str | None = None) -> pd.Series:
        for column in candidates:
            if column in data.columns:
                return data[column]
        if default is not None:
            return pd.Series([default] * len(data))
        raise KeyError("None of the candidate columns are present")

    def _normalise(self, payload: object, start: datetime, end: datetime) -> pd.DataFrame:
        raw = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
        data = pd.json_normalize(raw)
        if data.empty:
            return self._fallback(start, end)

        tidy = pd.DataFrame()
        tidy["event_id"] = self._coalesce_columns(data, "GLOBALEVENTID", "GlobalEventID", default="")
        if "SQLDATE" in data.columns:
            tidy["event_date"] = pd.to_datetime(data["SQLDATE"], format="%Y%m%d", errors="coerce")
        elif "EventDate" in data.columns:
            tidy["event_date"] = pd.to_datetime(data["EventDate"], errors="coerce")
        else:
            tidy["event_date"] = pd.date_range(start=start, periods=len(data), freq="D")

        tidy["country"] = (
            self._coalesce_columns(
                data,
                "ActionGeo_CountryCode",
                "Actor1CountryCode",
                "Actor2CountryCode",
                default="GLOBAL",
            )
            .fillna("GLOBAL")
            .str.upper()
        )
        tidy["actor1"] = self._coalesce_columns(
            data, "Actor1Name", "Actor1Code", default="UNKNOWN"
        ).fillna("UNKNOWN")
        tidy["actor2"] = self._coalesce_columns(
            data, "Actor2Name", "Actor2Code", default="UNKNOWN"
        ).fillna("UNKNOWN")
        tidy["themes"] = self._coalesce_columns(
            data, "Themes", "SourceCommonName", default=""
        ).fillna("")
        tidy["source_url"] = self._coalesce_columns(
            data, "SOURCEURL", "DocumentIdentifier", default=""
        ).fillna("")
        tidy["num_articles"] = pd.to_numeric(
            self._coalesce_columns(data, "NumArticles", default="0"), errors="coerce"
        ).fillna(0)
        tidy["tone"] = pd.to_numeric(
            self._coalesce_columns(data, "AvgTone", default="0"), errors="coerce"
        ).fillna(0.0)
        tidy["goldstein"] = pd.to_numeric(
            self._coalesce_columns(data, "GoldsteinScale", default="0"), errors="coerce"
        ).fillna(0.0)
        tidy["latitude"] = pd.to_numeric(
            self._coalesce_columns(
                data,
                "ActionGeo_Lat",
                "Actor1Geo_Lat",
                "Actor2Geo_Lat",
                default="nan",
            ),
            errors="coerce",
        )
        tidy["longitude"] = pd.to_numeric(
            self._coalesce_columns(
                data,
                "ActionGeo_Long",
                "Actor1Geo_Long",
                "Actor2Geo_Long",
                default="nan",
            ),
            errors="coerce",
        )

        tidy = tidy.sort_values("event_date").reset_index(drop=True)
        tidy["event_date"] = pd.to_datetime(tidy["event_date"], errors="coerce")
        tidy["event_date"] = tidy["event_date"].fillna(pd.Timestamp(start))
        return tidy

    def fetch_events(
        self,
        start: datetime,
        end: datetime,
        query: GDELTQuery | None = None,
        limit: int | None = None,
        sort: str | None = None,
    ) -> IngestionResult:
        """Fetch events from GDELT within the provided temporal window."""

        params = self.build_params(start, end)
        if limit is not None:
            params["maxrecords"] = str(min(limit, self.max_records))
        if sort:
            params["sort"] = sort
        elif self.default_sort:
            params.setdefault("sort", self.default_sort)

        if query and not query.is_empty():
            params["query"] = query.to_query(self.default_query)
        elif self.default_query:
            params.setdefault("query", self.default_query)

        params, headers = self._apply_auth(params)
        metadata = self._build_metadata(params)
        metadata["start_iso"] = start.isoformat()
        metadata["end_iso"] = end.isoformat()
        metadata["query"] = params.get("query", "")
        metadata["lookback_days"] = str((end - start).days)

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

    def fetch(self, start: datetime, end: datetime) -> IngestionResult:
        return self.fetch_events(start, end)

    def recent_events(
        self,
        days: int = 7,
        end: datetime | None = None,
        query: GDELTQuery | None = None,
        limit: int | None = None,
    ) -> IngestionResult:
        """Convenience helper to fetch events in the most recent window."""

        end = end or datetime.utcnow()
        start = end - timedelta(days=days)
        return self.fetch_events(start=start, end=end, query=query, limit=limit)
