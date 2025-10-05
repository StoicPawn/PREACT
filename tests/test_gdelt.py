"""Tests for the enhanced GDELT data source."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from preact.config import DataSourceConfig
from preact.data_ingestion.sources.gdelt import GDELTQuery, GDELTSource


def test_gdelt_query_compilation() -> None:
    query = GDELTQuery(
        keywords=["climate change", "energy"],
        countries=["it"],
        themes=["ENV_CLIMATE"],
        tone_min=-5,
        tone_max=3,
        extra_terms=["sourceLanguage:ENGLISH"],
    )
    compiled = query.to_query(base="baseclause")
    assert "\"climate change\"" in compiled
    assert "sourceCountry:IT" in compiled
    assert "theme:ENV_CLIMATE" in compiled
    assert "tone>-5" in compiled and "tone<3" in compiled
    assert "sourceLanguage:ENGLISH" in compiled
    assert "baseclause" in compiled


def test_gdelt_normalise_payload() -> None:
    config = DataSourceConfig(name="gdelt", endpoint="https://example.com")
    source = GDELTSource(config)
    payload = {
        "results": [
            {
                "GLOBALEVENTID": "1",
                "SQLDATE": "20240301",
                "ActionGeo_CountryCode": "ita",
                "Actor1Name": "Government",
                "Actor2Name": "Protesters",
                "Themes": "ENV_CLIMATE",
                "SOURCEURL": "https://example.com/event",
                "NumArticles": "3",
                "AvgTone": "-1.5",
                "GoldsteinScale": "2.0",
                "ActionGeo_Lat": "12.34",
                "ActionGeo_Long": "56.78",
            }
        ]
    }
    start = datetime(2024, 3, 1)
    end = start + timedelta(days=1)
    frame = source._normalise(payload, start, end)
    assert list(frame.columns) == [
        "event_id",
        "event_date",
        "country",
        "actor1_country",
        "actor2_country",
        "actor1",
        "actor2",
        "themes",
        "source_url",
        "num_articles",
        "tone",
        "goldstein",
        "latitude",
        "longitude",
    ]
    assert frame.iloc[0]["country"] == "ITA"
    assert pytest.approx(frame.iloc[0]["tone"], rel=1e-3) == -1.5
    assert frame.iloc[0]["event_date"] == pd.Timestamp("2024-03-01")


def test_fetch_events_builds_query(monkeypatch: pytest.MonkeyPatch) -> None:
    config = DataSourceConfig(
        name="gdelt",
        endpoint="https://example.com",
        params={"query": "base", "sort": "DateDesc", "maxrecords": "50"},
    )
    source = GDELTSource(config)

    captured: dict[str, dict[str, str]] = {}

    class DummyResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:  # pragma: no cover - simple stub
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, params: dict[str, str], headers: dict[str, str] | None, timeout: int) -> DummyResponse:  # type: ignore[override]
        captured["params"] = params
        return DummyResponse(
            {
                "results": [
                    {
                        "GLOBALEVENTID": "2",
                        "SQLDATE": "20240302",
                        "ActionGeo_CountryCode": "FRA",
                    }
                ]
            }
        )

    monkeypatch.setattr("preact.data_ingestion.sources.gdelt.requests.get", fake_get)

    start = datetime(2024, 3, 1)
    end = start + timedelta(days=1)
    query = GDELTQuery(keywords=["drought"])
    result = source.fetch_events(start=start, end=end, query=query, limit=10, sort="DateAsc")

    assert "query" in captured["params"]
    assert "\"drought\"" in captured["params"]["query"]
    assert "base" in captured["params"]["query"]
    assert captured["params"]["sort"] == "DateAsc"
    assert result.metadata["rows"] == "1"
