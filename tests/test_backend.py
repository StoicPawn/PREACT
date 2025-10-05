"""Tests for the PREACT FastAPI backend."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
try:  # pragma: no cover - prefer real FastAPI when available
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - fallback to stub implementation
    from preact.backend._fastapi_stub import install_fastapi_stub

    install_fastapi_stub()
    from fastapi.testclient import TestClient  # type: ignore

from preact.backend.app import create_app
from preact.config import DataSourceConfig
from preact.simulation.service import SimulationService
from preact.simulation.storage import SimulationRepository
from preact.data_ingestion.orchestrator import IngestionArtifacts
from preact.data_ingestion.sources import GDELTSource, IngestionResult


class DummyOrchestrator:
    """Stub orchestrator used to exercise the API layer."""

    def __init__(self) -> None:
        config = DataSourceConfig(name="gdelt", endpoint="https://example.com")
        self.gdelt_source = GDELTSource(config)
        self.sources = {"gdelt": self.gdelt_source}
        self.last_run: dict[str, object] | None = None
        self.recent_args: dict[str, object] | None = None

        gdelt_frame = pd.DataFrame(
            [
                {
                    "event_id": "1",
                    "event_date": pd.Timestamp("2024-03-01"),
                    "country": "ITA",
                    "actor1_country": "ITA",
                    "actor2_country": "FRA",
                    "actor1": "Government",
                    "actor2": "Citizens",
                    "themes": "POL_GOV",
                    "tone": -1.2,
                    "goldstein": 2.5,
                    "num_articles": 4,
                    "source_url": "https://example.com/event",
                    "latitude": 12.34,
                    "longitude": 45.67,
                }
            ]
        )
        self.gdelt_result = IngestionResult(
            data=gdelt_frame,
            metadata={"rows": "1", "fallback": "false"},
        )

        def fake_recent_events(*, days: int, query, limit: int | None):  # type: ignore[no-untyped-def]
            self.recent_args = {"days": days, "query": query, "limit": limit}
            return self.gdelt_result

        self.gdelt_source.recent_events = fake_recent_events  # type: ignore[assignment]

        bronze = {"gdelt": self.gdelt_result}
        silver = {"gdelt": gdelt_frame}
        gold = {"combined": gdelt_frame}
        self.artifacts = IngestionArtifacts(bronze=bronze, silver=silver, gold=gold)

    def run(self, lookback_days=None, end=None, persist=None):  # type: ignore[no-untyped-def]
        self.last_run = {
            "lookback_days": lookback_days,
            "end": end,
            "persist": persist,
        }
        return self.artifacts


def create_test_client(tmp_path) -> tuple[TestClient, DummyOrchestrator, SimulationService]:
    orchestrator = DummyOrchestrator()
    repository = SimulationRepository(tmp_path / "sim.duckdb", export_dir=tmp_path / "exports")
    service = SimulationService(repository=repository)
    app = create_app(orchestrator=orchestrator, simulation_service=service)
    client = TestClient(app)
    return client, orchestrator, service


def test_health_endpoint_reports_sources(tmp_path) -> None:
    client, orchestrator, _service = create_test_client(tmp_path)
    response = client.get("/health")
    payload = response.json()
    assert payload["status"] == "ok"
    assert "gdelt" in payload["sources"]
    assert isinstance(datetime.fromisoformat(payload["timestamp"]), datetime)
    assert orchestrator.last_run is None


def test_ingest_endpoint_runs_pipeline(tmp_path) -> None:
    client, orchestrator, _service = create_test_client(tmp_path)
    response = client.post("/ingest", json={"lookback_days": 10, "persist": True})
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["summary"]["bronze"]["gdelt"] == 1
    assert orchestrator.last_run == {"lookback_days": 10, "end": None, "persist": True}


def test_gdelt_events_endpoint_filters(tmp_path) -> None:
    client, orchestrator, _service = create_test_client(tmp_path)
    response = client.get("/gdelt/events", params={"country": "ita", "limit": 5})
    payload = response.json()
    assert payload["events"][0]["country"] == "ITA"
    assert payload["metadata"]["rows"] == 1
    assert payload["metadata"]["fallback"] is False
    assert orchestrator.recent_args["days"] == 7
    assert orchestrator.recent_args["limit"] == 5
    query = orchestrator.recent_args["query"]
    assert query is not None and query.countries[0].lower() == "ita"


def test_gdelt_state_graph_endpoint(tmp_path) -> None:
    client, orchestrator, _service = create_test_client(tmp_path)
    response = client.get(
        "/gdelt/state-graph",
        params={"lookback_days": 14, "limit": 50, "min_events": 1},
    )
    payload = response.json()
    assert payload["edges"][0]["source"] == "ITA"
    assert payload["edges"][0]["target"] == "FRA"
    states = {node["state"] for node in payload["nodes"]}
    assert {"ITA", "FRA"}.issubset(states)
    assert payload["metadata"]["edges"] == 1
    assert orchestrator.recent_args["days"] == 14

def test_simulation_run_and_results_endpoint(tmp_path) -> None:
    client, _orchestrator, _service = create_test_client(tmp_path)
    scenarios_resp = client.get("/simulation/scenarios")
    scenarios_payload = scenarios_resp.json()
    assert scenarios_payload["scenarios"]
    template_name = scenarios_payload["scenarios"][0]["name"]

    run_payload = {
        "template": template_name,
        "horizon": 4,
        "policy": {
            "brackets": [{"threshold": 30000, "rate": 0.2}],
            "base_deduction": 4500,
            "child_subsidy": 180.0,
            "unemployment_benefit": 950.0,
        },
    }
    run_resp = client.post("/simulation/run", json=run_payload)
    run_data = run_resp.json()
    assert run_data["base_run_id"]
    assert run_data["reform_run_id"]
    assert run_data["comparison"] is not None

    results_resp = client.get(f"/simulation/results/{run_data['base_run_id']}")
    assert results_resp.status_code == 200
    results_payload = results_resp.json()
    assert results_payload["kpis"]["tax_revenue"] >= 0
    assert results_payload["timeline"]

