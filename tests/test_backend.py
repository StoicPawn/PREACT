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
from preact.data_hub.news_store import SharedNewsStore
from preact.history.world_knowledge_store import WorldKnowledgeStore
from preact.intelligence.knowledge_update import AutonomousKnowledgeUpdater
from preact.intelligence.world_events import WorldEventObservation
from preact.intelligence.world_knowledge import (
    KnowledgeDomain,
    KnowledgeUpdateCandidate,
    KnowledgeUpdateKind,
    SourceEvidence,
)


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



def test_world_knowledge_status_requires_configured_store(tmp_path) -> None:
    client, _orchestrator, _service = create_test_client(tmp_path)
    with pytest.raises(Exception) as exc_info:
        client.get("/world/knowledge/status")
    assert getattr(exc_info.value, "status_code", None) == 503


def test_world_country_knowledge_endpoint_is_point_in_time(tmp_path) -> None:
    orchestrator = DummyOrchestrator()
    repository = SimulationRepository(tmp_path / "sim-world.duckdb", export_dir=tmp_path / "exports-world")
    service = SimulationService(repository=repository)
    store = WorldKnowledgeStore(tmp_path / "world-knowledge.duckdb")
    updater = AutonomousKnowledgeUpdater(store)

    first_time = datetime.fromisoformat("2026-09-26T12:00:00+00:00")
    second_time = datetime.fromisoformat("2026-09-27T12:00:00+00:00")

    def official(value: str, when: datetime, suffix: str) -> KnowledgeUpdateCandidate:
        source = SourceEvidence(
            source="official",
            source_ref=f"https://example.com/{suffix}",
            published_at=when,
            retrieved_at=when,
            independent_group="government",
            authoritative=True,
        )
        return KnowledgeUpdateCandidate(
            entity_id="country:ITA",
            field="head_of_government",
            value=value,
            valid_from=when,
            detected_at=when,
            domain=KnowledgeDomain.GOVERNANCE,
            kind=KnowledgeUpdateKind.FACT,
            confidence=0.95,
            evidence=(source,),
        )

    updater.process(official("Person A", first_time, "a"))
    updater.process(official("Person B", second_time, "b"))

    app = create_app(
        orchestrator=orchestrator,
        simulation_service=service,
        world_knowledge_store=store,
    )
    client = TestClient(app)

    current = client.get("/world/countries/ita/knowledge")
    assert current.status_code == 200
    current_payload = current.json()
    assert current_payload["knowledge"]["head_of_government"]["value"] == "Person B"
    assert current_payload["semantic_contract"]["forecasts_do_not_mutate_facts"] is True

    historical = client.get(
        "/world/countries/ITA/knowledge",
        params={
            "as_of": first_time.isoformat(),
            "known_cutoff": first_time.isoformat(),
        },
    )
    assert historical.status_code == 200
    assert historical.json()["knowledge"]["head_of_government"]["value"] == "Person A"

    status = client.get("/world/knowledge/status")
    assert status.status_code == 200
    assert status.json()["current_assertions"] == 1
    assert status.json()["queued_narrative_jobs"] >= 1


def test_world_country_events_endpoint_is_point_in_time(tmp_path) -> None:
    orchestrator = DummyOrchestrator()
    repository = SimulationRepository(
        tmp_path / "sim-events.duckdb",
        export_dir=tmp_path / "exports-events",
    )
    service = SimulationService(repository=repository)
    store = WorldKnowledgeStore(tmp_path / "world-events-api.duckdb")

    first_known = datetime.fromisoformat("2026-09-27T09:00:00+00:00")
    revised_known = datetime.fromisoformat("2026-09-27T10:00:00+00:00")
    event_time = datetime.fromisoformat("2026-09-27T08:00:00+00:00")

    store.record_world_events(
        [
            WorldEventObservation(
                provider="gdelt",
                provider_event_id="event-1",
                event_time=event_time,
                known_at=first_known,
                actor1_entity_id="country:ITA",
                actor2_entity_id="country:FRA",
                event_code="040",
                num_articles=2,
                snapshot_checksum="v1",
            ),
            WorldEventObservation(
                provider="gdelt",
                provider_event_id="event-1",
                event_time=event_time,
                known_at=revised_known,
                actor1_entity_id="country:ITA",
                actor2_entity_id="country:FRA",
                event_code="040",
                num_articles=7,
                snapshot_checksum="v2",
            ),
        ]
    )

    app = create_app(
        orchestrator=orchestrator,
        simulation_service=service,
        world_knowledge_store=store,
    )
    client = TestClient(app)

    early = client.get(
        "/world/countries/ITA/events",
        params={
            "as_of": "2026-09-27T09:30:00+00:00",
            "known_cutoff": "2026-09-27T09:30:00+00:00",
        },
    )
    assert early.status_code == 200
    early_payload = early.json()
    assert early_payload["events"][0]["num_articles"] == 2
    assert (
        early_payload["event_semantics"]
        == "provider_derived_observations_not_promoted_country_facts"
    )

    later = client.get(
        "/world/countries/ITA/events",
        params={
            "as_of": "2026-09-27T10:30:00+00:00",
            "known_cutoff": "2026-09-27T10:30:00+00:00",
        },
    )
    assert later.status_code == 200
    assert later.json()["events"][0]["num_articles"] == 7


def test_world_country_profile_endpoint_uses_local_evidence(tmp_path) -> None:
    orchestrator = DummyOrchestrator()
    repository = SimulationRepository(
        tmp_path / "sim-profile.duckdb",
        export_dir=tmp_path / "exports-profile",
    )
    service = SimulationService(repository=repository)
    world = WorldKnowledgeStore(tmp_path / "world-profile.duckdb")
    updater = AutonomousKnowledgeUpdater(world)

    when = datetime.fromisoformat("2026-09-27T08:00:00+00:00")
    evidence = SourceEvidence(
        source="official",
        source_ref="https://official.example/government-form",
        published_at=when,
        retrieved_at=when,
        independent_group="official_institution",
        authoritative=True,
    )
    updater.process(
        KnowledgeUpdateCandidate(
            entity_id="country:ITA",
            field="government_form",
            value="Test parliamentary republic",
            valid_from=when,
            detected_at=when,
            domain=KnowledgeDomain.POLITICS,
            kind=KnowledgeUpdateKind.FACT,
            confidence=0.98,
            evidence=(evidence,),
        )
    )

    news = SharedNewsStore(tmp_path / "shared-news-profile.duckdb")
    news.upsert_articles(
        provider="google_news_rss",
        articles=[
            {
                "title": "Italy political update - Example",
                "url": "https://news.google.com/rss/articles/profile",
                "seendate": "2026-09-27T08:10:00+00:00",
                "publisher": "Example",
                "domain": "example.com",
                "language": "en",
            }
        ],
        retrieved_at=datetime.fromisoformat("2026-09-27T08:15:00+00:00"),
        snapshot_checksum="profile-news",
        feed_id="world-politics-en",
    )

    app = create_app(
        orchestrator=orchestrator,
        simulation_service=service,
        world_knowledge_store=world,
        shared_news_store=news,
    )
    client = TestClient(app)

    response = client.get(
        "/world/countries/ita/profile",
        params={
            "as_of": "2026-09-27T09:00:00+00:00",
            "known_cutoff": "2026-09-27T09:00:00+00:00",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["identity"]["name"] == "Italy"
    government_form = next(
        item
        for item in payload["political_system"]["fields"]
        if item["key"] == "government_form"
    )
    assert government_form["value"] == "Test parliamentary republic"
    assert government_form["evidence"][0]["authoritative"] is True
    assert payload["recent_news"]["articles"][0]["title"].startswith("Italy")
    assert payload["semantic_contract"]["unknown_fields_remain_unknown"] is True
