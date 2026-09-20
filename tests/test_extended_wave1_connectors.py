from datetime import datetime, timezone

from preact.data_hub.gateway import ProviderResponse
from preact.history.connectors.cshapes import CShapesConnector
from preact.history.connectors.un_population import UNPopulationConnector


UTC = timezone.utc


class FakeGateway:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get_json(self, **kwargs):
        self.calls.append(kwargs)
        return ProviderResponse(
            source_id=kwargs["source_id"],
            operation=kwargs["operation"],
            payload=self.payloads.pop(0),
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            cached=False,
            request_fingerprint="x",
            snapshot_checksum="abc",
        )


def test_cshapes_csv_parser() -> None:
    rows = CShapesConnector.parse_rows(b"GWCODE,CNTRY_NAME\n325,Italy\n")
    assert rows == [{"GWCODE": "325", "CNTRY_NAME": "Italy"}]


def test_un_population_uses_bearer_and_rate_limit() -> None:
    gateway = FakeGateway([[{"TimeLabel": "2020", "Value": 60.0}]])
    rows = UNPopulationConnector(gateway, token="secret").fetch_all(
        indicators="49",
        locations="380",
        start_year=2020,
        end_year=2020,
    )
    assert len(rows) == 1
    call = gateway.calls[0]
    assert call["headers"]["Authorization"] == "Bearer secret"
    assert call["minimum_interval_seconds"] >= 2.0
    assert call["params"]["pageSize"] == 100


def test_un_population_requires_token() -> None:
    connector = UNPopulationConnector(FakeGateway([]), token=None)
    connector.token = None
    try:
        connector.fetch_all(
            indicators="49",
            locations="380",
            start_year=2020,
            end_year=2020,
        )
    except RuntimeError as exc:
        assert "UN_POPULATION_API_TOKEN" in str(exc)
    else:
        raise AssertionError("missing token should fail explicitly")
