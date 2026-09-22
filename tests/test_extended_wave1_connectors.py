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


def test_nmc_v7_nested_bundle_parser() -> None:
    import io
    import zipfile

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr(
            "NMC_v7_abridged.csv",
            "stateabb,ccode,year,milex,milper,irst,pec,tpop,upop,cinc\n"
            "ITA,325,2020,30000,170,24000,1000,60000,42000,0.02\n",
        )

    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("documentation/readme.txt", "docs")
        archive.writestr("NMC_v7_abridged.zip", inner.getvalue())

    from preact.history.connectors.cow_network import COWNetworkConnector

    rows = COWNetworkConnector.parse_nmc(outer.getvalue())
    assert rows[0]["ccode"] == "325"
    assert rows[0]["cinc"] == "0.02"
