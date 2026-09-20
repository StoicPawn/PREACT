from datetime import datetime, timezone
import io
import zipfile

from preact.data_hub.gateway import ProviderResponse
from preact.history.connectors.cow import COWStateSystemConnector
from preact.history.connectors.ucdp import UCDPConnector
from preact.history.connectors.unhcr import UNHCRConnector


UTC = timezone.utc


class FakeGateway:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get_json(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        return ProviderResponse(
            source_id=kwargs["source_id"],
            operation=kwargs["operation"],
            payload=payload,
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            cached=False,
            request_fingerprint="x",
            snapshot_checksum="abc",
        )


def test_ucdp_requires_explicit_version_and_token() -> None:
    gateway = FakeGateway([
        {"TotalPages": 1, "TotalCount": 1, "Result": [{"id": 7}]}
    ])
    rows = UCDPConnector(gateway, token="secret").fetch_all(
        resource="gedevents",
        version="26.1",
    )
    assert rows == [{"id": 7}]
    assert gateway.calls[0]["headers"]["x-ucdp-access-token"] == "secret"
    assert "26.1" in gateway.calls[0]["operation"]


def test_unhcr_paginates_items() -> None:
    gateway = FakeGateway([
        {"maxPages": 2, "items": [{"year": 2020}]},
        {"maxPages": 2, "items": [{"year": 2021}]},
    ])
    rows = UNHCRConnector(gateway).fetch_all(
        year_from=2020,
        year_to=2021,
        include_all_origins=True,
        include_all_asylum=True,
    )
    assert [row["year"] for row in rows] == [2020, 2021]
    assert len(gateway.calls) == 2


def test_cow_state_rows_create_temporal_entities() -> None:
    csv_text = (
        "StateAbb,CCode,StateNme,StYear,StMonth,StDay,EndYear,EndMonth,EndDay,Version\n"
        "XYZ,999,Example State,1900,1,2,1905,3,4,2024\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("states2024.csv", csv_text)
    rows = COWStateSystemConnector.parse_rows(buffer.getvalue())
    entities = COWStateSystemConnector.to_entities(rows)
    assert entities[0].name == "Example State"
    assert entities[0].valid_from.date().isoformat() == "1900-01-02"
    assert entities[0].valid_to.date().isoformat() == "1905-03-05"
    assert any(code.namespace == "cow_ccode" and code.value == "999" for code in entities[0].codes)
