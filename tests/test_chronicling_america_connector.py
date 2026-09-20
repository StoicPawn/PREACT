from datetime import datetime, timezone

from preact.data_hub.gateway import ProviderResponse
from preact.history.connectors.chronicling_america import ChroniclingAmericaConnector


class FakeGateway:
    def __init__(self):
        self.calls = []

    def get_json(self, **kwargs):
        self.calls.append(kwargs)
        return ProviderResponse(
            source_id="chronicling_america",
            operation="loc_search",
            payload={
                "results": [{"id": "https://www.loc.gov/resource/example/"}],
                "pagination": {"next": None},
            },
            retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            cached=False,
            request_fingerprint="x",
            snapshot_checksum="abc",
        )


def test_chronicling_search_uses_current_loc_api_contract() -> None:
    gateway = FakeGateway()
    rows = ChroniclingAmericaConnector(gateway).search_all(
        query="cuban missile",
        start_date="1962-10-01",
        end_date="1962-10-31",
        state="new york",
        front_pages_only=True,
    )
    assert len(rows) == 1
    call = gateway.calls[0]
    assert call["url"].endswith("/collections/chronicling-america/")
    assert call["params"]["fo"] == "json"
    assert call["params"]["dl"] == "page"
    assert call["params"]["qs"] == "cuban missile"
    assert call["params"]["front_pages_only"] == "true"
