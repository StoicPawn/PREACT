from datetime import datetime, timezone

from preact.data_hub.gateway import ProviderResponse
from preact.history.connectors.world_bank import WorldBankIndicatorConnector


class FakeGateway:
    def get_json(self, **kwargs):
        return ProviderResponse(
            source_id="world_bank",
            operation="indicator",
            payload=[
                {"page": 1},
                [
                    {
                        "countryiso3code": "ITA",
                        "date": "2020",
                        "value": 100.5,
                        "indicator": {"id": "X"},
                    }
                ],
            ],
            retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            cached=False,
            request_fingerprint="x",
            snapshot_checksum="abc",
        )


def test_world_bank_current_snapshot_is_not_backdated_for_replay() -> None:
    rows = WorldBankIndicatorConnector(FakeGateway()).fetch(
        country="ITA",
        indicator="X",
        start_year=2020,
        end_year=2020,
    )
    assert rows[0].country_iso3 == "ITA"
    assert rows[0].year == 2020
    assert rows[0].value == 100.5
    assert rows[0].replay_eligible_before_retrieval is False
    assert rows[0].snapshot_checksum == "abc"
