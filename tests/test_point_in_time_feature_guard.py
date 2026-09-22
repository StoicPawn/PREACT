from datetime import datetime, timezone

import pytest

from preact.feature_store.temporal import entity_feature_snapshot
from preact.history.schema import KnowledgeMode


UTC = timezone.utc
CUTOFF = datetime(2020, 1, 15, tzinfo=UTC)


class StubWarehouse:
    def __init__(self, rows):
        self.rows = rows

    def latest_observations_as_of(self, **kwargs):
        return self.rows


def _row(*, valid_from: datetime, known_at: datetime) -> dict:
    return {
        "entity_id": "iso3:AAA",
        "variable": "stress",
        "value_json": "0.7",
        "valid_from": valid_from,
        "known_at": known_at,
    }


def test_feature_snapshot_rejects_future_valid_time() -> None:
    warehouse = StubWarehouse([
        _row(
            valid_from=datetime(2020, 1, 16, tzinfo=UTC),
            known_at=datetime(2020, 1, 10, tzinfo=UTC),
        )
    ])

    with pytest.raises(ValueError, match="valid_from is after prediction cutoff"):
        entity_feature_snapshot(warehouse, entity_id="iso3:AAA", cutoff=CUTOFF)


def test_feature_snapshot_rejects_future_knowledge_in_strict_mode() -> None:
    warehouse = StubWarehouse([
        _row(
            valid_from=datetime(2020, 1, 1, tzinfo=UTC),
            known_at=datetime(2020, 1, 16, tzinfo=UTC),
        )
    ])

    with pytest.raises(ValueError, match="known_at is after prediction cutoff"):
        entity_feature_snapshot(warehouse, entity_id="iso3:AAA", cutoff=CUTOFF)


def test_feature_snapshot_allows_future_knowledge_only_in_hindsight_mode() -> None:
    warehouse = StubWarehouse([
        _row(
            valid_from=datetime(2020, 1, 1, tzinfo=UTC),
            known_at=datetime(2020, 1, 16, tzinfo=UTC),
        )
    ])

    features = entity_feature_snapshot(
        warehouse,
        entity_id="iso3:AAA",
        cutoff=CUTOFF,
        knowledge_mode=KnowledgeMode.HINDSIGHT,
    )
    assert features == {"stress": 0.7}
