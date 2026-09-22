from datetime import datetime, timezone

import pytest

from preact.feature_store.temporal import entity_feature_snapshot, entity_feature_snapshot_with_lineage
from preact.history.schema import KnowledgeMode


UTC = timezone.utc
CUTOFF = datetime(2020, 1, 15, tzinfo=UTC)


class StubWarehouse:
    def __init__(self, rows):
        self.rows = rows

    def latest_observations_as_of(self, **kwargs):
        return self.rows


def _row(*, valid_from: datetime, known_at: datetime, **overrides) -> dict:
    row = {
        "record_id": "record-1",
        "entity_id": "iso3:AAA",
        "variable": "stress",
        "value_json": "0.7",
        "valid_from": valid_from,
        "known_at": known_at,
        "source": "example-source",
        "source_ref": "snapshot-2020-01",
        "dataset_version": "v1",
        "retrieved_at": datetime(2020, 1, 12, tzinfo=UTC),
    }
    row.update(overrides)
    return row


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


def test_feature_snapshot_allows_future_knowledge_only_in_retrospective_mode() -> None:
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
        knowledge_mode=KnowledgeMode.RETROSPECTIVE,
    )
    assert features == {"stress": 0.7}


def test_feature_lineage_identifies_exact_source_vintage_stably() -> None:
    warehouse = StubWarehouse([
        _row(
            valid_from=datetime(2020, 1, 1, tzinfo=UTC),
            known_at=datetime(2020, 1, 10, tzinfo=UTC),
        )
    ])

    features, lineage = entity_feature_snapshot_with_lineage(
        warehouse, entity_id="iso3:AAA", cutoff=CUTOFF
    )
    _, repeated = entity_feature_snapshot_with_lineage(
        warehouse, entity_id="iso3:AAA", cutoff=CUTOFF
    )

    assert features == {"stress": 0.7}
    assert lineage["stress"]["record_id"] == "record-1"
    assert lineage["stress"]["dataset_version"] == "v1"
    assert lineage["stress"]["fingerprint"] == repeated["stress"]["fingerprint"]
    assert len(lineage["stress"]["fingerprint"]) == 64


def test_feature_lineage_changes_when_source_vintage_changes() -> None:
    base = _row(
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        known_at=datetime(2020, 1, 10, tzinfo=UTC),
    )
    revised = dict(base, record_id="record-2", dataset_version="v2")

    _, first = entity_feature_snapshot_with_lineage(
        StubWarehouse([base]), entity_id="iso3:AAA", cutoff=CUTOFF
    )
    _, second = entity_feature_snapshot_with_lineage(
        StubWarehouse([revised]), entity_id="iso3:AAA", cutoff=CUTOFF
    )

    assert first["stress"]["fingerprint"] != second["stress"]["fingerprint"]


def test_feature_lineage_fails_closed_when_provenance_is_incomplete() -> None:
    row = _row(
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        known_at=datetime(2020, 1, 10, tzinfo=UTC),
    )
    row.pop("source_ref")

    with pytest.raises(ValueError, match="feature lineage is incomplete: missing source_ref"):
        entity_feature_snapshot_with_lineage(
            StubWarehouse([row]), entity_id="iso3:AAA", cutoff=CUTOFF
        )
