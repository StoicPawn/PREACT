from datetime import datetime, timezone

from preact.history.ingestion_state import IngestionStateStore, SourceRun


UTC = timezone.utc


def test_ingestion_state_returns_latest_per_source(tmp_path) -> None:
    store = IngestionStateStore(tmp_path / "state.sqlite3")
    store.record(
        SourceRun(
            source_id="world_bank",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            completed_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
            status="ready",
            rows_seen=10,
            rows_inserted=10,
            snapshots=1,
            details={"run": 1},
        )
    )
    store.record(
        SourceRun(
            source_id="world_bank",
            started_at=datetime(2026, 1, 2, tzinfo=UTC),
            completed_at=datetime(2026, 1, 2, 0, 1, tzinfo=UTC),
            status="ready",
            rows_seen=11,
            rows_inserted=1,
            snapshots=1,
            details={"run": 2},
        )
    )
    latest = store.latest("world_bank")
    assert latest["rows_seen"] == 11
    assert latest["details"]["run"] == 2
