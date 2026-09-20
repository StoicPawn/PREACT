from datetime import datetime, timedelta, timezone

import pytest

from preact.history.replay import (
    ForecastObservation,
    HistoricalReplayEngine,
    ReplaySpec,
    evaluate_binary_forecasts,
)
from preact.history.schema import Provenance, TemporalRecord


UTC = timezone.utc


def dt(day: int) -> datetime:
    return datetime(2020, 1, day, tzinfo=UTC)


def record(record_id: str, known_day: int) -> TemporalRecord:
    return TemporalRecord(
        record_id=record_id,
        entity_id="country:x",
        variable="signal",
        value=known_day,
        valid_from=dt(1),
        known_at=dt(known_day),
        provenance=Provenance(
            source="test",
            source_ref=record_id,
            retrieved_at=dt(known_day),
        ),
    )


def test_replay_freeze_excludes_future_knowledge() -> None:
    engine = HistoricalReplayEngine()
    frozen = engine.freeze(
        [record("known", 2), record("future", 9)],
        ReplaySpec(cutoff=dt(5), horizon=timedelta(days=7)),
    )
    assert [item.record_id for item in frozen] == ["known"]


def test_leakage_audit_rejects_future_record() -> None:
    engine = HistoricalReplayEngine()
    with pytest.raises(ValueError, match="leakage"):
        engine.assert_no_future_knowledge([record("future", 9)], cutoff=dt(5))


def test_binary_replay_metrics() -> None:
    metrics = evaluate_binary_forecasts(
        [
            ForecastObservation("a", dt(1), timedelta(days=30), 0.8, 1),
            ForecastObservation("b", dt(1), timedelta(days=30), 0.2, 0),
        ]
    )
    assert metrics.rows == 2
    assert metrics.brier_score == pytest.approx(0.04)
    assert metrics.event_rate == pytest.approx(0.5)
    assert metrics.mean_probability == pytest.approx(0.5)
    assert metrics.calibration_gap == pytest.approx(0.0)
