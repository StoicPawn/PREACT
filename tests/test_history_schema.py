from datetime import datetime, timezone

from preact.history import EvidenceClass, HistoricalQuery, KnowledgeMode, Provenance, TemporalRecord

UTC = timezone.utc


def _dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _record(*, known_at: datetime, valid_from: datetime, valid_to=None) -> TemporalRecord:
    return TemporalRecord(
        record_id="r1",
        entity_id="country:example",
        variable="political_event",
        value=1,
        valid_from=valid_from,
        valid_to=valid_to,
        known_at=known_at,
        evidence_class=EvidenceClass.OBSERVATION,
        provenance=Provenance(source="test", source_ref="unit-test", retrieved_at=known_at),
    )


def test_future_knowledge_is_excluded_from_historical_replay() -> None:
    record = _record(known_at=_dt(1962, 10, 25), valid_from=_dt(1962, 10, 20))
    query = HistoricalQuery(knowledge_cutoff=_dt(1962, 10, 22), valid_at=_dt(1962, 10, 20))
    assert query.filter([record]) == []


def test_record_known_before_cutoff_is_available() -> None:
    record = _record(known_at=_dt(1962, 10, 20), valid_from=_dt(1962, 10, 20))
    query = HistoricalQuery(knowledge_cutoff=_dt(1962, 10, 22), valid_at=_dt(1962, 10, 20))
    assert query.filter([record]) == [record]


def test_valid_time_is_independent_from_knowledge_time() -> None:
    record = _record(
        known_at=_dt(2000, 1, 1),
        valid_from=_dt(1900, 1, 1),
        valid_to=_dt(1910, 1, 1),
    )
    query_inside = HistoricalQuery(knowledge_cutoff=_dt(2001, 1, 1), valid_at=_dt(1905, 1, 1))
    query_outside = HistoricalQuery(knowledge_cutoff=_dt(2001, 1, 1), valid_at=_dt(1915, 1, 1))
    assert query_inside.filter([record]) == [record]
    assert query_outside.filter([record]) == []



def test_retrospective_query_allows_later_coding_of_past() -> None:
    record = _record(
        known_at=_dt(2026, 1, 1),
        valid_from=_dt(1960, 1, 1),
        valid_to=_dt(1961, 1, 1),
    )
    strict = HistoricalQuery(
        knowledge_cutoff=_dt(1960, 6, 1),
        valid_at=_dt(1960, 6, 1),
    )
    retrospective = HistoricalQuery(
        knowledge_cutoff=_dt(1960, 6, 1),
        valid_at=_dt(1960, 6, 1),
        knowledge_mode=KnowledgeMode.RETROSPECTIVE,
    )
    assert strict.filter([record]) == []
    assert retrospective.filter([record]) == [record]
