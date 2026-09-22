from datetime import datetime, timezone

from preact.feature_store.temporal_dynamics import entity_temporal_dynamics_snapshot
from preact.history.schema import KnowledgeMode, Provenance, TemporalRecord
from preact.history.warehouse import HistoricalWarehouse

UTC = timezone.utc


def _record(rid, year, value, known_year):
    known = datetime(known_year, 1, 15, tzinfo=UTC)
    return TemporalRecord(
        record_id=rid,
        entity_id="x",
        variable="gdp",
        value=value,
        valid_from=datetime(year, 1, 1, tzinfo=UTC),
        known_at=known,
        provenance=Provenance(
            source="test",
            source_ref=rid,
            retrieved_at=known,
        ),
    )


def test_dynamics_use_only_vintages_known_at_cutoff(tmp_path):
    w = HistoricalWarehouse(tmp_path / "h.duckdb")
    w.insert_records(
        [
            _record("2018", 2018, 100.0, 2019),
            _record("2019", 2019, 105.0, 2020),
            _record("2020-late", 2020, 120.0, 2025),
        ]
    )
    strict = entity_temporal_dynamics_snapshot(
        w,
        entity_id="x",
        cutoff=datetime(2021, 1, 1, tzinfo=UTC),
        variables=["gdp"],
        knowledge_mode=KnowledgeMode.STRICT_AS_KNOWN,
    )
    retrospective = entity_temporal_dynamics_snapshot(
        w,
        entity_id="x",
        cutoff=datetime(2021, 1, 1, tzinfo=UTC),
        variables=["gdp"],
        knowledge_mode=KnowledgeMode.RETROSPECTIVE,
    )
    assert strict["dyn:last:gdp"] == 105.0
    assert strict["dyn:delta1:gdp"] == 5.0
    assert retrospective["dyn:last:gdp"] == 120.0
    assert retrospective["dyn:delta1:gdp"] == 15.0
