from datetime import datetime, timezone

from preact.history.schema import EvidenceClass, Provenance, TemporalRecord
from preact.history.warehouse import HistoricalWarehouse


UTC = timezone.utc


def test_warehouse_as_of_respects_bitemporal_cutoff(tmp_path) -> None:
    warehouse = HistoricalWarehouse(tmp_path / "history.duckdb")
    old = TemporalRecord(
        record_id="old",
        entity_id="country:x",
        variable="gdp",
        value=10,
        valid_from=datetime(2000, 1, 1, tzinfo=UTC),
        known_at=datetime(2001, 1, 1, tzinfo=UTC),
        provenance=Provenance(
            source="test",
            source_ref="old",
            retrieved_at=datetime(2001, 1, 1, tzinfo=UTC),
        ),
    )
    revised = TemporalRecord(
        record_id="revised",
        entity_id="country:x",
        variable="gdp",
        value=11,
        valid_from=datetime(2000, 1, 1, tzinfo=UTC),
        known_at=datetime(2005, 1, 1, tzinfo=UTC),
        provenance=Provenance(
            source="test",
            source_ref="revised",
            retrieved_at=datetime(2005, 1, 1, tzinfo=UTC),
        ),
        evidence_class=EvidenceClass.ESTIMATE,
    )
    assert warehouse.insert_records([old, revised]) == 2
    rows = warehouse.as_of(
        cutoff=datetime(2003, 1, 1, tzinfo=UTC),
        valid_at=datetime(2000, 6, 1, tzinfo=UTC),
        entity_id="country:x",
        variable="gdp",
    )
    assert [row["record_id"] for row in rows] == ["old"]
