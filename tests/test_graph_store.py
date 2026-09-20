from datetime import datetime, timezone

from preact.history.graph_store import HistoricalGraphStore
from preact.history.relations import HistoricalRelation

UTC=timezone.utc

def test_graph_store_is_bitemporal(tmp_path):
    store=HistoricalGraphStore(tmp_path/"g.duckdb")
    rel=HistoricalRelation(
        relation_id="a",
        relation_type="alliance",
        subject_entity_id="cow:1",
        object_entity_id="cow:2",
        valid_from=datetime(1950,1,1,tzinfo=UTC),
        valid_to=datetime(1960,1,1,tzinfo=UTC),
        known_at=datetime(2000,1,1,tzinfo=UTC),
        source="cow",
        source_ref="a",
        retrieved_at=datetime(2026,1,1,tzinfo=UTC),
    )
    assert store.insert([rel])==1
    assert store.as_of(
        cutoff=datetime(1990,1,1,tzinfo=UTC),
        valid_at=datetime(1955,1,1,tzinfo=UTC),
    )==[]
    assert len(store.as_of(
        cutoff=datetime(2001,1,1,tzinfo=UTC),
        valid_at=datetime(1955,1,1,tzinfo=UTC),
    ))==1
