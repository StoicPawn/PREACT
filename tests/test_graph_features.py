from datetime import datetime, timezone

from preact.feature_store.graph import graph_feature_snapshot
from preact.feature_store.graph_targets import binary_relation_target
from preact.history.graph_store import HistoricalGraphStore
from preact.history.relations import HistoricalRelation

UTC=timezone.utc

def rel(rid,kind,start,known):
    return HistoricalRelation(
        relation_id=rid,
        relation_type=kind,
        subject_entity_id="cow_ccode:1",
        object_entity_id="cow_ccode:2",
        valid_from=start,
        valid_to=None,
        known_at=known,
        source="test",
        source_ref=rid,
        retrieved_at=known,
    )

def test_graph_features_do_not_see_future_known_relations(tmp_path):
    graph=HistoricalGraphStore(tmp_path/"g.duckdb")
    graph.insert([
        rel("a","formal_alliance",datetime(2000,1,1,tzinfo=UTC),datetime(2000,1,1,tzinfo=UTC)),
        rel("b","militarized_interstate_dispute",datetime(2001,1,1,tzinfo=UTC),datetime(2002,1,1,tzinfo=UTC)),
    ])
    f=graph_feature_snapshot(
        graph,entity_id="cow_ccode:1",cutoff=datetime(2001,6,1,tzinfo=UTC)
    )
    assert f["graph_active:formal_alliance"]==1.0
    assert "graph_active:militarized_interstate_dispute" not in f

def test_relation_target_uses_realized_future_event_only_as_label(tmp_path):
    graph=HistoricalGraphStore(tmp_path/"g.duckdb")
    graph.insert([
        rel("m","militarized_interstate_dispute",datetime(2001,2,1,tzinfo=UTC),datetime(2005,1,1,tzinfo=UTC))
    ])
    y=binary_relation_target(
        graph,
        entity_id="cow_ccode:1",
        cutoffs=[datetime(2001,1,15,tzinfo=UTC)],
        relation_type="militarized_interstate_dispute",
        horizon_days=30,
    )
    assert y.iloc[0]==1
