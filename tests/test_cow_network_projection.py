from datetime import datetime, timezone

from preact.projections.cow_network import (
    cow_alliance_relations,cow_contiguity_relations,cow_mid_relations,cow_nmc_records
)

UTC=timezone.utc
KNOWN=datetime(2026,1,1,tzinfo=UTC)

def test_cow_graph_projections():
    alliances=cow_alliance_relations(
        [{"ccode1":"2","ccode2":"20","year":"1950","defense":"1"}],
        known_at=KNOWN,retrieved_at=KNOWN
    )
    assert alliances[0].relation_type=="formal_alliance"
    assert alliances[0].subject_entity_id=="cow_ccode:2"

    cont=cow_contiguity_relations(
        [{"state1no":"2","state2no":"20","year":"1950","conttype":"1"}],
        known_at=KNOWN,retrieved_at=KNOWN
    )
    assert cont[0].relation_type=="direct_contiguity"

    mids=cow_mid_relations(
        [{"ccode1":"2","ccode2":"20","strtyr":"1951","endyear":"1952","dispnum":"1"}],
        known_at=KNOWN,retrieved_at=KNOWN
    )
    assert mids[0].valid_to.year==1953

def test_cow_nmc_missing_values_are_not_features():
    records=cow_nmc_records(
        [{"ccode":"325","year":"2000","cinc":"0.02","milex":"-9","tpop":"57000"}],
        known_at=KNOWN,retrieved_at=KNOWN
    )
    variables={r.variable for r in records}
    assert "cow_nmc:cinc" in variables
    assert "cow_nmc:tpop" in variables
    assert "cow_nmc:milex" not in variables
