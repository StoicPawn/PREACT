from preact.history.source_plan import SOURCE_PLAN

def test_source_plan_is_unique_and_ordered():
    ids=[x.source_id for x in SOURCE_PLAN]
    assert len(ids)==len(set(ids))
    assert min(x.wave for x in SOURCE_PLAN)==0
    assert any(x.source_id=="cow" and x.wave==1 for x in SOURCE_PLAN)
    assert any(x.source_id=="chronicling_america" and x.wave==3 for x in SOURCE_PLAN)
