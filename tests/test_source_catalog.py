from preact.history.source_catalog import SOURCE_BY_ID, SOURCES


def test_source_ids_are_unique() -> None:
    assert len(SOURCE_BY_ID) == len(SOURCES)


def test_core_sources_have_replay_policy_and_url() -> None:
    core = [source for source in SOURCES if source.priority == "core"]
    assert core
    assert all(source.replay_policy for source in core)
    assert all(source.url.startswith("https://") for source in core)


def test_catalog_contains_all_three_temporal_layers() -> None:
    ids = SOURCE_BY_ID
    assert "gdelt" in ids              # contemporary / news-derived
    assert "ucdp" in ids               # modern conflict history
    assert "cow" in ids                # 19th-century international system
    assert "seshat" in ids             # deep history
    assert "chronicling_america" in ids # primary historical press
