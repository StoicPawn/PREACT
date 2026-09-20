from preact.history.licensing import DataUseContext, evaluate_source_use
from preact.history.source_catalog import SOURCE_BY_ID


def test_noncommercial_source_is_blocked_for_commercial_context() -> None:
    decision = evaluate_source_use(
        SOURCE_BY_ID["qog"],
        DataUseContext(commercial=True),
    )
    assert decision.allowed is False


def test_open_source_is_allowed_but_still_requires_metadata() -> None:
    decision = evaluate_source_use(
        SOURCE_BY_ID["world_bank"],
        DataUseContext(commercial=True, redistribute_raw=True),
    )
    assert decision.allowed is True


def test_mixed_rights_archive_is_not_assumed_redistributable() -> None:
    decision = evaluate_source_use(
        SOURCE_BY_ID["europeana_newspapers"],
        DataUseContext(redistribute_raw=True),
    )
    assert decision.allowed is False
