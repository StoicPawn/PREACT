from preact.history.wave1_runner import SourceRun, Wave1Runner


def test_manual_sources_are_explicit_not_synthetic() -> None:
    pending = {item.source_id: item for item in Wave1Runner.pending_manual_sources()}
    assert pending["vdem"].status == "requires_registration"
    assert pending["sipri"].status == "manual_release"


def test_source_run_contract() -> None:
    item = SourceRun("x", "success", rows=10, snapshots=1)
    assert item.source_id == "x"
    assert item.rows == 10
    assert item.snapshots == 1
