import pandas as pd

from preact.models.experiment_manifest import build_manifest, fingerprint_panel


def test_panel_fingerprint_changes_when_target_changes():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=3), ["a", "b"]],
        names=["date", "entity_id"],
    )
    x = pd.DataFrame({"x": range(6)}, index=idx)
    y1 = pd.Series([0, 0, 0, 1, 0, 1], index=idx)
    y2 = y1.copy()
    y2.iloc[0] = 1
    assert fingerprint_panel(x, y1) != fingerprint_panel(x, y2)

    manifest = build_manifest(
        x,
        y1,
        target_name="mid",
        horizon_days=365,
        knowledge_mode="retrospective",
    )
    assert manifest.rows == 6
    assert manifest.events == 2
    assert manifest.entities == 2
