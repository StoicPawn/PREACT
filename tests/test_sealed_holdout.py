from pathlib import Path

import pandas as pd
import pytest

from preact.models.sealed_holdout import (
    create_holdout_seal,
    development_view,
    load_holdout_seal,
    save_holdout_seal_once,
    validate_holdout_commitment,
)


def _panel():
    dates = pd.date_range("2000-01-01", periods=30, freq="YS")
    index = pd.MultiIndex.from_product(
        [dates, ["iso3:AAA", "iso3:BBB"]],
        names=["date", "entity_id"],
    )
    features = pd.DataFrame(
        {
            "x": range(len(index)),
            "z": [float(i % 7) for i in range(len(index))],
        },
        index=index,
    )
    target = pd.Series(
        [int(i % 11 == 0) for i in range(len(index))],
        index=index,
        dtype="Int64",
    )
    lineage = pd.Series(
        [f"{i:064x}"[-64:] for i in range(len(index))],
        index=index,
        dtype="string",
        name="feature_snapshot_fingerprint",
    )
    return features, target, lineage


def test_sealed_holdout_never_enters_development_view(tmp_path):
    features, target, lineage = _panel()
    seal = create_holdout_seal(
        features,
        target,
        target_name="event",
        horizon_days=365,
        holdout_fraction=0.20,
        min_holdout_dates=5,
        min_development_dates=20,
    )
    stored = save_holdout_seal_once(tmp_path / "seal.json", seal)
    validate_holdout_commitment(features, target, stored)

    x_dev, y_dev, lineage_dev = development_view(
        features,
        target,
        lineage,
        stored,
    )

    assert x_dev.index.equals(y_dev.index)
    assert x_dev.index.equals(lineage_dev.index)
    assert pd.to_datetime(x_dev.index.get_level_values("date")).max() < pd.Timestamp(
        stored.holdout_start
    )
    assert stored.holdout_dates >= 5


def test_existing_seal_cannot_be_moved_or_replaced(tmp_path):
    features, target, _ = _panel()
    path = tmp_path / "seal.json"
    seal = create_holdout_seal(
        features,
        target,
        target_name="event",
        horizon_days=365,
        holdout_fraction=0.20,
    )
    save_holdout_seal_once(path, seal)

    other = create_holdout_seal(
        features,
        target,
        target_name="event",
        horizon_days=365,
        holdout_fraction=0.30,
    )
    with pytest.raises(RuntimeError, match="refusing to replace"):
        save_holdout_seal_once(path, other)

    assert load_holdout_seal(path).holdout_start == seal.holdout_start


def test_target_tampering_after_seal_fails_closed():
    features, target, _ = _panel()
    seal = create_holdout_seal(
        features,
        target,
        target_name="event",
        horizon_days=365,
    )
    tampered = target.copy()
    held = pd.to_datetime(tampered.index.get_level_values("date")) >= pd.Timestamp(
        seal.holdout_start
    )
    position = int(pd.Series(held).to_numpy().nonzero()[0][0])
    tampered.iloc[position] = 1 - int(tampered.iloc[position])

    with pytest.raises(RuntimeError, match="target commitment"):
        validate_holdout_commitment(features, tampered, seal)


def test_feature_research_can_evolve_without_resealing_target():
    features, target, _ = _panel()
    seal = create_holdout_seal(
        features,
        target,
        target_name="event",
        horizon_days=365,
    )
    evolved = features.assign(new_research_feature=features["x"] ** 2)

    # Adding a candidate feature is allowed; the hidden target boundary remains fixed.
    validate_holdout_commitment(evolved, target, seal)


def test_post_holdout_censored_tail_can_mature_without_changing_seal():
    features, target, _ = _panel()
    extra_date = pd.Timestamp("2030-01-01")
    extra_index = pd.MultiIndex.from_product(
        [[extra_date], ["iso3:AAA", "iso3:BBB"]],
        names=["date", "entity_id"],
    )
    extra_features = pd.DataFrame(
        {"x": [100, 101], "z": [1.0, 2.0]},
        index=extra_index,
    )
    features = pd.concat([features, extra_features]).sort_index()
    target = pd.concat(
        [
            target,
            pd.Series([pd.NA, pd.NA], index=extra_index, dtype="Int64"),
        ]
    ).sort_index()

    seal = create_holdout_seal(
        features,
        target,
        target_name="event",
        horizon_days=365,
    )

    matured = target.copy()
    matured.loc[extra_index[0]] = 1
    matured.loc[extra_index[1]] = 0

    assert pd.Timestamp(seal.holdout_end) < extra_date
    validate_holdout_commitment(features, matured, seal)
