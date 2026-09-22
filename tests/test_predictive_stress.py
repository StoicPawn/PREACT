import numpy as np
import pandas as pd
import pytest

from preact.models.reporting import validate_benchmark_suite_fold_audits
from preact.models.stress_tests import (
    deterministic_entity_holdout,
    deterministic_feature_degradation,
    run_unseen_entity_stress,
)


def test_entity_holdout_is_stable_and_disjoint():
    entities = [f"c{i}" for i in range(20)]
    a = deterministic_entity_holdout(entities, fraction=0.2, salt="x")
    b = deterministic_entity_holdout(entities, fraction=0.2, salt="x")
    assert a == b
    assert set(a.train_entities).isdisjoint(a.test_entities)
    assert set(a.train_entities) | set(a.test_entities) == set(entities)


def _stress_fixture():
    dates = pd.date_range("2000-01-01", periods=55, freq="30D")
    entities = [f"c{i}" for i in range(12)]
    idx = pd.MultiIndex.from_product([dates, entities], names=["date", "entity_id"])
    rng = np.random.default_rng(8)
    x = pd.DataFrame(
        {
            "x": rng.normal(size=len(idx)),
            "trend": np.repeat(np.linspace(-1, 1, len(dates)), len(entities)),
        },
        index=idx,
    )
    p = 1 / (1 + np.exp(-(-2 + 1.2 * x["trend"] + 0.4 * x["x"])))
    y = pd.Series(rng.binomial(1, p), index=idx)
    return x, y


def test_feature_degradation_is_reproducible_and_scoped():
    x, _ = _stress_fixture()
    original = x.copy(deep=True)
    a, audit_a = deterministic_feature_degradation(x, columns=["x"], fraction=0.3, salt="outage")
    b, audit_b = deterministic_feature_degradation(x, columns=["x"], fraction=0.3, salt="outage")

    pd.testing.assert_frame_equal(a, b)
    assert audit_a == audit_b
    assert audit_a.degraded_cells == int(a["x"].isna().sum())
    assert 0 < audit_a.degraded_cells < len(x)
    pd.testing.assert_series_equal(a["trend"], original["trend"])
    pd.testing.assert_frame_equal(x, original)


def test_feature_degradation_selection_does_not_depend_on_outcomes():
    x, y = _stress_fixture()
    degraded, _ = deterministic_feature_degradation(x, columns=["x"], fraction=0.25, salt="source-a")
    flipped = 1 - y
    degraded_after_target_change, _ = deterministic_feature_degradation(
        x, columns=["x"], fraction=0.25, salt="source-a"
    )
    assert flipped.index.equals(y.index)
    pd.testing.assert_frame_equal(degraded, degraded_after_target_change)


def test_feature_degradation_fails_closed_on_unknown_column_or_duplicate_index():
    x, _ = _stress_fixture()
    with pytest.raises(KeyError, match="unknown feature columns"):
        deterministic_feature_degradation(x, columns=["does_not_exist"])
    duplicated = pd.concat([x.iloc[:1], x])
    with pytest.raises(ValueError, match="index must be unique"):
        deterministic_feature_degradation(duplicated, columns=["x"])


def test_unseen_entity_stress_never_trains_on_holdout_entities():
    x, y = _stress_fixture()
    split, result = run_unseen_entity_stress(
        x,
        y,
        horizon_days=30,
        holdout_fraction=0.25,
        min_train_dates=20,
        calibration_dates=4,
        test_dates_per_fold=4,
        bootstrap_samples=20,
    )

    assert result.models
    audits = validate_benchmark_suite_fold_audits(result)
    assert len(audits) == len(result.folds)
    for model in result.models.values():
        if not model.predictions.empty:
            assert set(model.predictions["entity_id"]).issubset(
                set(split.test_entities)
            )


def test_nested_suite_audit_fails_closed_on_corrupted_oos_date():
    x, y = _stress_fixture()
    _, result = run_unseen_entity_stress(
        x,
        y,
        horizon_days=30,
        holdout_fraction=0.25,
        min_train_dates=20,
        calibration_dates=4,
        test_dates_per_fold=4,
        bootstrap_samples=20,
    )
    model = next(model for model in result.models.values() if not model.predictions.empty)
    model.predictions.loc[model.predictions.index[0], "date"] = pd.Timestamp("1900-01-01")

    with pytest.raises(ValueError, match="violates temporal fold audit"):
        validate_benchmark_suite_fold_audits(result)
