"""Regression tests for nested predictive-artifact feature lineage."""

from types import SimpleNamespace

import pandas as pd
import pytest

from preact.models.oos_lineage import benchmark_suite_feature_lineage


def _fingerprints() -> pd.Series:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2020-01-01", tz="UTC"), "A")],
        names=["date", "entity_id"],
    )
    return pd.Series(["a" * 64], index=index)


def _audit() -> dict[str, object]:
    return {
        "fold": 0,
        "fit_end": "2019-01-01",
        "fit_cutoff": "2019-02-01",
        "calibration_start": "2019-02-01",
        "calibration_end": "2019-03-01",
        "training_cutoff": "2019-04-01",
        "test_start": "2020-01-01",
        "test_end": "2020-01-01",
        "test_dates": 1,
        "test_date_values": ["2020-01-01"],
    }


def _suite(predictions: pd.DataFrame):
    fold = SimpleNamespace(audit_record=lambda: _audit())
    benchmark = SimpleNamespace(predictions=predictions)
    return SimpleNamespace(folds=[fold], models={"model": benchmark})


def test_nested_lineage_summary_is_auditable_and_compact():
    predictions = pd.DataFrame(
        {"fold": [0], "date": ["2020-01-01"], "entity_id": ["A"], "y_true": [0], "y_prob": [0.2]}
    )
    assert benchmark_suite_feature_lineage(_suite(predictions), _fingerprints()) == {
        "model": {"rows": 1, "unique_feature_snapshots": 1}
    }


def test_nested_lineage_summary_fails_closed_on_missing_vintage():
    predictions = pd.DataFrame(
        {"fold": [0], "date": ["2020-01-01"], "entity_id": ["B"], "y_true": [0], "y_prob": [0.2]}
    )
    with pytest.raises(ValueError, match="missing point-in-time feature provenance"):
        benchmark_suite_feature_lineage(_suite(predictions), _fingerprints())
