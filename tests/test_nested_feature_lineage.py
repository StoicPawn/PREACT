"""Regression coverage for feature lineage on nested OOS benchmark artifacts."""

import pandas as pd
import pytest

from preact.models.reporting import attach_prediction_feature_provenance


def _fingerprints() -> pd.Series:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2020-01-01", tz="UTC"), "A"),
            (pd.Timestamp("2020-01-01", tz="UTC"), "B"),
            (pd.Timestamp("2021-01-01", tz="UTC"), "A"),
            (pd.Timestamp("2021-01-01", tz="UTC"), "B"),
        ],
        names=["date", "entity_id"],
    )
    return pd.Series(["a" * 64, "b" * 64, "c" * 64, "d" * 64], index=index)


def test_nested_stress_subset_keeps_exact_point_in_time_lineage():
    """A geographic-stress subset must resolve vintages from the original panel."""
    predictions = pd.DataFrame(
        {"fold": [0, 1], "date": ["2020-01-01", "2021-01-01"], "entity_id": ["B", "B"], "y_true": [0, 1], "y_prob": [0.2, 0.7]}
    )
    attached = attach_prediction_feature_provenance(predictions, _fingerprints())
    assert attached["feature_snapshot_fingerprint"].tolist() == ["b" * 64, "d" * 64]
    assert len(attached) == len(predictions)


def test_nested_artifact_fails_closed_when_stress_key_has_no_source_vintage():
    predictions = pd.DataFrame({"fold": [0], "date": ["2022-01-01"], "entity_id": ["B"], "y_true": [0], "y_prob": [0.1]})
    with pytest.raises(ValueError, match="missing point-in-time feature provenance"):
        attach_prediction_feature_provenance(predictions, _fingerprints())


def test_nested_artifact_cannot_smuggle_a_preexisting_fingerprint():
    predictions = pd.DataFrame({"fold": [0], "date": ["2020-01-01"], "entity_id": ["A"], "y_true": [0], "y_prob": [0.1], "feature_snapshot_fingerprint": ["f" * 64]})
    attached = attach_prediction_feature_provenance(predictions, _fingerprints())
    assert attached.loc[0, "feature_snapshot_fingerprint"] == "a" * 64


def test_nested_artifact_rejects_duplicate_oos_provenance_keys():
    predictions = pd.DataFrame({"fold": [0, 0], "date": ["2020-01-01", "2020-01-01"], "entity_id": ["A", "A"], "y_true": [0, 0], "y_prob": [0.1, 0.1]})
    with pytest.raises(ValueError, match="duplicate OOS provenance keys"):
        attach_prediction_feature_provenance(predictions, _fingerprints())


def test_nested_artifact_rejects_ambiguous_source_vintage():
    """A duplicated canonical panel key must never be resolved arbitrarily."""
    fingerprints = _fingerprints()
    duplicated = pd.concat([fingerprints, fingerprints.iloc[[0]]])
    predictions = pd.DataFrame({"fold": [0], "date": ["2020-01-01"], "entity_id": ["A"], "y_true": [0], "y_prob": [0.1]})
    with pytest.raises(ValueError, match="duplicate panel keys"):
        attach_prediction_feature_provenance(predictions, duplicated)


def test_nested_artifact_rejects_non_sha256_source_fingerprint():
    """Corrupt lineage values cannot enter stress/placebo/ablation artifacts."""
    fingerprints = _fingerprints().copy()
    fingerprints.iloc[0] = "not-a-sha256"
    predictions = pd.DataFrame({"fold": [0], "date": ["2020-01-01"], "entity_id": ["A"], "y_true": [0], "y_prob": [0.1]})
    with pytest.raises(ValueError, match="invalid feature snapshot fingerprint"):
        attach_prediction_feature_provenance(predictions, fingerprints)
