import pandas as pd
import pytest

from preact.models.calibration_diagnostics import temporal_calibration_diagnostics


def test_temporal_calibration_diagnostics_exposes_fold_drift_hidden_by_global_gap():
    # Opposite fold biases cancel globally; a global calibration-gap metric alone
    # would therefore hide a material time-local calibration failure.
    frame = pd.DataFrame(
        {
            "fold": [0, 0, 0, 0, 1, 1, 1, 1],
            "actual": [0, 0, 1, 1, 0, 0, 1, 1],
            "probability": [0.2, 0.2, 0.8, 0.8, 0.0, 0.0, 0.6, 0.6],
        }
    )

    result = temporal_calibration_diagnostics(frame, bins=5)

    assert result.folds == 2
    assert result.weighted_gap == pytest.approx(-0.1)
    assert result.worst_abs_fold_gap == pytest.approx(0.2)
    assert result.fold_gap_std == pytest.approx(0.1)
    assert result.expected_calibration_error >= abs(result.weighted_gap)


def test_temporal_calibration_diagnostics_exposes_ece_hidden_by_fold_pooling():
    # The same probability bin is over-confident in one temporal fold and
    # under-confident in another. Pooling folds makes the bin look perfectly
    # calibrated, while each deployment period is materially miscalibrated.
    frame = pd.DataFrame(
        {
            "fold": [0] * 10 + [1] * 10,
            "actual": [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8,
            "probability": [0.5] * 20,
        }
    )

    result = temporal_calibration_diagnostics(frame, bins=10)

    assert result.expected_calibration_error == pytest.approx(0.0)
    assert result.worst_fold_expected_calibration_error == pytest.approx(0.3)


def test_temporal_calibration_diagnostics_rejects_invalid_probabilities():
    frame = pd.DataFrame({"fold": [0], "actual": [1], "probability": [1.1]})
    with pytest.raises(ValueError, match="probability"):
        temporal_calibration_diagnostics(frame)


def test_temporal_calibration_diagnostics_rejects_nonbinary_targets():
    frame = pd.DataFrame(
        {"fold": [0, 0, 0], "actual": [0, 1, 2], "probability": [0.1, 0.8, 0.9]}
    )
    with pytest.raises(ValueError, match="binary outcomes"):
        temporal_calibration_diagnostics(frame)


def test_temporal_calibration_diagnostics_rejects_missing_fold_labels():
    # groupby would otherwise silently drop the unlabeled row from fold drift
    # while global ECE/gap still include it, making governance evidence disagree.
    frame = pd.DataFrame(
        {"fold": [0, None, 1], "actual": [0, 1, 1], "probability": [0.1, 0.8, 0.7]}
    )
    with pytest.raises(ValueError, match="fold must be present"):
        temporal_calibration_diagnostics(frame)


def test_temporal_calibration_diagnostics_handles_empty_oos_frame():
    frame = pd.DataFrame(columns=["fold", "actual", "probability"])
    result = temporal_calibration_diagnostics(frame)
    assert result.folds == 0
    assert result.expected_calibration_error is None
    assert result.worst_fold_expected_calibration_error is None
