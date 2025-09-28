"""Counterfactual scenario simulations for PREACT."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import pandas as pd


@dataclass
class Intervention:
    """Representation of a policy intervention to simulate."""

    name: str
    feature_adjustments: Dict[str, float]


def apply_intervention(
    baseline_features: Mapping[str, pd.DataFrame],
    intervention: Intervention,
) -> Dict[str, pd.DataFrame]:
    """Adjust features to simulate a counterfactual scenario."""

    adjusted: Dict[str, pd.DataFrame] = {}
    for name, df in baseline_features.items():
        delta = intervention.feature_adjustments.get(name, 0.0)
        adjusted[name] = df + delta
    return adjusted


def scenario_analysis(
    predictions: pd.Series,
    baseline_predictions: pd.Series,
) -> pd.DataFrame:
    """Compare predictions under an intervention to the baseline."""

    comparison = pd.DataFrame(
        {
            "baseline_probability": baseline_predictions,
            "scenario_probability": predictions,
        }
    )
    comparison["delta"] = comparison["scenario_probability"] - comparison["baseline_probability"]
    return comparison

