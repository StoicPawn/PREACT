"""Evaluation utilities for PREACT."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def compute_brier_score(probabilities: pd.Series, outcomes: pd.Series) -> float:
    aligned_outcomes = outcomes.reindex(probabilities.index).fillna(0)
    return float(((probabilities - aligned_outcomes) ** 2).mean())


def compute_precision_recall(
    probabilities: pd.Series, outcomes: pd.Series, threshold: float
) -> Dict[str, float]:
    aligned_outcomes = outcomes.reindex(probabilities.index).fillna(0)
    predictions = (probabilities >= threshold).astype(int)
    tp = float(((predictions == 1) & (aligned_outcomes == 1)).sum())
    fp = float(((predictions == 1) & (aligned_outcomes == 0)).sum())
    fn = float(((predictions == 0) & (aligned_outcomes == 1)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {"precision": precision, "recall": recall}


def summary_table(
    probabilities: Dict[str, pd.Series],
    outcomes: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for name, probs in probabilities.items():
        row = {
            "model": name,
            "brier": compute_brier_score(probs, outcomes),
        }
        row.update(compute_precision_recall(probs, outcomes, threshold))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")

