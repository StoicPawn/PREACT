"""Bayesian explainability utilities for PREACT."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class BayesianEvidence:
    """Evidence summary for a single feature bucket."""

    feature: str
    bucket: str
    posterior: float
    likelihood_ratio: float
    evidence_strength: str
    samples: int


class BayesianExplainer:
    """Naive Bayesian evidence aggregation for calibrated model outputs."""

    def __init__(self, prior: float = 0.05, smoothing: float = 1.0, bins: int = 4) -> None:
        if not 0 < prior < 1:
            raise ValueError("Prior must be within (0, 1)")
        if bins < 2:
            raise ValueError("At least two bins are required for Bayesian explanations")
        self.prior = prior
        self.smoothing = smoothing
        self.bins = bins

    def explain(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Return Bayesian evidence for the latest observation per feature."""

        if features.empty:
            return pd.DataFrame(columns=BayesianEvidence.__dataclass_fields__.keys())

        aligned_target = target.reindex(features.index).fillna(0).astype(int)
        prior_odds = self.prior / (1 - self.prior)
        evidences: List[Dict[str, object]] = []

        latest_index = features.index[-1]
        for column in features.columns:
            raw = pd.to_numeric(features[column], errors="coerce")
            series = raw.dropna()
            if series.empty or series.nunique() < 2:
                continue
            try:
                categories, bin_edges = pd.qcut(
                    series,
                    q=min(self.bins, series.nunique()),
                    labels=False,
                    retbins=True,
                    duplicates="drop",
                )
            except ValueError:
                continue
            bin_assignment = pd.Series(categories, index=series.index)
            if latest_index not in bin_assignment.index:
                # Attempt to use the closest prior observation
                available = bin_assignment.index[bin_assignment.index <= latest_index]
                available = available.sort_values()
                if len(available) == 0:
                    continue
                latest_bin = int(bin_assignment.loc[available[-1]])
            else:
                latest_bin = int(bin_assignment.loc[latest_index])

            df = pd.DataFrame(
                {
                    "bin": bin_assignment,
                    "event": aligned_target.reindex(bin_assignment.index).fillna(0).astype(int),
                }
            )
            counts = df.groupby("bin")["event"].agg(["sum", "count"])
            if latest_bin not in counts.index:
                continue
            summary = counts.loc[latest_bin]
            event_count = float(summary["sum"])
            total = float(summary["count"])
            posterior = (event_count + self.smoothing * self.prior) / (
                total + self.smoothing
            )
            posterior = float(np.clip(posterior, 1e-6, 1 - 1e-6))
            odds = posterior / (1 - posterior)
            likelihood_ratio = float(odds / prior_odds)
            evidence_strength = self._describe_strength(likelihood_ratio)
            bucket_label = self._format_bucket(bin_edges, latest_bin)

            evidences.append(
                {
                    "feature": column,
                    "bucket": bucket_label,
                    "posterior": posterior,
                    "likelihood_ratio": likelihood_ratio,
                    "evidence_strength": evidence_strength,
                    "samples": int(total),
                }
            )

        if not evidences:
            return pd.DataFrame(columns=BayesianEvidence.__dataclass_fields__.keys())

        frame = pd.DataFrame(evidences)
        return frame.sort_values("posterior", ascending=False).reset_index(drop=True)

    def _format_bucket(self, edges: np.ndarray, index: int) -> str:
        lower = edges[index]
        upper = edges[min(index + 1, len(edges) - 1)]
        if np.isinf(lower) and np.isinf(upper):
            return "all values"
        if np.isinf(lower):
            return f"<= {upper:.2f}"
        if np.isinf(upper):
            return f"> {lower:.2f}"
        return f"[{lower:.2f}, {upper:.2f})"

    def _describe_strength(self, likelihood_ratio: float) -> str:
        if likelihood_ratio >= 10:
            return "very strong increase"
        if likelihood_ratio >= 3:
            return "strong increase"
        if likelihood_ratio >= 1.5:
            return "moderate increase"
        if likelihood_ratio >= 1.1:
            return "slight increase"
        if likelihood_ratio <= 0.1:
            return "very strong decrease"
        if likelihood_ratio <= 0.33:
            return "strong decrease"
        if likelihood_ratio <= 0.66:
            return "moderate decrease"
        if likelihood_ratio <= 0.9:
            return "slight decrease"
        return "neutral"
