"""Sample data generators for the PREACT dashboard UI MVP."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd

COUNTRIES = [
    "Nigeria",
    "Ethiopia",
    "South Sudan",
    "Somalia",
    "Democratic Republic of the Congo",
    "Burkina Faso",
]


def _generate_time_index(days: int = 30) -> pd.DatetimeIndex:
    """Return a date range ending today for the provided number of days."""

    end = datetime.utcnow().date()
    start = end - timedelta(days=days - 1)
    return pd.date_range(start=start, end=end, freq="D", name="date")


def generate_sample_predictions() -> Dict[str, pd.Series]:
    """Return a dictionary of country -> probability series for demo purposes."""

    index = _generate_time_index()
    rng = np.random.default_rng(seed=42)
    predictions: Dict[str, pd.Series] = {}

    for country in COUNTRIES:
        base = rng.uniform(0.25, 0.65)
        noise = rng.normal(0, 0.04, len(index)).cumsum()
        series = np.clip(base + noise, 0.01, 0.99)
        predictions[country] = pd.Series(series, index=index, name="probability")

    return predictions


def generate_sample_outcomes(predictions: Dict[str, pd.Series]) -> pd.Series:
    """Create a binary outcome series that loosely follows the final probabilities."""

    latest = {country: series.iloc[-1] for country, series in predictions.items()}
    ordered = sorted(latest.items(), key=lambda item: item[1], reverse=True)
    triggered = {country for country, prob in ordered[:2] if prob > 0.6}

    outcomes = pd.Series({country: int(country in triggered) for country in latest.keys()}, name="outcome")
    return outcomes


def generate_sample_evidence(predictions: Dict[str, pd.Series]) -> pd.DataFrame:
    """Create a Bayesian evidence-like summary table for demonstration."""

    rng = np.random.default_rng(seed=24)
    records = []
    for country, series in predictions.items():
        latest = float(series.iloc[-1])
        signal_strength = rng.uniform(0.1, 0.9)
        posterior_odds = latest / (1 - latest)
        records.append(
            {
                "country": country,
                "signal_strength": round(signal_strength, 2),
                "posterior_odds": round(posterior_odds, 2),
                "updated": series.index[-1].strftime("%Y-%m-%d"),
            }
        )

    return pd.DataFrame.from_records(records)

