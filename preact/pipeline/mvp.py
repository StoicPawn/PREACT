"""Synthetic MVP pipeline to produce model-ready predictions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import FeatureConfig, ModelConfig, PREACTConfig
from ..models.predictor import ModelOutput, PredictiveEngine

DEFAULT_COUNTRIES = (
    "Nigeria",
    "Ethiopia",
    "South Sudan",
    "Somalia",
    "Democratic Republic of the Congo",
    "Burkina Faso",
)


@dataclass
class PipelineArtifacts:
    """Container for the main artefacts produced by the MVP pipeline."""

    predictions: Dict[str, pd.Series]
    outcomes: pd.Series
    evidence: pd.DataFrame
    model_outputs: list[ModelOutput]


def _multi_index(dates: pd.DatetimeIndex, countries: Sequence[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product((dates, countries), names=["date", "country"])


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_")


def _build_feature_frame(
    values: Mapping[str, np.ndarray],
    dates: pd.DatetimeIndex,
    countries: Sequence[str],
) -> pd.DataFrame:
    index = _multi_index(dates, countries)
    flattened: list[float] = []
    for day in range(len(dates)):
        for country in countries:
            flattened.append(float(values[country][day]))
    frame = pd.DataFrame({"value": flattened}, index=index)
    return frame


def _zscore(series: np.ndarray) -> np.ndarray:
    std = float(series.std())
    if std < 1e-6:
        return np.zeros_like(series)
    return (series - float(series.mean())) / std


def generate_synthetic_training_data(
    feature_configs: Iterable[FeatureConfig],
    model_configs: Iterable[ModelConfig],
    countries: Sequence[str] = DEFAULT_COUNTRIES,
    days: int = 120,
    seed: int = 7,
) -> tuple[Dict[str, pd.DataFrame], pd.Series]:
    """Create deterministic synthetic features and targets for MVP usage."""

    rng = np.random.default_rng(seed)
    end = datetime.utcnow().date()
    start = end - timedelta(days=days - 1)
    dates = pd.date_range(start=start, end=end, freq="D")

    per_country: Dict[str, Dict[str, np.ndarray]] = {country: {} for country in countries}
    features: Dict[str, pd.DataFrame] = {}

    base_intensity = {country: rng.uniform(0.2, 0.8) for country in countries}

    for cfg in feature_configs:
        domain = cfg.name.split("_", 1)[0]
        feature_key = f"{domain}__{cfg.name}"
        values: Dict[str, np.ndarray] = {}

        for country in countries:
            trend = rng.normal(0, 0.02, len(dates)).cumsum()
            seasonal = 0.1 * np.sin(np.linspace(0, 3 * np.pi, len(dates)))
            base = base_intensity[country]

            if domain == "events":
                lam = np.clip(base * 6 + trend + seasonal + rng.normal(0, 0.3, len(dates)), 0.2, None)
                series = rng.poisson(lam=lam) + rng.binomial(1, 0.1, len(dates))
            elif domain == "economic":
                drift = rng.normal(0.0, 0.05)
                series = base + drift + trend + seasonal + rng.normal(0, 0.05, len(dates))
            elif domain == "humanitarian":
                series = (
                    base
                    + 0.3 * trend
                    + 0.2 * seasonal
                    + rng.normal(0, 0.04, len(dates)).cumsum()
                )
            else:
                series = base + trend + seasonal

            # Ensure positive values for count-like series
            series = np.clip(series, 0.0, None)
            values[country] = series.astype(float)
            per_country[country][feature_key] = values[country]

        features[feature_key] = _build_feature_frame(values, dates, countries)

    feature_names = {name for model in model_configs for name in model.features}
    missing = feature_names - features.keys()
    if missing:
        raise KeyError(f"Synthetic generator missing features: {sorted(missing)}")

    index = _multi_index(dates, countries)
    target_values = np.zeros(len(index))

    for day in range(len(dates)):
        for country_idx, country in enumerate(countries):
            offset = day * len(countries) + country_idx
            events = per_country[country].get("events__events_coup_attempts")
            violence = per_country[country].get("events__events_violence_against_civilians")
            economic = per_country[country].get("economic__economic_pressure")
            displacement = per_country[country].get(
                "humanitarian__humanitarian_displacement_pressure"
            )
            constraints = per_country[country].get(
                "humanitarian__humanitarian_operational_constraints"
            )

            stacked = np.stack(
                [
                    _zscore(events)[day] if events is not None else 0.0,
                    _zscore(violence)[day] if violence is not None else 0.0,
                    _zscore(economic)[day] if economic is not None else 0.0,
                    _zscore(displacement)[day] if displacement is not None else 0.0,
                    _zscore(constraints)[day] if constraints is not None else 0.0,
                ]
            )
            score = 0.6 * stacked[0] + 0.5 * stacked[1] + 0.3 * stacked[3] + 0.2 * stacked[2] + 0.2 * stacked[4]
            score += rng.normal(0, 0.4)
            probability = 1.0 / (1.0 + np.exp(-score))
            target_values[offset] = rng.binomial(1, probability)

    target = pd.Series(target_values, index=index, name="event")
    return features, target


def combine_model_outputs(outputs: Iterable[ModelOutput]) -> Dict[str, pd.Series]:
    """Average multiple model outputs into per-country probability series."""

    stacked_frames: list[pd.DataFrame] = []
    for output in outputs:
        frame = output.probabilities.unstack("country")
        stacked_frames.append(frame)

    if not stacked_frames:
        return {}

    ensemble = sum(stacked_frames) / len(stacked_frames)
    return {country: ensemble[country].rename("probability") for country in ensemble.columns}


def build_evidence(predictions: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Generate deterministic evidence records from the predictions."""

    records = []
    for country, series in predictions.items():
        if series.empty:
            continue
        latest = float(series.iloc[-1])
        reference = float(series.iloc[-7]) if len(series) > 7 else float(series.iloc[0])
        change = latest - reference
        odds = latest / max(1e-3, 1 - latest)
        records.append(
            {
                "country": country,
                "signal_strength": round(float(np.clip(abs(change) * 4, 0.05, 0.95)), 2),
                "posterior_odds": round(float(odds), 2),
                "updated": series.index[-1].strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame.from_records(records)


def save_predictions(predictions: Mapping[str, pd.Series], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for country, series in predictions.items():
        frame = series.to_frame(name="probability")
        frame["country"] = country
        frame.to_parquet(output_dir / f"{_slugify(country)}.parquet")


def save_outcomes(target: pd.Series, output_dir: Path) -> pd.Series:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_date = target.index.get_level_values("date").max()
    latest = target.xs(latest_date, level="date")
    latest.name = "outcome"
    latest.to_frame().to_parquet(output_dir / "outcomes.parquet")
    return latest


def save_evidence(evidence: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence.to_json(output_dir / "bayesian_evidence.json", orient="records", indent=2)


def run_mvp_prediction_pipeline(
    output_dir: Path,
    config: PREACTConfig,
    countries: Sequence[str] = DEFAULT_COUNTRIES,
    days: int = 120,
    seed: int = 7,
) -> PipelineArtifacts:
    """Train models on synthetic data and export country-level predictions."""

    features, target = generate_synthetic_training_data(
        config.features, config.models, countries=countries, days=days, seed=seed
    )

    engine = PredictiveEngine(config.models)
    models_dir = config.storage.models_dir
    if engine.has_persisted_models(models_dir):
        models = engine.load_models(models_dir)
    else:
        models = engine.train(features, target)
        engine.save_models(models, models_dir)
    outputs = engine.predict(models, features, target)

    predictions = combine_model_outputs(outputs)
    save_predictions(predictions, output_dir / "predictions")

    outcomes = save_outcomes(target, output_dir)
    evidence = build_evidence(predictions)
    save_evidence(evidence, output_dir / "predictions")

    return PipelineArtifacts(
        predictions=predictions,
        outcomes=outcomes,
        evidence=evidence,
        model_outputs=list(outputs),
    )


__all__ = [
    "PipelineArtifacts",
    "build_evidence",
    "combine_model_outputs",
    "generate_synthetic_training_data",
    "run_mvp_prediction_pipeline",
    "save_evidence",
    "save_outcomes",
    "save_predictions",
]
