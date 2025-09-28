"""Command line interface to run the PREACT daily pipeline."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from preact.config import (
    DataSourceConfig,
    FeatureConfig,
    ModelConfig,
    PREACTConfig,
    StorageConfig,
)
from preact.data_ingestion.sources import build_sources, fetch_all
from preact.feature_store.builder import build_feature_store
from preact.models.predictor import PredictiveEngine
from preact.utils.io import save_config

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def default_config(root: Path) -> PREACTConfig:
    data_sources = [
        DataSourceConfig(name="GDELT", endpoint="https://api.gdeltproject.org/api/v2/events"),
        DataSourceConfig(name="ACLED", endpoint="https://api.acleddata.com/acled/read"),
        DataSourceConfig(name="Synthetic_Economic", endpoint="synthetic"),
    ]
    features = [
        FeatureConfig(
            name="events_coup_attempts",
            inputs=["GDELT"],
            aggregation="count",
            window_days=7,
        ),
        FeatureConfig(
            name="events_violence_against_civilians",
            inputs=["ACLED"],
            aggregation="count",
            window_days=7,
        ),
        FeatureConfig(
            name="economic_pressure",
            inputs=["Synthetic_Economic"],
            aggregation="mean",
            window_days=14,
        ),
    ]
    models = [
        ModelConfig(
            name="coup_risk_gb",
            target="coup",
            horizon_days=30,
            features=["events__events_coup_attempts", "economic__economic_pressure"],
            hyperparameters={"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
        ),
        ModelConfig(
            name="atrocity_risk_gb",
            target="atrocity",
            horizon_days=30,
            features=[
                "events__events_violence_against_civilians",
                "economic__economic_pressure",
            ],
            hyperparameters={"n_estimators": 120, "learning_rate": 0.04, "max_depth": 3},
        ),
    ]
    storage = StorageConfig(root_dir=root)
    return PREACTConfig(
        data_sources=data_sources,
        features=features,
        models=models,
        storage=storage,
    )


def run_pipeline(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    config = default_config(root)
    save_config(config, root / "config.json")

    sources = build_sources(config.data_sources)
    raw_data = fetch_all(sources, lookback_days=args.lookback_days)
    ingestion_frames = {name: result.data for name, result in raw_data.items() if result}

    feature_store = build_feature_store(ingestion_frames, config.features)

    # Placeholder targets using synthetic rare events
    index = next(iter(feature_store.tables.values())).index
    target_series = pd.Series(0, index=index, name="coup")
    target_series.iloc[-1] = 1  # simulate a recent event

    engine = PredictiveEngine(config.models)
    models = engine.train(feature_store.tables, target_series)
    outputs = engine.predict(models, feature_store.tables, target_series)

    predictions_dir = root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for output in outputs:
        df = pd.DataFrame({"probability": output.probabilities})
        df.to_parquet(predictions_dir / f"{output.name}.parquet")

    LOGGER.info("Pipeline completed with %d models", len(outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PREACT daily pipeline")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory for outputs")
    parser.add_argument("--lookback-days", type=int, default=30, help="Days of history to fetch")
    run_pipeline(parser.parse_args())

