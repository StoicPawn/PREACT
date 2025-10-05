"""Neural network based predictive engine for PREACT."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import ModelConfig
from .predictor import ModelOutput


@dataclass
class NeuralTrainingSummary:
    """Container for diagnostics generated during training."""

    loss_curve: List[float]
    n_iter: int
    train_accuracy: float


class NeuralNetworkEngine:
    """Train and serve neural network models for PREACT predictions."""

    def __init__(self, configs: Iterable[ModelConfig], random_state: int = 42) -> None:
        self.configs = list(configs)
        self.random_state = random_state
        self.training_history: Dict[str, NeuralTrainingSummary] = {}

    def prepare_dataset(
        self, features: Mapping[str, pd.DataFrame], target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Align feature frames and targets into a single learning dataset."""

        feature_frames: List[pd.DataFrame] = []
        feature_names = sorted({name for config in self.configs for name in config.features})
        for feature_name in feature_names:
            if feature_name not in features:
                raise KeyError(f"Feature {feature_name} missing from feature store")
            frame = features[feature_name].copy()
            frame.columns = [f"{feature_name}__{col}" for col in frame.columns]
            feature_frames.append(frame)
        dataset = pd.concat(feature_frames, axis=1)
        dataset = dataset.ffill().fillna(0)
        aligned_target = target.reindex(dataset.index).fillna(0)
        return dataset, aligned_target

    def _build_model(self, config: ModelConfig) -> Pipeline:
        hyperparams = dict(config.hyperparameters)
        hidden_layer_sizes = hyperparams.pop("hidden_layer_sizes", (64, 32))
        if isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = tuple(int(h) for h in hidden_layer_sizes)
        random_state = int(hyperparams.pop("random_state", self.random_state))
        classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            **hyperparams,
        )
        return Pipeline([("scaler", StandardScaler()), ("mlp", classifier)])

    def train(
        self, features: Mapping[str, pd.DataFrame], target: pd.Series
    ) -> Dict[str, Pipeline]:
        """Train neural models for each configured horizon."""

        trained: Dict[str, Pipeline] = {}
        dataset, aligned_target = self.prepare_dataset(features, target)
        for config in self.configs:
            model = self._build_model(config)
            model.fit(dataset, aligned_target)
            mlp: MLPClassifier = model.named_steps["mlp"]
            loss_curve = list(getattr(mlp, "loss_curve_", []))
            summary = NeuralTrainingSummary(
                loss_curve=loss_curve,
                n_iter=int(getattr(mlp, "n_iter_", len(loss_curve))),
                train_accuracy=float(model.score(dataset, aligned_target)),
            )
            self.training_history[config.name] = summary
            trained[config.name] = model
        return trained

    def predict(
        self,
        models: Mapping[str, Pipeline],
        features: Mapping[str, pd.DataFrame],
        target: pd.Series,
    ) -> List[ModelOutput]:
        """Generate probabilities and diagnostics for configured models."""

        dataset, aligned_target = self.prepare_dataset(features, target)
        outputs: List[ModelOutput] = []
        for config in self.configs:
            model = models[config.name]
            probs = model.predict_proba(dataset)[:, 1]
            series = pd.Series(probs, index=dataset.index)
            diagnostics = self._diagnostics(series, aligned_target)
            outputs.append(
                ModelOutput(
                    name=config.name,
                    horizon_days=config.horizon_days,
                    probabilities=series,
                    diagnostics=diagnostics,
                )
            )
        return outputs

    def _diagnostics(self, probs: pd.Series, target: pd.Series) -> Dict[str, float]:
        preds = (probs >= 0.5).astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            return {
                "brier": float(brier_score_loss(target, probs)),
                "precision": float(precision_score(target, preds, zero_division=0)),
                "recall": float(recall_score(target, preds, zero_division=0)),
            }

