"""Predictive engine for PREACT."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from ..config import ModelConfig


@dataclass
class ModelOutput:
    """Container for model predictions and diagnostics."""

    name: str
    horizon_days: int
    probabilities: pd.Series
    diagnostics: Dict[str, float]


class PredictiveEngine:
    """Train and run predictive models for coup and atrocity risk."""

    def __init__(self, configs: Iterable[ModelConfig]) -> None:
        self.configs = list(configs)

    def prepare_dataset(
        self, features: Mapping[str, pd.DataFrame], target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        feature_frames = []
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

    def _calibrate(
        self, model: GradientBoostingClassifier, X: pd.DataFrame, y: pd.Series
    ) -> ClassifierMixin:
        class_counts = y.value_counts()
        min_count = int(class_counts.min()) if not class_counts.empty else 0
        if min_count < 3:
            model.fit(X, y)
            return model
        calibrator = CalibratedClassifierCV(model, cv=3, method="isotonic")
        calibrator.fit(X, y)
        return calibrator

    def train(
        self, features: Mapping[str, pd.DataFrame], target: pd.Series
    ) -> Dict[str, ClassifierMixin]:
        trained: Dict[str, ClassifierMixin] = {}
        dataset, aligned_target = self.prepare_dataset(features, target)
        for config in self.configs:
            base = GradientBoostingClassifier(**config.hyperparameters)
            calibrated = self._calibrate(base, dataset, aligned_target)
            trained[config.name] = calibrated
        return trained

    def predict(
        self,
        models: Mapping[str, ClassifierMixin],
        features: Mapping[str, pd.DataFrame],
        target: pd.Series,
    ) -> List[ModelOutput]:
        dataset, aligned_target = self.prepare_dataset(features, target)
        outputs: List[ModelOutput] = []
        for config in self.configs:
            model = models[config.name]
            probs = pd.Series(model.predict_proba(dataset)[:, 1], index=dataset.index)
            diagnostics = self._diagnostics(probs, aligned_target)
            outputs.append(
                ModelOutput(
                    name=config.name,
                    horizon_days=config.horizon_days,
                    probabilities=probs,
                    diagnostics=diagnostics,
                )
            )
        return outputs

    def _diagnostics(self, probs: pd.Series, target: pd.Series) -> Dict[str, float]:
        preds = (probs >= 0.5).astype(int)
        return {
            "brier": float(brier_score_loss(target, probs)),
            "precision": float(precision_score(target, preds, zero_division=0)),
            "recall": float(recall_score(target, preds, zero_division=0)),
        }


def rolling_backtest(
    engine: PredictiveEngine,
    features: Mapping[str, pd.DataFrame],
    target: pd.Series,
    n_splits: int = 5,
) -> Dict[str, List[Dict[str, float]]]:
    """Run a time-series cross validation for robustness checks."""

    dataset, aligned_target = engine.prepare_dataset(features, target)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    history: Dict[str, List[Dict[str, float]]] = {}
    for config in engine.configs:
        history[config.name] = []
        base = GradientBoostingClassifier(**config.hyperparameters)
        for train_idx, test_idx in splitter.split(dataset):
            model = engine._calibrate(base, dataset.iloc[train_idx], aligned_target.iloc[train_idx])
            probs = model.predict_proba(dataset.iloc[test_idx])[:, 1]
            diagnostics = engine._diagnostics(
                pd.Series(probs, index=dataset.index[test_idx]),
                aligned_target.iloc[test_idx],
            )
            history[config.name].append(diagnostics)
    return history

