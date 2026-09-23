"""PREACT-specific relational hazard mixture.

This estimator combines a conservative discrete-time hazard expert with a nonlinear
world-context expert. Its mixing weight is selected only on an internal chronological
validation tail separated from the inner fit sample by the same target-horizon purge used
by the outer benchmark. The final experts are then refit on the outer fit sample.

The architecture is deliberately sparse: world-context features summarize a potentially
huge international relation/news graph, so model complexity grows with feature families
rather than with every possible country pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .hazard import ComplementaryLogLogHazard


class PREACTRelationalHazardMixture(BaseEstimator, ClassifierMixin):
    """Nested-temporal mixture of local hazard and nonlinear world-context experts."""

    def __init__(
        self,
        *,
        horizon_days: int,
        context_prefix: str = "world_context:",
        gate_validation_dates: int = 5,
        weight_grid_size: int = 21,
        random_state: int = 42,
    ) -> None:
        self.horizon_days = int(horizon_days)
        self.context_prefix = str(context_prefix)
        self.gate_validation_dates = int(gate_validation_dates)
        self.weight_grid_size = int(weight_grid_size)
        self.random_state = int(random_state)

    def _validate_frame(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PREACTRelationalHazardMixture requires a pandas DataFrame")
        if not isinstance(X.index, pd.MultiIndex) or "date" not in X.index.names:
            raise TypeError("X must use a MultiIndex containing 'date'")
        if X.columns.duplicated().any():
            raise ValueError("X contains duplicate feature columns")
        return X

    def _local_columns(self, X: pd.DataFrame) -> list[str]:
        local = [
            str(column)
            for column in X.columns
            if not str(column).startswith(self.context_prefix)
        ]
        return local or [str(column) for column in X.columns]

    @staticmethod
    def _hazard_pipeline() -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", ComplementaryLogLogHazard(l2=1.0, max_iter=1000)),
            ]
        )

    def _context_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=200,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    @staticmethod
    def _balanced_weights(y: pd.Series) -> np.ndarray:
        values = np.asarray(y, dtype=int)
        positives = max(1, int(values.sum()))
        negatives = max(1, int(len(values) - values.sum()))
        positive_weight = negatives / positives
        return np.where(values == 1, positive_weight, 1.0).astype(float)

    def _fit_experts(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[Pipeline, Pipeline]:
        local = self._hazard_pipeline()
        context = self._context_pipeline()
        local.fit(X.loc[:, self.local_columns_], y)
        context.fit(
            X.loc[:, self.feature_columns_],
            y,
            model__sample_weight=self._balanced_weights(y),
        )
        return local, context

    def _internal_gate_split(
        self,
        X: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        raw_dates = pd.to_datetime(X.index.get_level_values("date"), errors="raise")
        dates = pd.DatetimeIndex(raw_dates.unique()).sort_values()
        if len(dates) < 8:
            return None

        n_validation = min(
            max(2, self.gate_validation_dates),
            max(2, len(dates) // 4),
        )
        validation_dates = dates[-n_validation:]
        validation_start = pd.Timestamp(validation_dates[0])
        training_cutoff = validation_start - pd.Timedelta(days=self.horizon_days)
        fit_dates = dates[dates < training_cutoff]
        if len(fit_dates) < 5:
            return None

        fit_mask = raw_dates.isin(fit_dates)
        validation_mask = raw_dates.isin(validation_dates)
        if not fit_mask.any() or not validation_mask.any():
            return None

        self.inner_training_cutoff_ = training_cutoff
        self.internal_validation_start_ = validation_start
        self.internal_validation_end_ = pd.Timestamp(validation_dates[-1])
        return np.asarray(fit_mask), np.asarray(validation_mask)

    def _select_weight(self, X: pd.DataFrame, y: pd.Series) -> float:
        split = self._internal_gate_split(X)
        if split is None:
            self.gate_status_ = "fixed_no_valid_inner_split"
            return 0.5

        fit_mask, validation_mask = split
        y_fit = y.iloc[np.flatnonzero(fit_mask)]
        y_validation = y.iloc[np.flatnonzero(validation_mask)]
        if y_fit.nunique() < 2 or y_validation.empty:
            self.gate_status_ = "fixed_insufficient_inner_outcomes"
            return 0.5

        X_fit = X.iloc[np.flatnonzero(fit_mask)]
        X_validation = X.iloc[np.flatnonzero(validation_mask)]
        local, context = self._fit_experts(X_fit, y_fit)

        local_probability = local.predict_proba(
            X_validation.loc[:, self.local_columns_]
        )[:, 1]
        context_probability = context.predict_proba(
            X_validation.loc[:, self.feature_columns_]
        )[:, 1]

        grid = np.linspace(0.0, 1.0, max(2, self.weight_grid_size))
        losses = []
        actual = np.asarray(y_validation, dtype=int)
        for context_weight in grid:
            probability = (
                (1.0 - context_weight) * local_probability
                + context_weight * context_probability
            )
            losses.append(float(brier_score_loss(actual, probability)))

        best = int(np.argmin(losses))
        self.gate_status_ = "nested_temporal_validation"
        self.gate_validation_brier_ = float(losses[best])
        return float(grid[best])

    def fit(self, X, y):
        frame = self._validate_frame(X).copy()
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.gate_validation_dates < 1:
            raise ValueError("gate_validation_dates must be >= 1")
        if self.weight_grid_size < 2:
            raise ValueError("weight_grid_size must be >= 2")

        target = pd.Series(np.asarray(y, dtype=int), index=frame.index)
        if target.nunique() < 2:
            raise ValueError("y must contain both outcome classes")

        self.classes_ = np.array([0, 1], dtype=int)
        self.feature_columns_ = [str(column) for column in frame.columns]
        self.local_columns_ = self._local_columns(frame)
        self.context_columns_ = [
            column
            for column in self.feature_columns_
            if column.startswith(self.context_prefix)
        ]
        self.n_features_in_ = len(self.feature_columns_)
        self.context_feature_count_ = len(self.context_columns_)

        self.context_weight_ = self._select_weight(frame, target)
        self.local_model_, self.context_model_ = self._fit_experts(frame, target)
        return self

    def _aligned(self, X) -> pd.DataFrame:
        frame = self._validate_frame(X)
        missing = [
            column for column in self.feature_columns_ if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                "prediction frame is missing fitted columns: " + ", ".join(missing)
            )
        return frame.loc[:, self.feature_columns_]

    def predict_proba(self, X):
        frame = self._aligned(X)
        local_probability = self.local_model_.predict_proba(
            frame.loc[:, self.local_columns_]
        )[:, 1]
        context_probability = self.context_model_.predict_proba(frame)[:, 1]
        probability = (
            (1.0 - self.context_weight_) * local_probability
            + self.context_weight_ * context_probability
        )
        probability = np.clip(probability, 1e-9, 1.0 - 1e-9)
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


__all__ = ["PREACTRelationalHazardMixture"]
