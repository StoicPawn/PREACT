"""Discrete-time event-history model with complementary-log-log link."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin


class ComplementaryLogLogHazard(BaseEstimator, ClassifierMixin):
    """Penalized Bernoulli hazard model using the cloglog link.

    p(event in interval | survived to interval) = 1 - exp(-exp(x beta)).

    This is a lightweight event-history baseline suitable for rare geopolitical
    events. It follows the sklearn estimator interface and is normally placed
    after imputation/scaling in a Pipeline.
    """

    def __init__(
        self,
        *,
        l2: float = 1.0,
        max_iter: int = 500,
        tolerance: float = 1e-7,
    ) -> None:
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

    @staticmethod
    def _probability(eta: np.ndarray) -> np.ndarray:
        eta = np.clip(eta, -25.0, 20.0)
        cumulative_hazard = np.exp(eta)
        p = -np.expm1(-cumulative_hazard)
        return np.clip(p, 1e-9, 1.0 - 1e-9)

    def fit(self, X, y):
        x = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if len(target) != len(x):
            raise ValueError("X/y length mismatch")
        classes = np.unique(target)
        if not np.all(np.isin(classes, [0.0, 1.0])):
            raise ValueError("y must be binary")
        self.classes_ = np.array([0, 1], dtype=int)

        design = np.column_stack([np.ones(len(x)), x])
        initial = np.zeros(design.shape[1], dtype=float)

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            eta = design @ beta
            p = self._probability(eta)

            loss = -np.sum(
                target * np.log(p)
                + (1.0 - target) * np.log1p(-p)
            )
            loss += 0.5 * self.l2 * float(beta[1:] @ beta[1:])

            eta_clip = np.clip(eta, -25.0, 20.0)
            hazard = np.exp(eta_clip)
            dp_deta = np.exp(eta_clip - hazard)
            dloss_deta = (
                (p - target)
                / np.clip(p * (1.0 - p), 1e-12, None)
            ) * dp_deta
            gradient = design.T @ dloss_deta
            gradient[1:] += self.l2 * beta[1:]
            return float(loss), np.asarray(gradient, dtype=float)

        result = minimize(
            lambda b: objective(b)[0],
            initial,
            jac=lambda b: objective(b)[1],
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tolerance},
        )
        if not result.success and not np.isfinite(result.fun):
            raise RuntimeError(f"hazard optimization failed: {result.message}")

        self.intercept_ = float(result.x[0])
        self.coef_ = np.asarray(result.x[1:], dtype=float).reshape(1, -1)
        self.n_features_in_ = x.shape[1]
        self.optimization_success_ = bool(result.success)
        self.optimization_message_ = str(result.message)
        return self

    def predict_proba(self, X):
        x = np.asarray(X, dtype=float)
        eta = self.intercept_ + x @ self.coef_[0]
        p = self._probability(eta)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
