"""Simulation output utilities for PREACT MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, TYPE_CHECKING

import numpy as np
import pandas as pd

from .policy import PolicyParameters

if TYPE_CHECKING:  # pragma: no cover - type checking helpers
    from .engine import SimulationConfig


def _gini(values: pd.Series) -> float:
    """Compute the Gini coefficient of a distribution."""

    arr = values.to_numpy(dtype=float)
    if len(arr) == 0:
        return 0.0
    if np.allclose(arr, 0):
        return 0.0
    sorted_arr = np.sort(arr)
    index = np.arange(1, len(sorted_arr) + 1)
    return float((2 * np.sum(index * sorted_arr) / (len(sorted_arr) * sorted_arr.sum())) - (len(sorted_arr) + 1) / len(sorted_arr))


@dataclass
class SimulationResults:
    """Container exposing convenient accessors for simulation outputs."""

    scenario_name: str
    policy: PolicyParameters
    timeline: pd.DataFrame
    agent_metrics: pd.DataFrame
    config: "SimulationConfig"
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        self.timeline = self.timeline.sort_values("tick").reset_index(drop=True)
        frame = self.agent_metrics.copy()
        if frame.index.name == "agent_id":
            frame = frame.reset_index(drop=True)
        self.agent_metrics = frame.sort_values("agent_id").reset_index(drop=True)

    def kpis(self) -> Dict[str, object]:
        """Return the minimum KPI set requested by the MVP blueprint."""

        latest = self.timeline.iloc[-1]
        consumption_deciles = self._consumption_by_decile()
        sentiment_deciles = self._sentiment_by_decile()
        winners = self.winners_losers()

        return {
            "scenario": self.scenario_name,
            "tax_revenue": float(latest["tax_revenue"]),
            "transfer_spending": float(latest["transfer_spending"]),
            "budget_balance": float(latest["budget_balance"]),
            "unemployment_rate": float(latest["unemployment_rate"]),
            "employment_rate": float(latest["employment_rate"]),
            "cpi": float(latest["cpi"]),
            "sentiment": float(latest["sentiment"]),
            "consumption_by_decile": consumption_deciles,
            "sentiment_by_decile": sentiment_deciles,
            "gini_pre": _gini(self.agent_metrics["baseline_disposable"]),
            "gini_post": _gini(self.agent_metrics["average_disposable"]),
            "winners_losers": winners,
        }

    def _consumption_by_decile(self) -> Dict[str, float]:
        consumption = self.agent_metrics["average_consumption"]
        if consumption.empty:
            return {}
        deciles = pd.qcut(consumption, 10, labels=False, duplicates="drop")
        groups = consumption.groupby(deciles)
        return {f"decile_{int(k)+1}": float(v) for k, v in groups.mean().items()}

    def _sentiment_by_decile(self) -> Dict[str, float]:
        disposable = self.agent_metrics["average_disposable"]
        if disposable.empty:
            return {}
        deciles = pd.qcut(disposable, 10, labels=False, duplicates="drop")
        mean_disposable = disposable.mean()
        scores: Dict[str, float] = {}
        for decile in sorted(deciles.dropna().unique()):
            mask = deciles == decile
            subset = disposable[mask]
            delta = 0.0 if mean_disposable == 0 else (subset.mean() - mean_disposable) / mean_disposable
            scores[f"decile_{int(decile)+1}"] = float(np.clip(55 + 100 * 0.4 * delta, 0, 100))
        return scores

    def winners_losers(self, clusters: int = 5) -> pd.DataFrame:
        """Return a winners/losers table based on disposable income delta."""

        delta = self.agent_metrics["delta_disposable"]
        if delta.empty:
            return pd.DataFrame(columns=["cluster", "mean_delta", "count"])
        try:
            bins = pd.qcut(delta, clusters, labels=False, duplicates="drop")
        except ValueError:
            bins = pd.Series(np.zeros(len(delta), dtype=int), index=delta.index)
        summary = (
            pd.DataFrame({"cluster": bins, "delta": delta})
            .groupby("cluster")
            .agg(mean_delta=("delta", "mean"), count=("delta", "size"))
            .reset_index()
        )
        return summary

    def to_frame(self) -> pd.DataFrame:
        """Return a flattened frame combining timeline and agent-level aggregates."""

        latest = self.timeline.iloc[-1]
        data = {
            "scenario": self.scenario_name,
            "tax_revenue": latest["tax_revenue"],
            "transfer_spending": latest["transfer_spending"],
            "budget_balance": latest["budget_balance"],
            "unemployment_rate": latest["unemployment_rate"],
            "employment_rate": latest["employment_rate"],
            "cpi": latest["cpi"],
            "sentiment": latest["sentiment"],
            "gini_pre": _gini(self.agent_metrics["baseline_disposable"]),
            "gini_post": _gini(self.agent_metrics["average_disposable"]),
        }
        return pd.DataFrame([data])


@dataclass
class SimulationComparison:
    """Pair of scenarios compared A/B style."""

    base: SimulationResults
    reform: SimulationResults

    def delta(self) -> Dict[str, float]:
        """Return headline differences between reform and base."""

        base_kpis = self.base.kpis()
        reform_kpis = self.reform.kpis()
        deltas: Dict[str, float] = {}
        for key in [
            "tax_revenue",
            "transfer_spending",
            "budget_balance",
            "unemployment_rate",
            "employment_rate",
            "cpi",
            "sentiment",
            "gini_pre",
            "gini_post",
        ]:
            deltas[key] = float(reform_kpis[key]) - float(base_kpis[key])
        return deltas
