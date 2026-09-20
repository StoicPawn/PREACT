"""Leakage-safe sequential ensemble of OOS model forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


@dataclass(frozen=True)
class EnsembleResult:
    predictions: pd.DataFrame
    fold_weights: Mapping[int, Mapping[str, float]]


def _normalized_inverse_brier(
    history: Mapping[str, pd.DataFrame],
    *,
    floor: float = 1e-4,
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for name, frame in history.items():
        if frame.empty:
            continue
        score = float(brier_score_loss(frame["actual"], frame["probability"]))
        raw[name] = 1.0 / max(score, floor)
    if not raw:
        return {}
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


def sequential_oos_ensemble(
    predictions_by_model: Mapping[str, pd.DataFrame],
) -> EnsembleResult:
    """Combine models using only earlier-fold OOS performance.

    For fold 0, models receive equal weights. For later folds, weights are based
    exclusively on Brier scores from strictly earlier OOS folds.
    """

    if not predictions_by_model:
        return EnsembleResult(pd.DataFrame(), {})

    names = tuple(sorted(predictions_by_model))
    nonempty = [df for df in predictions_by_model.values() if not df.empty]
    if not nonempty:
        return EnsembleResult(pd.DataFrame(), {})

    folds = sorted(
        {
            int(fold)
            for df in nonempty
            for fold in df["fold"].unique().tolist()
        }
    )
    output: list[dict] = []
    fold_weights: dict[int, dict[str, float]] = {}

    for fold in folds:
        history = {
            name: frame.loc[frame["fold"] < fold]
            for name, frame in predictions_by_model.items()
        }
        weights = _normalized_inverse_brier(history)
        if not weights:
            weights = {name: 1.0 / len(names) for name in names}
        else:
            missing = [name for name in names if name not in weights]
            if missing:
                # Models without prior OOS history are excluded until they have
                # earned a weight from a previous fold.
                for name in missing:
                    weights[name] = 0.0
                total = sum(weights.values())
                weights = {k: v / total for k, v in weights.items()}

        fold_weights[int(fold)] = dict(weights)

        current = {
            name: frame.loc[frame["fold"] == fold].copy()
            for name, frame in predictions_by_model.items()
        }
        keys = None
        for frame in current.values():
            if frame.empty:
                continue
            frame_key = frame[["date", "entity_id", "actual", "baseline_probability"]]
            keys = frame_key if keys is None else keys
        if keys is None:
            continue

        merged = keys.drop_duplicates(["date", "entity_id"]).copy()
        for name, frame in current.items():
            if frame.empty:
                continue
            merged = merged.merge(
                frame[["date", "entity_id", "probability"]].rename(
                    columns={"probability": f"p:{name}"}
                ),
                on=["date", "entity_id"],
                how="left",
            )

        probability = np.zeros(len(merged), dtype=float)
        effective = np.zeros(len(merged), dtype=float)
        for name, weight in weights.items():
            column = f"p:{name}"
            if column not in merged.columns or weight <= 0:
                continue
            available = merged[column].notna().to_numpy()
            probability[available] += (
                merged.loc[available, column].to_numpy(dtype=float) * weight
            )
            effective[available] += weight

        valid = effective > 0
        probability[valid] /= effective[valid]
        probability[~valid] = merged.loc[
            ~valid, "baseline_probability"
        ].to_numpy(dtype=float)

        for row, p in zip(merged.to_dict(orient="records"), probability):
            output.append(
                {
                    "date": row["date"],
                    "entity_id": row["entity_id"],
                    "fold": int(fold),
                    "actual": int(row["actual"]),
                    "probability": float(np.clip(p, 1e-6, 1 - 1e-6)),
                    "baseline_probability": float(row["baseline_probability"]),
                }
            )

    frame = pd.DataFrame(output)
    if not frame.empty:
        frame = frame.sort_values(["date", "entity_id"]).reset_index(drop=True)
    return EnsembleResult(frame, fold_weights)
