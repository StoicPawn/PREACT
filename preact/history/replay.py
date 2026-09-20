"""Leakage-safe historical replay orchestration and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Callable, Iterable, Mapping, Sequence

from .schema import HistoricalQuery, TemporalRecord


@dataclass(frozen=True)
class ReplaySpec:
    cutoff: datetime
    horizon: timedelta
    valid_at: datetime | None = None
    entity_ids: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForecastObservation:
    entity_id: str
    issued_at: datetime
    horizon: timedelta
    probability: float
    outcome: int
    outcome_time: datetime | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.probability) <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if self.outcome not in (0, 1):
            raise ValueError("outcome must be binary")


@dataclass(frozen=True)
class ReplayMetrics:
    rows: int
    brier_score: float | None
    log_loss: float | None
    mean_probability: float | None
    event_rate: float | None
    calibration_gap: float | None


class HistoricalReplayEngine:
    """Freeze evidence at a cutoff, run a model, and audit leakage."""

    def freeze(
        self,
        records: Iterable[TemporalRecord],
        spec: ReplaySpec,
    ) -> list[TemporalRecord]:
        query = HistoricalQuery(
            knowledge_cutoff=spec.cutoff,
            valid_at=spec.valid_at,
            entity_ids=spec.entity_ids,
            variables=spec.variables,
        )
        frozen = query.filter(records)
        self.assert_no_future_knowledge(frozen, cutoff=spec.cutoff)
        return frozen

    @staticmethod
    def assert_no_future_knowledge(
        records: Iterable[TemporalRecord],
        *,
        cutoff: datetime,
    ) -> None:
        leaked = [record.record_id for record in records if record.known_at > cutoff]
        if leaked:
            preview = ", ".join(leaked[:5])
            raise ValueError(f"future-knowledge leakage detected: {preview}")

    def run(
        self,
        records: Iterable[TemporalRecord],
        spec: ReplaySpec,
        model: Callable[[Sequence[TemporalRecord], ReplaySpec], object],
    ) -> object:
        frozen = self.freeze(records, spec)
        return model(frozen, spec)

    @staticmethod
    def rolling_specs(
        *,
        start: datetime,
        end: datetime,
        step: timedelta,
        horizon: timedelta,
    ) -> list[ReplaySpec]:
        if step <= timedelta(0):
            raise ValueError("step must be positive")
        if horizon <= timedelta(0):
            raise ValueError("horizon must be positive")
        specs: list[ReplaySpec] = []
        cursor = start
        while cursor <= end:
            specs.append(ReplaySpec(cutoff=cursor, horizon=horizon))
            cursor += step
        return specs


def evaluate_binary_forecasts(
    observations: Iterable[ForecastObservation],
    *,
    epsilon: float = 1e-12,
) -> ReplayMetrics:
    rows = list(observations)
    if not rows:
        return ReplayMetrics(0, None, None, None, None, None)

    probabilities = [float(row.probability) for row in rows]
    outcomes = [int(row.outcome) for row in rows]
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(rows)
    eps = max(float(epsilon), 1e-15)
    log_loss = -sum(
        y * math.log(min(max(p, eps), 1.0 - eps))
        + (1 - y) * math.log(min(max(1.0 - p, eps), 1.0 - eps))
        for p, y in zip(probabilities, outcomes)
    ) / len(rows)
    mean_probability = sum(probabilities) / len(rows)
    event_rate = sum(outcomes) / len(rows)
    return ReplayMetrics(
        rows=len(rows),
        brier_score=float(brier),
        log_loss=float(log_loss),
        mean_probability=float(mean_probability),
        event_rate=float(event_rate),
        calibration_gap=float(mean_probability - event_rate),
    )
