"""Service contracts for PREACT's three primary product surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd

from preact.feature_store.replay_dataset import build_replay_dataset

from preact.history.replay import (
    ForecastObservation,
    HistoricalReplayEngine,
    ReplayMetrics,
    ReplaySpec,
    evaluate_binary_forecasts,
)
from preact.history.schema import EvidenceClass, Provenance, TemporalRecord
from preact.history.warehouse import HistoricalWarehouse
from preact.models.replay_baseline import (
    ReplayBacktestResult,
    purged_walk_forward_backtest,
)


@dataclass(frozen=True)
class AtlasState:
    entity_id: str
    knowledge_cutoff: datetime
    valid_at: datetime
    records: tuple[dict, ...]


class HistoricalAtlasService:
    """Read a historical world state with explicit knowledge and valid-time clocks."""

    def __init__(self, warehouse: HistoricalWarehouse) -> None:
        self.warehouse = warehouse

    def state(
        self,
        *,
        entity_id: str,
        knowledge_cutoff: datetime,
        valid_at: datetime | None = None,
        variable: str | None = None,
    ) -> AtlasState:
        world_time = valid_at or knowledge_cutoff
        rows = self.warehouse.as_of(
            cutoff=knowledge_cutoff,
            valid_at=world_time,
            entity_id=entity_id,
            variable=variable,
        )
        return AtlasState(
            entity_id=entity_id,
            knowledge_cutoff=knowledge_cutoff,
            valid_at=world_time,
            records=tuple(rows),
        )


class ReplayLabService:
    """Orchestrate frozen historical model runs and evaluate probabilistic outputs."""

    def __init__(self, engine: HistoricalReplayEngine | None = None) -> None:
        self.engine = engine or HistoricalReplayEngine()

    def run(
        self,
        *,
        records: Iterable[TemporalRecord],
        cutoff: datetime,
        horizon: timedelta,
        model: Callable[[Sequence[TemporalRecord], ReplaySpec], object],
        entity_ids: tuple[str, ...] = (),
        variables: tuple[str, ...] = (),
    ) -> object:
        spec = ReplaySpec(
            cutoff=cutoff,
            horizon=horizon,
            entity_ids=entity_ids,
            variables=variables,
        )
        return self.engine.run(records, spec, model)

    @staticmethod
    def evaluate(observations: Iterable[ForecastObservation]) -> ReplayMetrics:
        return evaluate_binary_forecasts(observations)


    @staticmethod
    def backtest_binary(
        *,
        features: pd.DataFrame,
        target: pd.Series,
        horizon_days: int,
        n_splits: int = 5,
        calibration_fraction: float = 0.20,
    ) -> ReplayBacktestResult:
        return purged_walk_forward_backtest(
            features,
            target,
            horizon_days=horizon_days,
            n_splits=n_splits,
            calibration_fraction=calibration_fraction,
        )


    @staticmethod
    def backtest_from_warehouse(
        *,
        warehouse: HistoricalWarehouse,
        entity_id: str,
        cutoffs: Iterable[datetime],
        feature_variables: Iterable[str],
        target_variable: str,
        horizon_days: int,
        n_splits: int = 5,
        calibration_fraction: float = 0.20,
    ) -> ReplayBacktestResult:
        dataset = build_replay_dataset(
            warehouse,
            entity_id=entity_id,
            cutoffs=cutoffs,
            feature_variables=feature_variables,
            target_variable=target_variable,
            horizon_days=horizon_days,
        )
        return purged_walk_forward_backtest(
            dataset.features,
            dataset.target,
            horizon_days=horizon_days,
            n_splits=n_splits,
            calibration_fraction=calibration_fraction,
        )


@dataclass(frozen=True)
class ScenarioIntervention:
    variable: str
    operation: str
    value: float
    description: str = ""

    def __post_init__(self) -> None:
        if self.operation not in {"add", "multiply", "replace"}:
            raise ValueError("operation must be add, multiply or replace")


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    baseline_cutoff: datetime
    generated_at: datetime
    records: tuple[TemporalRecord, ...]
    interventions: tuple[ScenarioIntervention, ...]


class ScenarioLabService:
    """Create transparent baseline perturbations.

    This service is deliberately only the scenario contract. Advanced stochastic,
    structural and agent-based models can consume the resulting counterfactual
    state later; these deterministic perturbations are not presented as forecasts.
    """

    @staticmethod
    def _apply(value: object, intervention: ScenarioIntervention) -> object:
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"scenario intervention {intervention.variable} requires numeric baseline"
            )
        if intervention.operation == "add":
            return float(value) + intervention.value
        if intervention.operation == "multiply":
            return float(value) * intervention.value
        return float(intervention.value)

    def apply(
        self,
        *,
        scenario_id: str,
        baseline: Sequence[TemporalRecord],
        interventions: Sequence[ScenarioIntervention],
        baseline_cutoff: datetime,
        generated_at: datetime | None = None,
    ) -> ScenarioResult:
        generated_at = generated_at or datetime.now(timezone.utc)
        by_variable = {item.variable: item for item in interventions}
        output: list[TemporalRecord] = []

        for record in baseline:
            intervention = by_variable.get(record.variable)
            if intervention is None:
                output.append(record)
                continue
            value = self._apply(record.value, intervention)
            output.append(
                TemporalRecord(
                    record_id=f"scenario:{scenario_id}:{record.record_id}",
                    entity_id=record.entity_id,
                    variable=record.variable,
                    value=value,
                    valid_from=record.valid_from,
                    valid_to=record.valid_to,
                    known_at=generated_at,
                    evidence_class=EvidenceClass.COUNTERFACTUAL,
                    uncertainty=record.uncertainty,
                    attributes={
                        **record.attributes,
                        "scenario_id": scenario_id,
                        "baseline_record_id": record.record_id,
                        "intervention": {
                            "operation": intervention.operation,
                            "value": intervention.value,
                            "description": intervention.description,
                        },
                    },
                    provenance=Provenance(
                        source="scenario_lab",
                        source_ref=scenario_id,
                        retrieved_at=generated_at,
                        dataset_version=None,
                        licence=None,
                        transform=(
                            f"{intervention.operation}({record.variable}, "
                            f"{intervention.value})"
                        ),
                        notes="Counterfactual state; not an observed or forecast value.",
                    ),
                )
            )

        return ScenarioResult(
            scenario_id=scenario_id,
            baseline_cutoff=baseline_cutoff,
            generated_at=generated_at,
            records=tuple(output),
            interventions=tuple(interventions),
        )
