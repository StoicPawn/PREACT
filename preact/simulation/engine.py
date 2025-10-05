"""Tick-based simulation engine orchestrating the PREACT MVP modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from .economy import EconomyCore, EconomyParameters, EconomyState
from .policy import PolicyCore, PolicyParameters
from .scenario import Scenario
from .sentiment import SentimentCore
from .events import EventTimeline


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation runtime parameters."""

    horizon: int = 12
    time_step: str = "month"
    store_agent_history: bool = True


SimulationProgressCallback = Callable[[int, int, list[str]], None]


class SimulationEngine:
    """Coordinate the policy, economy and sentiment cores."""

    def __init__(
        self,
        *,
        simulation_config: Optional[SimulationConfig] = None,
        policy_core: Optional[PolicyCore] = None,
        economy_core: Optional[EconomyCore] = None,
        sentiment_core: Optional[SentimentCore] = None,
    ) -> None:
        self.simulation_config = simulation_config
        self._policy_core = policy_core
        self._economy_core = economy_core
        self._sentiment_core = sentiment_core

    def _ensure_modules(self, policy: PolicyParameters, economy: EconomyParameters) -> tuple[PolicyCore, EconomyCore, SentimentCore]:
        policy_core = self._policy_core or PolicyCore(policy)
        economy_core = self._economy_core or EconomyCore(economy)
        sentiment_core = self._sentiment_core or SentimentCore()
        return policy_core, economy_core, sentiment_core

    def run(
        self,
        scenario: Scenario,
        *,
        progress_callback: SimulationProgressCallback | None = None,
    ) -> "SimulationResults":
        """Execute the simulation for the provided scenario."""

        config = self.simulation_config or scenario.simulation_config
        policy_core, economy_core, sentiment_core = self._ensure_modules(scenario.policy, scenario.economy)

        population = scenario.population.copy(deep=True)
        population.set_index("agent_id", inplace=True, drop=False)
        baseline_frame = policy_core.apply(population)
        baseline_disposable = baseline_frame["disposable_income"].copy()

        employment_rate = float(population["employment_status"].str.lower().eq("employed").mean())
        state = EconomyState(
            cpi=scenario.economy.initial_cpi,
            unemployment_rate=1 - employment_rate,
            employment_rate=employment_rate,
            labour_demand_ratio=1.0,
        )
        previous_state: EconomyState | None = None

        disposable_history = pd.Series(0.0, index=population.index)
        consumption_history = pd.Series(0.0, index=population.index)
        tax_history = pd.Series(0.0, index=population.index)
        transfer_history = pd.Series(0.0, index=population.index)

        timeline_records: list[dict[str, float | str]] = []
        events_timeline: EventTimeline | None = scenario.events

        for tick in range(config.horizon):
            event_snapshot = (
                events_timeline.snapshot(tick, base=scenario.shock)
                if events_timeline
                else None
            )
            effective_shock = event_snapshot.shock if event_snapshot else scenario.shock
            adjustment = event_snapshot.adjustment if event_snapshot else None

            fiscal_frame = policy_core.apply(population, adjustment=adjustment)
            consumption = economy_core.compute_consumption(fiscal_frame)

            disposable_history = disposable_history.add(fiscal_frame["disposable_income"], fill_value=0.0)
            consumption_history = consumption_history.add(consumption, fill_value=0.0)
            tax_history = tax_history.add(fiscal_frame["tax_liability"], fill_value=0.0)
            transfer_history = transfer_history.add(fiscal_frame["transfers"], fill_value=0.0)

            population = economy_core.update_labour_market(
                population=population,
                consumption=consumption,
                firms=scenario.firms,
                tick=tick,
                shock=effective_shock,
            )
            state = economy_core.update_state(
                population=population,
                consumption=consumption,
                previous_state=state,
                tick=tick,
                shock=effective_shock,
            )

            if event_snapshot and event_snapshot.inflation_delta:
                state = EconomyState(
                    cpi=max(40.0, state.cpi * (1 + event_snapshot.inflation_delta)),
                    unemployment_rate=state.unemployment_rate,
                    employment_rate=state.employment_rate,
                    labour_demand_ratio=state.labour_demand_ratio,
                )

            sentiment = sentiment_core.compute(
                fiscal_frame["disposable_income"],
                state=state,
                previous_state=previous_state,
                baseline_income=baseline_disposable.mean(),
            )
            previous_state = state

            tick_record: dict[str, float | str] = {
                "tick": tick,
                "tax_revenue": float(fiscal_frame["tax_liability"].sum()),
                "transfer_spending": float(fiscal_frame["transfers"].sum()),
                "budget_balance": float(
                    fiscal_frame["tax_liability"].sum() - fiscal_frame["transfers"].sum()
                ),
                "unemployment_rate": state.unemployment_rate,
                "employment_rate": state.employment_rate,
                "consumption_total": float(consumption.sum()),
                "consumption_mean": float(consumption.mean()),
                "cpi": state.cpi,
                "sentiment": sentiment,
                "labour_demand_ratio": state.labour_demand_ratio,
            }

            if event_snapshot:
                tick_record.update(
                    {
                        "active_events": ", ".join(event_snapshot.event_names),
                        "event_economic_intensity": event_snapshot.economic_intensity,
                        "event_inflation_delta": event_snapshot.inflation_delta,
                        "policy_adjustment_multiplier": (
                            event_snapshot.adjustment.unemployment_benefit_multiplier
                            if event_snapshot.adjustment
                            else 1.0
                        ),
                    }
                )
            else:
                tick_record.update(
                    {
                        "active_events": "",
                        "event_economic_intensity": 0.0,
                        "event_inflation_delta": 0.0,
                        "policy_adjustment_multiplier": 1.0,
                    }
                )
            timeline_records.append(tick_record)

            if progress_callback:
                log_messages = [
                    f"Budget balance: {tick_record['budget_balance']:.2f}",
                    f"Employment rate: {tick_record['employment_rate']:.1%}",
                    f"Unemployment rate: {tick_record['unemployment_rate']:.1%}",
                    f"CPI: {tick_record['cpi']:.2f}",
                    f"Sentiment: {tick_record['sentiment']:.2f}",
                ]
                if event_snapshot and event_snapshot.event_names:
                    log_messages.append("Events: " + ", ".join(event_snapshot.event_names))
                progress_callback(tick, config.horizon, log_messages)

        average_disposable = disposable_history / config.horizon
        average_consumption = consumption_history / config.horizon
        final_disposable = fiscal_frame["disposable_income"].copy()

        agent_metrics = pd.DataFrame(
            {
                "agent_id": population.index,
                "baseline_disposable": baseline_disposable,
                "average_disposable": average_disposable,
                "final_disposable": final_disposable,
                "average_consumption": average_consumption,
                "total_taxes": tax_history,
                "total_transfers": transfer_history,
            }
        )
        agent_metrics["delta_disposable"] = agent_metrics["average_disposable"] - agent_metrics["baseline_disposable"]

        timeline = pd.DataFrame(timeline_records)

        from .results import SimulationResults  # local import to avoid circular dependency

        return SimulationResults(
            scenario_name=scenario.name,
            policy=scenario.policy,
            timeline=timeline,
            agent_metrics=agent_metrics,
            config=config,
            metadata=scenario.metadata or {},
        )
