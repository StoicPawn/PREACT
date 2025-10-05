"""Tests for adaptive policy agents."""
from __future__ import annotations

from preact.simulation import (
    EconomyParameters,
    FirmParameters,
    PolicyParameters,
    PopulationParameters,
    ScenarioBuilder,
    SimulationConfig,
    SimulationEngine,
    TaxBracket,
)
from preact.agents import AdaptivePolicyAgent


def _default_scenario() -> tuple[ScenarioBuilder, SimulationEngine]:
    population_params = PopulationParameters(
        size=80,
        income_mean=26_000,
        income_sigma=0.5,
        employment_rate=0.88,
        sector_shares={"services": 0.7, "industry": 0.3},
    )
    policy = PolicyParameters(
        tax_brackets=[TaxBracket(threshold=32_000, rate=0.2)],
        base_deduction=4_800,
        child_subsidy=110.0,
    )
    economy = EconomyParameters(
        baseline_consumption=population_params.size * population_params.income_mean * 0.65,
    )
    firms = FirmParameters(
        size=90,
        sector_shares={"services": 0.6, "industry": 0.4},
        productivity_mean=1.1,
        productivity_sigma=0.25,
        employment_capacity_mean=35,
    )
    config = SimulationConfig(horizon=4)
    builder = ScenarioBuilder(
        population_params=population_params,
        firm_params=firms,
        economy_params=economy,
        policy_params=policy,
        simulation_config=config,
        seed=91,
    )
    engine = SimulationEngine(simulation_config=config)
    return builder, engine


def test_adaptive_agent_improves_reward() -> None:
    builder, engine = _default_scenario()
    scenario = builder.build(name="Baseline")

    def reward_fn(results) -> float:
        kpis = results.kpis()
        unemployment_penalty = float(kpis["unemployment_rate"])
        inflation_penalty = max(0.0, (float(kpis["cpi"]) - 105.0) / 100.0)
        sentiment_bonus = float(kpis["sentiment"]) / 200.0
        return sentiment_bonus - unemployment_penalty - inflation_penalty

    agent = AdaptivePolicyAgent(engine, exploration_rate=0.1, learning_rate=0.4)
    outcome = agent.train(scenario, episodes=4, reward_fn=reward_fn)

    assert len(outcome.reward_history) == 5  # baseline + 4 episodes
    assert outcome.best_reward >= outcome.reward_history[0]
    assert isinstance(outcome.best_policy, PolicyParameters)
