"""Tests for the PREACT simulation stack."""

from __future__ import annotations

from preact.simulation import (
    EconomyParameters,
    FirmParameters,
    PolicyParameters,
    PopulationParameters,
    ScenarioBuilder,
    SimulationComparison,
    SimulationConfig,
    SimulationEngine,
    TaxBracket,
)


def _build_policy() -> PolicyParameters:
    brackets = [TaxBracket(threshold=30_000, rate=0.18), TaxBracket(threshold=75_000, rate=0.28)]
    return PolicyParameters(tax_brackets=brackets, base_deduction=5_000, child_subsidy=120.0)


def _build_economy(population_size: int, income_mean: float) -> EconomyParameters:
    baseline_consumption = population_size * income_mean * 0.7
    return EconomyParameters(baseline_consumption=baseline_consumption)


def _build_population_params(population_size: int) -> PopulationParameters:
    return PopulationParameters(
        size=population_size,
        income_mean=28_000,
        income_sigma=0.6,
        employment_rate=0.9,
        sector_shares={"services": 0.6, "industry": 0.4},
    )


def _build_firm_params() -> FirmParameters:
    return FirmParameters(
        size=150,
        sector_shares={"services": 0.6, "industry": 0.4},
        productivity_mean=1.2,
        productivity_sigma=0.3,
        employment_capacity_mean=40,
    )


def test_scenario_builder_generates_expected_shapes() -> None:
    policy = _build_policy()
    population_params = _build_population_params(200)
    economy_params = _build_economy(population_params.size, population_params.income_mean)
    firm_params = _build_firm_params()
    config = SimulationConfig(horizon=6)
    builder = ScenarioBuilder(
        population_params=population_params,
        firm_params=firm_params,
        economy_params=economy_params,
        policy_params=policy,
        simulation_config=config,
        seed=123,
    )

    scenario = builder.build(name="TestScenario")
    assert len(scenario.population) == population_params.size
    assert {"gross_income", "employment_status", "num_children"}.issubset(scenario.population.columns)
    assert len(scenario.firms) == firm_params.size

    updated_policy = PolicyParameters(
        tax_brackets=[TaxBracket(threshold=25_000, rate=0.2)],
        base_deduction=6_000,
        child_subsidy=150.0,
    )
    scenario_reform = scenario.with_policy(updated_policy, name="Reform")
    assert scenario_reform.name == "Reform"
    assert scenario_reform.policy is updated_policy
    assert scenario_reform.population.equals(scenario.population)


def test_simulation_engine_runs_and_returns_kpis() -> None:
    population_params = _build_population_params(120)
    policy = _build_policy()
    economy_params = _build_economy(population_params.size, population_params.income_mean)
    firm_params = _build_firm_params()
    config = SimulationConfig(horizon=4)
    builder = ScenarioBuilder(
        population_params=population_params,
        firm_params=firm_params,
        economy_params=economy_params,
        policy_params=policy,
        simulation_config=config,
        seed=321,
    )
    base_scenario = builder.build(name="Base")
    reform_policy = PolicyParameters(
        tax_brackets=[TaxBracket(threshold=35_000, rate=0.16), TaxBracket(threshold=80_000, rate=0.26)],
        base_deduction=5_500,
        child_subsidy=180.0,
    )
    reform_scenario = base_scenario.with_policy(reform_policy, name="Reform")

    engine = SimulationEngine(simulation_config=config)
    base_results = engine.run(base_scenario)
    reform_results = engine.run(reform_scenario)

    assert len(base_results.timeline) == config.horizon
    assert len(reform_results.timeline) == config.horizon
    assert len(base_results.agent_metrics) == population_params.size
    assert "delta_disposable" in base_results.agent_metrics.columns

    kpis = base_results.kpis()
    for key in [
        "tax_revenue",
        "transfer_spending",
        "budget_balance",
        "unemployment_rate",
        "employment_rate",
        "cpi",
        "sentiment",
        "consumption_by_decile",
        "gini_pre",
        "gini_post",
    ]:
        assert key in kpis

    comparison = SimulationComparison(base=base_results, reform=reform_results)
    delta = comparison.delta()
    assert set(delta.keys()) == {
        "tax_revenue",
        "transfer_spending",
        "budget_balance",
        "unemployment_rate",
        "employment_rate",
        "cpi",
        "sentiment",
        "gini_pre",
        "gini_post",
    }
