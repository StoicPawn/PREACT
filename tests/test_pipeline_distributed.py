"""Tests for the distributed simulation runner."""
from __future__ import annotations

from preact.pipeline.distributed import DistributedSimulationRunner
from preact.simulation import (
    EconomyParameters,
    FirmParameters,
    PolicyParameters,
    PopulationParameters,
    ScenarioBuilder,
    SimulationConfig,
    TaxBracket,
)


def _builder() -> ScenarioBuilder:
    population = PopulationParameters(
        size=60,
        income_mean=25_000,
        income_sigma=0.55,
        employment_rate=0.9,
        sector_shares={"services": 0.6, "industry": 0.4},
    )
    policy = PolicyParameters(
        tax_brackets=[TaxBracket(threshold=28_000, rate=0.17)],
        base_deduction=4_500,
        child_subsidy=90.0,
    )
    economy = EconomyParameters(
        baseline_consumption=population.size * population.income_mean * 0.6,
    )
    firms = FirmParameters(
        size=70,
        sector_shares={"services": 0.65, "industry": 0.35},
        productivity_mean=1.05,
        productivity_sigma=0.2,
        employment_capacity_mean=30,
    )
    config = SimulationConfig(horizon=3)
    return ScenarioBuilder(
        population_params=population,
        firm_params=firms,
        economy_params=economy,
        policy_params=policy,
        simulation_config=config,
        seed=55,
    )
def test_thread_backend_executes_batch() -> None:
    builder = _builder()
    base = builder.build(name="Base")
    alt_policy = PolicyParameters(
        tax_brackets=[TaxBracket(threshold=30_000, rate=0.19)],
        base_deduction=4_700,
        child_subsidy=120.0,
    )
    variant = base.with_policy(alt_policy, name="Variant")

    runner = DistributedSimulationRunner(backend="thread", max_workers=2)
    results = runner.run_batch([base, variant])

    names = {result.scenario_name for result in results}
    assert names == {"Base", "Variant"}

    summary = runner.map_reduce([base, variant], lambda items: max(items, key=lambda res: res.kpis()["tax_revenue"]))
    assert summary.scenario_name in names
