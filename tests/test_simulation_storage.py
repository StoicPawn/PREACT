from pathlib import Path

from preact.simulation import (
    SimulationEngine,
    SimulationRepository,
    SimulationService,
    ScenarioBuilder,
    SimulationConfig,
    PolicyParameters,
    TaxBracket,
    PopulationParameters,
    FirmParameters,
    EconomyParameters,
)


def _scenario_builder() -> ScenarioBuilder:
    population = PopulationParameters(
        size=80,
        income_mean=26000,
        income_sigma=0.5,
        employment_rate=0.88,
        sector_shares={"services": 0.6, "industry": 0.4},
    )
    firms = FirmParameters(
        size=50,
        sector_shares={"services": 0.7, "industry": 0.3},
        productivity_mean=1.1,
        productivity_sigma=0.2,
        employment_capacity_mean=20,
    )
    policy = PolicyParameters(
        tax_brackets=[TaxBracket(threshold=28000, rate=0.19)],
        base_deduction=4000,
        child_subsidy=100.0,
    )
    economy = EconomyParameters(baseline_consumption=population.size * population.income_mean * 0.7)
    config = SimulationConfig(horizon=3)
    return ScenarioBuilder(
        population_params=population,
        firm_params=firms,
        economy_params=economy,
        policy_params=policy,
        simulation_config=config,
        seed=99,
    )


def test_repository_store_and_fetch(tmp_path) -> None:
    builder = _scenario_builder()
    scenario = builder.build(name="Test")
    engine = SimulationEngine()
    results = engine.run(scenario)

    repository = SimulationRepository(tmp_path / "runs.duckdb", export_dir=tmp_path / "exports")
    run_id = repository.store(results)
    assert run_id

    loaded = repository.fetch(run_id)
    assert loaded.scenario_name == "Test"
    assert not loaded.timeline.empty

    exports = repository.export(run_id, format="csv")
    assert exports["timeline"].exists()
    assert exports["agent_metrics"].exists()

    html_report = repository.export(run_id, format="html")
    assert html_report["report"].exists()

    pdf_report = repository.export(run_id, format="pdf")
    assert pdf_report["report"].exists()


def test_simulation_service_runs_and_compares(tmp_path) -> None:
    repository = SimulationRepository(tmp_path / "service.duckdb", export_dir=tmp_path / "exports")
    service = SimulationService(repository=repository)
    summary = service.run(
        template_name="Medium City",
        horizon=3,
        policy_payload={
            "brackets": [{"threshold": 30000, "rate": 0.2}],
            "base_deduction": 4200,
            "child_subsidy": 150.0,
        },
    )
    assert summary.base_run_id
    assert summary.reform_run_id
    assert summary.comparison is not None
    base_results = repository.fetch(summary.base_run_id)
    assert base_results.timeline.shape[0] == 3

    report_with_comparison = repository.export(
        summary.base_run_id,
        format="html",
        reform_run_id=summary.reform_run_id,
    )
    assert report_with_comparison["report"].exists()
