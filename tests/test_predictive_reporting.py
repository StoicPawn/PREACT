from preact.models.benchmark_suite import (
    BenchmarkMetrics,
    DependenceDiagnostic,
    ModelBenchmark,
    SkillInterval,
)
from preact.models.reporting import benchmark_diagnostics


def test_benchmark_diagnostics_preserves_dependence_aware_and_iid_intervals():
    metrics = BenchmarkMetrics(100, 5, 0.08, 0.10, 0.20, 0.3, 0.7, 0.2, 0.01, 0.05)
    block = SkillInterval(0.02, 0.18, 0.31, 500)
    iid = SkillInterval(0.08, 0.19, 0.27, 500)
    diagnostic = DependenceDiagnostic(block=block, iid=iid, width_ratio=1.526315789)
    benchmark = ModelBenchmark("candidate", None, metrics, block, diagnostic)

    payload = benchmark_diagnostics(benchmark)

    assert payload["brier_skill_interval"] == payload["dependence_diagnostic"]["block"]
    assert payload["dependence_diagnostic"]["iid"]["lower"] == 0.08
    assert payload["dependence_diagnostic"]["width_ratio"] > 1.5


def test_benchmark_diagnostics_handles_legacy_benchmark_without_diagnostic():
    metrics = BenchmarkMetrics(0, 0, None, None, None, None, None, None, None, None)
    interval = SkillInterval(None, None, None, 0)
    benchmark = ModelBenchmark("legacy", None, metrics, interval)

    payload = benchmark_diagnostics(benchmark)

    assert payload["dependence_diagnostic"] is None
