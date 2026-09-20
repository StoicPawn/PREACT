"""Stable serialization helpers for predictive research diagnostics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .benchmark_suite import ModelBenchmark


def benchmark_diagnostics(benchmark: ModelBenchmark) -> dict[str, Any]:
    """Serialize benchmark uncertainty without dropping dependence diagnostics.

    The dependence-aware block interval remains the primary uncertainty measure;
    the IID interval and width ratio are emitted only as diagnostics so reports
    can expose when serial dependence materially changes uncertainty.
    """
    payload: dict[str, Any] = {
        "name": benchmark.name,
        "metrics": asdict(benchmark.metrics),
        "brier_skill_interval": asdict(benchmark.brier_skill_interval),
    }
    diagnostic = benchmark.dependence_diagnostic
    payload["dependence_diagnostic"] = (
        None
        if diagnostic is None
        else {
            "block": asdict(diagnostic.block),
            "iid": asdict(diagnostic.iid),
            "width_ratio": diagnostic.width_ratio,
            "iid_understates_uncertainty": (
                diagnostic.width_ratio is not None and diagnostic.width_ratio > 1.0
            ),
        }
    )
    return payload
