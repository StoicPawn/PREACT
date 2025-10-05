"""Distributed execution helpers for PREACT simulations."""
from __future__ import annotations

import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence

from ..simulation.engine import SimulationEngine
from ..simulation.results import SimulationResults
from ..simulation.scenario import Scenario


def _run_simulation(scenario: Scenario) -> SimulationResults:
    engine = SimulationEngine()
    return engine.run(scenario)


class DistributedSimulationRunner:
    """Execute multiple simulation scenarios using parallel backends."""

    def __init__(
        self,
        *,
        backend: str | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.backend = backend or self._auto_backend()
        self.max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)

    def _auto_backend(self) -> str:
        if self._ray_available():
            return "ray"
        return "thread"

    @staticmethod
    def _ray_available() -> bool:
        return importlib.util.find_spec("ray") is not None

    def run_batch(self, scenarios: Sequence[Scenario]) -> list[SimulationResults]:
        if not scenarios:
            return []
        if self.backend == "ray":
            if not self._ray_available():
                raise RuntimeError("Ray backend requested but ray is not installed")
            return self._run_with_ray(scenarios)
        if self.backend == "process":
            executor_cls = ProcessPoolExecutor
        else:
            executor_cls = ThreadPoolExecutor
        results: list[SimulationResults] = []
        with executor_cls(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_run_simulation, scenario): scenario.name for scenario in scenarios}
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda res: res.scenario_name)
        return results

    def _run_with_ray(self, scenarios: Sequence[Scenario]) -> list[SimulationResults]:
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="WARNING")

        @ray.remote
        def _simulate(scenario: Scenario) -> SimulationResults:
            engine = SimulationEngine()
            return engine.run(scenario)

        futures = [_simulate.remote(scenario) for scenario in scenarios]
        results = ray.get(futures)
        if ray.is_initialized():
            ray.shutdown()
        results.sort(key=lambda res: res.scenario_name)
        return results

    def map_reduce(
        self,
        scenarios: Sequence[Scenario],
        reducer: Callable[[Iterable[SimulationResults]], SimulationResults],
    ) -> SimulationResults:
        """Execute scenarios in parallel and apply a reducer to the results."""

        results = self.run_batch(scenarios)
        return reducer(results)


__all__ = ["DistributedSimulationRunner", "_run_simulation"]
