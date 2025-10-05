"""High-level orchestration service to run and persist simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional

from .economy import Shock
from .engine import SimulationEngine, SimulationProgressCallback
from .results import SimulationResults, SimulationComparison
from .policy import PolicyParameters, TaxBracket
from .storage import SimulationRepository
from .templates import ScenarioTemplate, default_templates


@dataclass
class SimulationRunSummary:
    """Summary payload returned by :class:`SimulationService`."""

    base_run_id: str
    base_kpis: Dict[str, Any]
    reform_run_id: str | None
    reform_kpis: Optional[Dict[str, Any]]
    comparison: Optional[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_run_id": self.base_run_id,
            "base_kpis": self.base_kpis,
            "reform_run_id": self.reform_run_id,
            "reform_kpis": self.reform_kpis,
            "comparison": self.comparison,
        }


class SimulationService:
    """Coordinate scenario templates, engine execution and persistence."""

    def __init__(
        self,
        repository: SimulationRepository,
        *,
        templates: Mapping[str, ScenarioTemplate] | None = None,
        engine: SimulationEngine | None = None,
    ) -> None:
        self.repository = repository
        available = templates or default_templates()
        self.templates: Dict[str, ScenarioTemplate] = {tpl.name: tpl for tpl in available.values()} if isinstance(available, Mapping) else {}
        if not self.templates:
            # allow callers to pass an iterable of templates
            self.templates = {template.name: template for template in available}  # type: ignore[arg-type]
        self.engine = engine or SimulationEngine()

    def list_scenarios(self) -> list[Dict[str, object]]:
        """Return summaries for available templates."""

        return [template.to_summary() for template in self.templates.values()]

    def run(
        self,
        *,
        template_name: str,
        horizon: int | None = None,
        policy_payload: Optional[Mapping[str, Any]] = None,
        base_policy_payload: Optional[Mapping[str, Any]] = None,
        shock_payload: Optional[Mapping[str, Any]] = None,
        seed: int = 42,
        metadata: Optional[MutableMapping[str, Any]] = None,
        progress_callback: SimulationProgressCallback | None = None,
    ) -> SimulationRunSummary:
        """Run a simulation (and optional reform scenario) and persist the results."""

        template = self._get_template(template_name)
        builder = template.builder(horizon=horizon, seed=seed)

        base_policy = self._policy_from_payload(base_policy_payload) if base_policy_payload else template.policy
        base_metadata = {"template": template.name, "role": "base"}
        if metadata:
            base_metadata.update(metadata)
        shock = self._shock_from_payload(shock_payload)
        base_scenario = builder.build(name=f"{template.name} - Base", policy=base_policy, shock=shock, metadata=base_metadata)
        base_results = self.engine.run(
            base_scenario,
            progress_callback=self._wrap_progress_callback(
                progress_callback, role="Scenario base"
            ),
        )
        base_run_id = self.repository.store(base_results)

        reform_results: SimulationResults | None = None
        reform_run_id: str | None = None

        if policy_payload:
            reform_policy = self._policy_from_payload(policy_payload)
            reform_metadata = {"template": template.name, "role": "reform"}
            if metadata:
                reform_metadata.update(metadata)
            reform_scenario = builder.build(
                name=f"{template.name} - Reform",
                policy=reform_policy,
                shock=shock,
                metadata=reform_metadata,
            )
            reform_results = self.engine.run(
                reform_scenario,
                progress_callback=self._wrap_progress_callback(
                    progress_callback, role="Scenario riforma"
                ),
            )
            reform_run_id = self.repository.store(reform_results)

        comparison = None
        reform_kpis = None
        if reform_results is not None:
            comparison = SimulationComparison(base=base_results, reform=reform_results).delta()
            reform_kpis = reform_results.kpis()

        return SimulationRunSummary(
            base_run_id=base_run_id,
            base_kpis=base_results.kpis(),
            reform_run_id=reform_run_id,
            reform_kpis=reform_kpis,
            comparison=comparison,
        )

    @staticmethod
    def _wrap_progress_callback(
        callback: SimulationProgressCallback | None, *, role: str
    ) -> SimulationProgressCallback | None:
        if not callback:
            return None

        def _wrapped(tick: int, horizon: int, messages: list[str]) -> None:
            annotated = [f"{role}: {message}" for message in messages]
            callback(tick, horizon, annotated)

        return _wrapped

    def fetch(self, run_id: str) -> SimulationResults:
        """Retrieve a stored simulation run."""

        return self.repository.fetch(run_id)

    def _get_template(self, name: str) -> ScenarioTemplate:
        try:
            return self.templates[name]
        except KeyError as exc:  # pragma: no cover - defensive clause
            raise KeyError(f"Unknown template '{name}'") from exc

    @staticmethod
    def _policy_from_payload(payload: Mapping[str, Any]) -> PolicyParameters:
        brackets_payload = payload.get("brackets") or payload.get("tax_brackets") or []
        brackets = [
            TaxBracket(threshold=float(item["threshold"]), rate=float(item["rate"]))
            for item in brackets_payload
        ]
        return PolicyParameters(
            tax_brackets=brackets,
            base_deduction=float(payload.get("base_deduction", 0.0)),
            child_subsidy=float(payload.get("child_subsidy", 0.0)),
            unemployment_benefit=float(payload.get("unemployment_benefit", 900.0)),
        )

    @staticmethod
    def _shock_from_payload(payload: Optional[Mapping[str, Any]]) -> Shock | None:
        if not payload:
            return None
        return Shock(
            name=str(payload.get("name", "shock")),
            intensity=float(payload.get("intensity", 0.0)),
            start_tick=int(payload.get("start_tick", 0)),
            end_tick=payload.get("end_tick"),
        )
