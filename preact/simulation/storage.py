"""Persistence utilities for simulation runs."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping
from uuid import uuid4

import duckdb
import pandas as pd

from .engine import SimulationConfig
from .policy import PolicyParameters, TaxBracket
from .results import SimulationResults


class SimulationRepository:
    """Persist :class:`SimulationResults` objects in DuckDB and export files."""

    def __init__(self, db_path: Path, *, export_dir: Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.export_dir = Path(export_dir) if export_dir else self.db_path.parent / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(self.db_path))
        self._initialise()

    def close(self) -> None:
        """Close the underlying DuckDB connection."""

        if self._connection:
            self._connection.close()

    def _initialise(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                scenario VARCHAR,
                created_at TIMESTAMP,
                policy JSON,
                config JSON,
                metadata JSON
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS timeline (
                run_id VARCHAR,
                tick INTEGER,
                tax_revenue DOUBLE,
                transfer_spending DOUBLE,
                budget_balance DOUBLE,
                unemployment_rate DOUBLE,
                employment_rate DOUBLE,
                consumption_total DOUBLE,
                consumption_mean DOUBLE,
                cpi DOUBLE,
                sentiment DOUBLE,
                labour_demand_ratio DOUBLE,
                active_events VARCHAR,
                event_economic_intensity DOUBLE,
                event_inflation_delta DOUBLE,
                policy_adjustment_multiplier DOUBLE
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agents (
                run_id VARCHAR,
                agent_id BIGINT,
                baseline_disposable DOUBLE,
                average_disposable DOUBLE,
                final_disposable DOUBLE,
                average_consumption DOUBLE,
                total_taxes DOUBLE,
                total_transfers DOUBLE,
                delta_disposable DOUBLE
            )
            """
        )

    def store(self, results: SimulationResults) -> str:
        """Persist a simulation result set and return the generated run identifier."""

        run_id = uuid4().hex
        created_at = datetime.utcnow()
        policy_payload = self._serialise_policy(results.policy)
        config_payload = asdict(results.config)
        metadata_payload = json.dumps(results.metadata)

        self._connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            [
                run_id,
                results.scenario_name,
                created_at,
                json.dumps(policy_payload),
                json.dumps(config_payload),
                metadata_payload,
            ],
        )

        timeline = results.timeline.copy()
        timeline.insert(0, "run_id", run_id)
        self._connection.register("timeline_df", timeline)
        self._connection.execute("INSERT INTO timeline SELECT * FROM timeline_df")

        agents = results.agent_metrics.copy()
        agents.insert(0, "run_id", run_id)
        self._connection.register("agents_df", agents)
        self._connection.execute("INSERT INTO agents SELECT * FROM agents_df")

        return run_id

    def fetch(self, run_id: str) -> SimulationResults:
        """Retrieve a simulation result previously stored."""

        row = self._connection.execute(
            "SELECT scenario, policy, config, metadata FROM runs WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            raise KeyError(f"Run '{run_id}' not found")

        scenario_name, policy_json, config_json, metadata_json = row
        policy = self._deserialise_policy(json.loads(policy_json))
        config = SimulationConfig(**json.loads(config_json))
        metadata = json.loads(metadata_json) if metadata_json else {}

        timeline = self._connection.execute(
            "SELECT * FROM timeline WHERE run_id = ? ORDER BY tick",
            [run_id],
        ).df()
        timeline = timeline.drop(columns=["run_id"])

        agents = self._connection.execute(
            "SELECT * FROM agents WHERE run_id = ? ORDER BY agent_id",
            [run_id],
        ).df()
        agents = agents.drop(columns=["run_id"])

        return SimulationResults(
            scenario_name=scenario_name,
            policy=policy,
            timeline=timeline,
            agent_metrics=agents,
            config=config,
            metadata=metadata,
        )

    def list_runs(self) -> pd.DataFrame:
        """Return metadata for stored runs."""

        frame = self._connection.execute(
            "SELECT run_id, scenario, created_at FROM runs ORDER BY created_at DESC"
        ).df()
        return frame

    def export(
        self,
        run_id: str,
        *,
        format: str = "parquet",
        reform_run_id: str | None = None,
    ) -> Dict[str, Path]:
        """Export results in the requested format.

        ``csv`` and ``parquet`` return the raw timeline/agent metrics, while
        ``html``/``pdf`` produce an analytical report. When exporting a
        comparison report, pass both ``run_id`` (base) and ``reform_run_id``.
        """

        allowed = {"csv", "parquet", "html", "pdf"}
        if format not in allowed:
            raise ValueError(f"Unsupported export format: {format}")

        results = self.fetch(run_id)
        paths: Dict[str, Path] = {}

        if format in {"csv", "parquet"}:
            timeline_path = self.export_dir / f"{run_id}_timeline.{format}"
            agents_path = self.export_dir / f"{run_id}_agents.{format}"

            if format == "csv":
                results.timeline.to_csv(timeline_path, index=False)
                results.agent_metrics.to_csv(agents_path, index=False)
            else:
                results.timeline.to_parquet(timeline_path, index=False)
                results.agent_metrics.to_parquet(agents_path, index=False)

            paths["timeline"] = timeline_path
            paths["agent_metrics"] = agents_path
            return paths

        from .reporting import build_html_report, build_pdf_report
        from .results import SimulationComparison

        reform_results = self.fetch(reform_run_id) if reform_run_id else None
        comparison = (
            SimulationComparison(base=results, reform=reform_results)
            if reform_results is not None
            else None
        )

        report_path = self.export_dir / f"{run_id}_report.{format}"
        if format == "html":
            html = build_html_report(results, reform_results, comparison)
            report_path.write_text(html, encoding="utf-8")
        else:
            build_pdf_report(report_path, results, reform_results, comparison)

        paths["report"] = report_path
        return paths

    @staticmethod
    def _serialise_policy(policy: PolicyParameters) -> Dict[str, object]:
        return {
            "tax_brackets": [
                {"threshold": bracket.threshold, "rate": bracket.rate}
                for bracket in sorted(policy.tax_brackets, key=lambda item: item.threshold)
            ],
            "base_deduction": policy.base_deduction,
            "child_subsidy": policy.child_subsidy,
            "unemployment_benefit": policy.unemployment_benefit,
        }

    @staticmethod
    def _deserialise_policy(payload: Mapping[str, object]) -> PolicyParameters:
        brackets = [
            TaxBracket(threshold=float(item["threshold"]), rate=float(item["rate"]))
            for item in payload.get("tax_brackets", [])
        ]
        return PolicyParameters(
            tax_brackets=brackets,
            base_deduction=float(payload.get("base_deduction", 0.0)),
            child_subsidy=float(payload.get("child_subsidy", 0.0)),
            unemployment_benefit=float(payload.get("unemployment_benefit", 0.0)),
        )
