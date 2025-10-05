"""Streamlit dashboard delivering the Config → Run → Results fiscal workflow."""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlencode

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from preact.simulation import (
    SimulationComparison,
    SimulationRepository,
    SimulationRunSummary,
    SimulationService,
    default_templates,
)

from preact.dashboard.layout import (
    comparison_payload,
    render_equity_section,
    render_executive_panel,
    render_downloads,
    render_kpi_grid,
    render_macro_section,
    render_policy_controls,
    render_timeline,
    render_winners_section,
)

st.set_page_config(page_title="PREACT – Fiscal Sandbox", layout="wide")
st.title("PREACT – Fiscal policy sandbox")


def _init_service() -> SimulationService:
    if "simulation_service" not in st.session_state:
        storage_root = PROJECT_ROOT / "data"
        storage_root.mkdir(parents=True, exist_ok=True)
        repository = SimulationRepository(
            storage_root / "dashboard_runs.duckdb",
            export_dir=storage_root / "exports",
        )
        st.session_state["simulation_repository"] = repository
        st.session_state["simulation_service"] = SimulationService(repository=repository)
    return st.session_state["simulation_service"]


def _get_repository() -> SimulationRepository:
    service = _init_service()
    return st.session_state["simulation_repository"]


def _config_state() -> Dict[str, Any]:
    if "config" not in st.session_state:
        st.session_state["config"] = {}
    return st.session_state["config"]


def _read_query_params() -> Dict[str, list[str]]:
    try:
        params = st.query_params  # type: ignore[attr-defined]
    except AttributeError:
        return st.experimental_get_query_params()
    return {key: [value] for key, value in params.to_dict().items()}


def _write_query_params(params: Dict[str, str]) -> None:
    try:
        query_params = st.query_params  # type: ignore[attr-defined]
    except AttributeError:
        st.experimental_set_query_params(**params)
        return
    query_params.clear()
    query_params.update(params)


service = _init_service()
repository = _get_repository()
templates = {template.name: template for template in default_templates().values()}
state = _config_state()
query_params = _read_query_params()

config_tab, run_tab, results_tab = st.tabs(["Config", "Run", "Results"])


with config_tab:
    st.markdown(
        """
        Configura lo scenario di riferimento, seleziona un template e calibra i parametri
        fiscali tramite slider. Al salvataggio la configurazione viene resa disponibile
        nella scheda "Run".
        """
    )

    default_template_name = state.get("template", "Medium City")

    with st.form("scenario_config"):
        template_name = st.selectbox(
            "Template territoriale",
            options=list(templates.keys()),
            index=list(templates.keys()).index(default_template_name)
            if default_template_name in templates
            else 0,
        )
        template = templates[template_name]
        st.caption(template.description)

        horizon = st.slider(
            "Orizzonte simulazione (mesi)",
            min_value=6,
            max_value=36,
            value=int(state.get("horizon", template.horizon)),
            step=1,
        )
        seed = st.number_input(
            "Seed riproducibilità",
            min_value=1,
            max_value=9999,
            value=int(state.get("seed", 42)),
        )

        base_policy_defaults = state.get("base_policy") if state else None
        base_policy_payload = render_policy_controls(
            label="Scenario base",
            policy=template.policy,
            key_prefix="base",
            initial=base_policy_defaults,
        )

        enable_reform = st.checkbox(
            "Configura scenario di riforma",
            value=bool(state.get("reform_enabled", True)),
        )

        reform_policy_payload = None
        if enable_reform:
            reform_policy_defaults = state.get("reform_policy") if state else None
            reform_policy_payload = render_policy_controls(
                label="Scenario riforma",
                policy=template.policy,
                key_prefix="reform",
                initial=reform_policy_defaults,
            )

        submitted = st.form_submit_button("Salva configurazione", type="primary")
        if submitted:
            state.update(
                {
                    "template": template_name,
                    "horizon": horizon,
                    "seed": seed,
                    "base_policy": base_policy_payload,
                    "reform_policy": reform_policy_payload,
                    "reform_enabled": enable_reform,
                }
            )
            st.success("Configurazione aggiornata. Procedi alla scheda 'Run'.")


with run_tab:
    st.markdown(
        """
        Lancia l'esecuzione del motore di simulazione. Verranno generati automaticamente
        lo scenario base e, se configurato, quello di riforma.
        """
    )

    if not state:
        st.info("Configura uno scenario nella scheda precedente per abilitare il run.")
    else:
        st.write(
            "**Template selezionato:**",
            state.get("template", "Non definito"),
            "– orizzonte",
            state.get("horizon", "?"),
            "mesi",
        )

        if st.button("Esegui simulazione", type="primary"):
            progress_bar = st.progress(0, text="Preparazione scenario...")
            status_placeholder = st.empty()
            log_container = st.container()
            log_placeholder = log_container.empty()

            start_time = time.perf_counter()
            scenario_labels = ["Scenario base"]
            if state.get("reform_enabled") and state.get("reform_policy"):
                scenario_labels.append("Scenario riforma")
            horizon_value = int(state["horizon"])
            total_ticks = max(len(scenario_labels) * horizon_value, 1)

            progress_state: Dict[str, Any] = {
                "scenario_index": 0,
                "last_tick": None,
                "last_horizon": horizon_value,
            }
            log_history: list[str] = []

            def _on_progress(tick: int, horizon: int, messages: list[str]) -> None:
                if (
                    tick == 0
                    and progress_state["last_tick"] is not None
                    and progress_state["last_horizon"] is not None
                    and progress_state["last_tick"] == progress_state["last_horizon"] - 1
                ):
                    progress_state["scenario_index"] = min(
                        progress_state["scenario_index"] + 1,
                        len(scenario_labels) - 1,
                    )

                progress_state["last_tick"] = tick
                progress_state["last_horizon"] = horizon
                scenario_label = scenario_labels[progress_state["scenario_index"]]

                global_tick = progress_state["scenario_index"] * horizon_value + tick + 1
                completion = min(global_tick / total_ticks, 1.0)
                progress_bar.progress(
                    completion,
                    text=f"{scenario_label} – step {tick + 1}/{horizon}",
                )

                elapsed = time.perf_counter() - start_time
                eta_text = "--"
                if global_tick and global_tick < total_ticks:
                    rate = elapsed / global_tick
                    eta_seconds = max(int(rate * (total_ticks - global_tick)), 0)
                    eta_text = str(datetime.timedelta(seconds=eta_seconds))
                status_placeholder.caption(
                    f"Tempo trascorso: {elapsed:.1f}s · ETA: {eta_text}"
                )

                log_history.extend(messages)
                formatted_logs = "\n".join(f"• {message}" for message in log_history[-50:])
                log_placeholder.markdown(formatted_logs or "Inizializzazione...")

            with st.spinner("Esecuzione in corso..."):
                summary = service.run(
                    template_name=state["template"],
                    horizon=horizon_value,
                    seed=int(state.get("seed", 42)),
                    base_policy_payload=state.get("base_policy"),
                    policy_payload=state.get("reform_policy") if state.get("reform_enabled") else None,
                    metadata={"source": "streamlit"},
                    progress_callback=_on_progress,
                )

            progress_bar.progress(1.0, text="Simulazione completata")
            status_placeholder.caption("Esecuzione terminata")
            st.session_state["last_summary"] = summary

            st.success("Simulazione completata. Consulta i risultati nella scheda dedicata.")

            st.markdown("#### Riepilogo esecuzione")
            base_budget = summary.base_kpis.get("budget_balance") if summary.base_kpis else None
            base_sentiment = summary.base_kpis.get("sentiment") if summary.base_kpis else None
            st.write(
                "Scenario base",
                f"Run ID: `{summary.base_run_id}`",
                f"Saldo di bilancio: {base_budget:.2f}" if base_budget is not None else "",
                f"Sentiment: {base_sentiment:.2f}" if base_sentiment is not None else "",
            )
            if summary.reform_run_id and summary.reform_kpis:
                reform_budget = summary.reform_kpis.get("budget_balance")
                reform_sentiment = summary.reform_kpis.get("sentiment")
                st.write(
                    "Scenario riforma",
                    f"Run ID: `{summary.reform_run_id}`",
                    f"Saldo di bilancio: {reform_budget:.2f}" if reform_budget is not None else "",
                    f"Sentiment: {reform_sentiment:.2f}" if reform_sentiment is not None else "",
                )
            if summary.comparison:
                delta_budget = summary.comparison.get("budget_balance")
                delta_sentiment = summary.comparison.get("sentiment")
                st.caption(
                    "Δ Riforma vs Base – "
                    + ", ".join(
                        filter(
                            None,
                            [
                                f"Bilancio: {delta_budget:+.2f}" if delta_budget is not None else None,
                                f"Sentiment: {delta_sentiment:+.2f}" if delta_sentiment is not None else None,
                            ],
                        )
                    )
                )

            link_params = {"base": summary.base_run_id}
            if summary.reform_run_id:
                link_params["reform"] = summary.reform_run_id
            link_target = f"?{urlencode(link_params)}#results"
            st.link_button("Compare with Base", link_target, type="primary")

            _write_query_params(link_params)


with results_tab:
    st.markdown(
        """
        Analizza KPI, timeline A/B e scarica i report finali. Seleziona nuovamente la
        scheda "Config" per modificare le impostazioni e rilanciare la simulazione.
        """
    )

    summary = st.session_state.get("last_summary")
    if not summary and query_params.get("base"):
        base_run_id = query_params["base"][0]
        reform_values = query_params.get("reform")
        reform_run_id = reform_values[0] if reform_values else None
        base_results = service.fetch(base_run_id)
        reform_results = service.fetch(reform_run_id) if reform_run_id else None
        comparison = (
            SimulationComparison(base=base_results, reform=reform_results).delta()
            if reform_results
            else None
        )
        summary = SimulationRunSummary(
            base_run_id=base_run_id,
            base_kpis=base_results.kpis(),
            reform_run_id=reform_run_id,
            reform_kpis=reform_results.kpis() if reform_results else None,
            comparison=comparison,
        )
        st.session_state["last_summary"] = summary
    if not summary:
        st.info("Nessuna esecuzione disponibile. Avvia una simulazione dalla scheda 'Run'.")
    else:
        base_results = service.fetch(summary.base_run_id)
        reform_results = (
            service.fetch(summary.reform_run_id) if summary.reform_run_id else None
        )
        comparison = comparison_payload(base_results, reform_results)

        def _normalize_takeaway(value: object) -> object:
            if isinstance(value, dict):
                return [f"{key}: {val}" for key, val in value.items()]
            if isinstance(value, (list, tuple, set)):
                return list(value)
            if isinstance(value, str):
                return value
            return str(value)

        takeaways_payload: Dict[str, object] = {}
        base_takeaways = summary.base_kpis.get("takeaways") if summary.base_kpis else None
        if base_takeaways:
            takeaways_payload[base_results.scenario_name or "Scenario base"] = _normalize_takeaway(base_takeaways)
        if reform_results and summary.reform_kpis:
            reform_takeaways = summary.reform_kpis.get("takeaways")
            if reform_takeaways:
                takeaways_payload[reform_results.scenario_name or "Scenario riforma"] = _normalize_takeaway(reform_takeaways)

        executive_tab, equity_tab, macro_tab = st.tabs(["Executive", "Equità", "Macro"])

        with executive_tab:
            render_executive_panel(
                base_results,
                reform=reform_results,
                comparison=comparison,
                takeaways=takeaways_payload or None,
            )
            with st.expander("Altri KPI", expanded=False):
                render_kpi_grid(summary.base_kpis, comparison)

        with equity_tab:
            render_equity_section(base_results, reform_results)

        with macro_tab:
            render_macro_section(base_results, reform_results)
            st.divider()
            st.markdown("#### Finanza pubblica")
            fiscal_columns = st.columns(2)
            with fiscal_columns[0]:
                render_timeline("tax_revenue", base_results, reform_results)
            with fiscal_columns[1]:
                render_timeline("budget_balance", base_results, reform_results)

        st.divider()
        with st.expander("Winners & losers", expanded=False):
            render_winners_section(base_results, reform_results)

        with st.expander("Export risultati", expanded=True):
            render_downloads(summary, repository, enable_reform=bool(reform_results))


if __name__ == "__main__":  # pragma: no cover - entry point
    _init_service()
