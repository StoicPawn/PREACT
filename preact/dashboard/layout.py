"""Reusable layout helpers for the fiscal MVP dashboard."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from preact.simulation.results import SimulationComparison, SimulationResults


def render_policy_controls(
    *,
    label: str,
    policy,
    key_prefix: str,
    initial: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Render sliders for a policy configuration and return the payload."""

    st.subheader(label)
    initial_brackets = list(initial.get("brackets", [])) if initial else []
    bracket_payload: list[Dict[str, float]] = []

    columns = st.columns(max(len(policy.tax_brackets), 1))
    for index, bracket in enumerate(policy.tax_brackets):
        default_rate = float(bracket.rate)
        if index < len(initial_brackets):
            default_rate = float(initial_brackets[index].get("rate", default_rate))
        column = columns[index % len(columns)]
        rate = column.slider(
            f"Aliquota su soglia €{int(bracket.threshold):,}".replace(",", "."),
            min_value=0.0,
            max_value=0.6,
            value=default_rate,
            step=0.01,
            key=f"{key_prefix}_bracket_{index}",
        )
        bracket_payload.append({"threshold": float(bracket.threshold), "rate": float(rate)})

    base_default = float(initial.get("base_deduction", policy.base_deduction)) if initial else float(policy.base_deduction)
    child_default = float(initial.get("child_subsidy", policy.child_subsidy)) if initial else float(policy.child_subsidy)
    unemployment_default = (
        float(initial.get("unemployment_benefit", policy.unemployment_benefit))
        if initial
        else float(policy.unemployment_benefit)
    )

    base_deduction = st.slider(
        "Detrazione base per nucleo",
        min_value=0.0,
        max_value=12000.0,
        value=base_default,
        step=100.0,
        key=f"{key_prefix}_deduction",
    )
    child_subsidy = st.slider(
        "Sussidio mensile per figlio",
        min_value=0.0,
        max_value=400.0,
        value=child_default,
        step=10.0,
        key=f"{key_prefix}_child_subsidy",
    )
    unemployment_benefit = st.slider(
        "Sussidio disoccupazione",
        min_value=0.0,
        max_value=1800.0,
        value=unemployment_default,
        step=50.0,
        key=f"{key_prefix}_unemployment",
    )

    return {
        "brackets": bracket_payload,
        "base_deduction": float(base_deduction),
        "child_subsidy": float(child_subsidy),
        "unemployment_benefit": float(unemployment_benefit),
    }


def render_kpi_grid(base_kpis: Mapping[str, float], comparison: Mapping[str, float] | None = None) -> None:
    """Display KPI cards for base (and optional delta vs reform)."""

    cards: Sequence[tuple[str, str]] = (
        ("tax_revenue", "Gettito fiscale"),
        ("budget_balance", "Saldo PA"),
        ("unemployment_rate", "Tasso disoccupazione"),
        ("employment_rate", "Tasso occupazione"),
        ("sentiment", "Sentiment medio"),
        ("gini_post", "Indice di Gini"),
    )
    rows = [cards[i : i + 3] for i in range(0, len(cards), 3)]
    for row in rows:
        cols = st.columns(len(row))
        for column, (key, label) in zip(cols, row):
            base_value = base_kpis.get(key)
            if base_value is None:
                continue
            delta = None
            if comparison is not None and key in comparison:
                delta = comparison[key]
            column.metric(label, f"{base_value:,.2f}".replace(",", "."), delta=f"{delta:+.2f}" if delta is not None else None)


def render_timeline(metric: str, base: SimulationResults, reform: SimulationResults | None = None) -> None:
    """Render a timeline chart comparing base and reform."""

    label = metric.replace("_", " ").title()
    chart_data = pd.DataFrame({"Tick": base.timeline["tick"], "Base": base.timeline[metric]})
    if reform is not None:
        chart_data["Reform"] = reform.timeline[metric]
    chart_data = chart_data.set_index("Tick")
    st.line_chart(chart_data, height=260)
    st.caption(f"Serie temporale per {label} su orizzonte simulazione.")


def render_winners_section(base: SimulationResults, reform: SimulationResults | None = None) -> None:
    """Display winners/losers tables for the selected scenarios."""

    st.subheader("Winners & losers")
    base_winners = base.winners_losers()
    base_winners = base_winners.rename(columns={"mean_delta": "Δ reddito", "count": "Numero agenti"})
    st.dataframe(base_winners, hide_index=True, use_container_width=True)

    if reform is not None:
        reform_winners = reform.winners_losers().rename(
            columns={"mean_delta": "Δ reddito", "count": "Numero agenti"}
        )
        st.dataframe(reform_winners, hide_index=True, use_container_width=True)


def render_downloads(
    summary,
    repository,
    *,
    enable_reform: bool,
) -> None:
    """Expose download buttons for CSV/Parquet/HTML/PDF exports."""

    st.subheader("Export")
    base_run = summary.base_run_id
    reform_run = summary.reform_run_id if enable_reform else None

    csv_paths = repository.export(base_run, format="csv")
    parquet_paths = repository.export(base_run, format="parquet")
    html_report = repository.export(base_run, format="html", reform_run_id=reform_run)
    pdf_report = repository.export(base_run, format="pdf", reform_run_id=reform_run)

    col1, col2 = st.columns(2)
    with col1:
        for label, path in ("Timeline CSV", csv_paths["timeline"]), ("Agent CSV", csv_paths["agent_metrics"]):
            with open(path, "rb") as handle:
                st.download_button(label=f"Scarica {label}", data=handle.read(), file_name=path.name, mime="text/csv")
        for label, path in ("Timeline Parquet", parquet_paths["timeline"]), (
            "Agent Parquet",
            parquet_paths["agent_metrics"],
        ):
            with open(path, "rb") as handle:
                st.download_button(
                    label=f"Scarica {label}",
                    data=handle.read(),
                    file_name=path.name,
                    mime="application/octet-stream",
                )
    with col2:
        with open(html_report["report"], "rb") as handle:
            st.download_button(
                label="Report HTML",
                data=handle.read(),
                file_name=html_report["report"].name,
                mime="text/html",
            )
        with open(pdf_report["report"], "rb") as handle:
            st.download_button(
                label="Report PDF",
                data=handle.read(),
                file_name=pdf_report["report"].name,
                mime="application/pdf",
            )


def comparison_payload(base: SimulationResults, reform: SimulationResults | None) -> Mapping[str, float] | None:
    """Return a delta dictionary if the reform scenario is available."""

    if reform is None:
        return None
    return SimulationComparison(base=base, reform=reform).delta()


__all__ = [
    "render_policy_controls",
    "render_kpi_grid",
    "render_timeline",
    "render_winners_section",
    "render_downloads",
    "comparison_payload",
]

