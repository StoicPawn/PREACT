"""Reusable layout helpers for the fiscal MVP dashboard."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from preact.simulation.results import SimulationComparison, SimulationResults


def _format_value(value: float) -> str:
    """Format numeric values for KPI cards."""

    return f"{value:,.2f}".replace(",", ".")


def _scenario_label(results: SimulationResults | None, fallback: str) -> str:
    """Return a readable label for a simulation scenario."""

    if results is None:
        return fallback
    return results.scenario_name or fallback


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


def render_executive_panel(
    base: SimulationResults,
    *,
    reform: SimulationResults | None = None,
    comparison: Mapping[str, float] | None = None,
    takeaways: Mapping[str, Iterable[str] | str] | Iterable[str] | str | None = None,
) -> None:
    """Render headline KPIs and takeaway notes for the executive section."""

    base_kpis = base.kpis()
    reform_kpis = reform.kpis() if reform is not None else None
    metrics = (
        ("tax_revenue", "Gettito fiscale"),
        ("budget_balance", "Saldo PA"),
        ("employment_rate", "Tasso occupazione"),
        ("sentiment", "Sentiment medio"),
    )

    cols = st.columns(len(metrics))
    for column, (key, label) in zip(cols, metrics):
        value = base_kpis.get(key)
        if value is None:
            continue
        delta_value = None
        if comparison is not None and key in comparison:
            delta_value = comparison[key]
        elif reform_kpis is not None and key in reform_kpis:
            delta_value = float(reform_kpis[key]) - float(value)
        column.metric(label, _format_value(float(value)), delta=f"{delta_value:+.2f}" if delta_value is not None else None)

    st.markdown("#### Takeaways")
    normalized: Dict[str, list[str]] = {}
    if takeaways is not None:
        if isinstance(takeaways, Mapping):
            for label, entries in takeaways.items():
                if entries is None:
                    continue
                if isinstance(entries, str):
                    lines = [line.strip() for line in entries.splitlines() if line.strip()]
                else:
                    lines = [str(line).strip() for line in entries if str(line).strip()]
                if lines:
                    normalized[label] = lines
        elif isinstance(takeaways, str):
            lines = [line.strip() for line in takeaways.splitlines() if line.strip()]
            if lines:
                normalized[_scenario_label(base, "Scenario")] = lines
        else:
            entries = [str(line).strip() for line in takeaways if str(line).strip()]
            if entries:
                normalized[_scenario_label(base, "Scenario")] = entries

    if normalized:
        for label, messages in normalized.items():
            st.markdown(f"**{label}**")
            for message in messages:
                st.markdown(f"- {message}")
    else:
        placeholder_key = f"executive_takeaways_{base.scenario_name}"
        st.text_area(
            "Note di sintesi",
            key=placeholder_key,
            height=160,
            placeholder="Aggiungi osservazioni e takeaway principali...",
        )


def _decile_frame(
    *,
    data: Mapping[str, float],
    scenario_label: str,
) -> pd.DataFrame:
    """Convert a decile dictionary into a plotting frame."""

    if not data:
        return pd.DataFrame(columns=["Decile", "Scenario", "Valore"])

    def _rank(item: str) -> int:
        try:
            return int(item.split("_")[-1])
        except (ValueError, IndexError):  # pragma: no cover - defensive guard
            return 0

    ordered_keys = sorted(data.keys(), key=_rank)
    return pd.DataFrame(
        {
            "Decile": [f"Decile {int(key.split('_')[-1])}" for key in ordered_keys],
            "Scenario": scenario_label,
            "Valore": [float(data[key]) for key in ordered_keys],
        }
    )


def _decile_delta_frame(
    *,
    base: Mapping[str, float],
    reform: Mapping[str, float],
) -> pd.DataFrame:
    """Return a frame with delta between reform and base deciles."""

    def _rank(item: str) -> int:
        try:
            return int(item.split("_")[-1])
        except (ValueError, IndexError):  # pragma: no cover - defensive guard
            return 0

    all_keys = sorted({*base.keys(), *reform.keys()}, key=_rank)
    rows = []
    for key in all_keys:
        base_value = float(base.get(key, 0.0))
        reform_value = float(reform.get(key, base_value))
        delta = reform_value - base_value
        rows.append(
            {
                "Decile": f"Decile {int(key.split('_')[-1])}",
                "Delta": delta,
                "Segno": "positivo" if delta >= 0 else "negativo",
            }
        )
    return pd.DataFrame(rows)


def render_equity_section(base: SimulationResults, reform: SimulationResults | None = None) -> None:
    """Render consumption and sentiment decile charts for the equity section."""

    base_kpis = base.kpis()
    reform_kpis = reform.kpis() if reform is not None else None
    base_consumption = base_kpis.get("consumption_by_decile", {})
    base_sentiment = base_kpis.get("sentiment_by_decile", {})

    if not base_consumption and not base_sentiment:
        st.info("Dati per decili non disponibili per questo scenario.")
        return

    scenario_base = _scenario_label(base, "Base")
    scenario_reform = _scenario_label(reform, "Riforma")

    col1, col2 = st.columns(2)

    consumption_frame = _decile_frame(data=base_consumption, scenario_label=scenario_base)
    if reform_kpis is not None:
        reform_consumption = reform_kpis.get("consumption_by_decile", {})
        if reform_consumption:
            consumption_frame = pd.concat(
                [
                    consumption_frame,
                    _decile_frame(data=reform_consumption, scenario_label=scenario_reform),
                ],
                ignore_index=True,
            )
    if not consumption_frame.empty:
        chart = (
            alt.Chart(consumption_frame)
            .mark_bar()
            .encode(
                x=alt.X("Decile:N", sort=list(consumption_frame["Decile"].unique())),
                y=alt.Y("Valore:Q", title="Consumo medio"),
                color=alt.Color("Scenario:N"),
                tooltip=["Scenario:N", "Decile:N", alt.Tooltip("Valore:Q", format=",.2f")],
            )
            .properties(height=280, title="Consumo per decile")
        )
        with col1:
            st.altair_chart(chart, use_container_width=True)

    sentiment_frame = _decile_frame(data=base_sentiment, scenario_label=scenario_base)
    if reform_kpis is not None:
        reform_sentiment = reform_kpis.get("sentiment_by_decile", {})
        if reform_sentiment:
            sentiment_frame = pd.concat(
                [
                    sentiment_frame,
                    _decile_frame(data=reform_sentiment, scenario_label=scenario_reform),
                ],
                ignore_index=True,
            )
    if not sentiment_frame.empty:
        chart = (
            alt.Chart(sentiment_frame)
            .mark_bar()
            .encode(
                x=alt.X("Decile:N", sort=list(sentiment_frame["Decile"].unique())),
                y=alt.Y("Valore:Q", title="Sentiment"),
                color=alt.Color("Scenario:N"),
                tooltip=["Scenario:N", "Decile:N", alt.Tooltip("Valore:Q", format=",.2f")],
            )
            .properties(height=280, title="Sentiment per decile")
        )
        with col2:
            st.altair_chart(chart, use_container_width=True)

    if reform_kpis is not None:
        reform_consumption = reform_kpis.get("consumption_by_decile", {})
        reform_sentiment = reform_kpis.get("sentiment_by_decile", {})
        delta_columns = st.columns(2)
        if reform_consumption:
            delta_frame = _decile_delta_frame(base=base_consumption, reform=reform_consumption)
            chart = (
                alt.Chart(delta_frame)
                .mark_bar()
                .encode(
                    x=alt.X("Decile:N", sort=list(delta_frame["Decile"].unique())),
                    y=alt.Y("Delta:Q", title="Δ Consumo"),
                    color=alt.Color(
                        "Segno:N",
                        legend=None,
                        scale=alt.Scale(domain=["positivo", "negativo"], range=["#16a34a", "#dc2626"]),
                    ),
                    tooltip=["Decile:N", alt.Tooltip("Delta:Q", format=",.2f")],
                )
                .properties(height=220, title="Delta consumo (Riforma - Base)")
            )
            with delta_columns[0]:
                st.altair_chart(chart, use_container_width=True)
        if reform_sentiment:
            delta_frame = _decile_delta_frame(base=base_sentiment, reform=reform_sentiment)
            chart = (
                alt.Chart(delta_frame)
                .mark_bar()
                .encode(
                    x=alt.X("Decile:N", sort=list(delta_frame["Decile"].unique())),
                    y=alt.Y("Delta:Q", title="Δ Sentiment"),
                    color=alt.Color(
                        "Segno:N",
                        legend=None,
                        scale=alt.Scale(domain=["positivo", "negativo"], range=["#16a34a", "#dc2626"]),
                    ),
                    tooltip=["Decile:N", alt.Tooltip("Delta:Q", format=",.2f")],
                )
                .properties(height=220, title="Delta sentiment (Riforma - Base)")
            )
            with delta_columns[1]:
                st.altair_chart(chart, use_container_width=True)


def _timeline_long_frame(results: SimulationResults, metric: str, label: str) -> pd.DataFrame:
    """Return a tidy frame for an altair line chart."""

    frame = results.timeline[["tick", metric]].copy()
    frame.columns = ["Periodo", "Valore"]
    frame["Scenario"] = _scenario_label(results, label)
    return frame


def _macro_chart(
    base: SimulationResults,
    reform: SimulationResults | None,
    *,
    metric: str,
    title: str,
) -> alt.Chart:
    """Build a line chart comparing base and reform scenarios for macro indicators."""

    base_frame = _timeline_long_frame(base, metric, "Base")
    frames = [base_frame]
    if reform is not None:
        frames.append(_timeline_long_frame(reform, metric, "Riforma"))
    data = pd.concat(frames, ignore_index=True)
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("Periodo:Q", title="Tick"),
            y=alt.Y("Valore:Q", title=title),
            color=alt.Color("Scenario:N"),
            tooltip=["Scenario:N", alt.Tooltip("Periodo:Q", format=".0f"), alt.Tooltip("Valore:Q", format=",.2f")],
        )
        .properties(height=280, title=title)
    )


def render_macro_section(base: SimulationResults, reform: SimulationResults | None = None) -> None:
    """Display macro charts for employment, CPI and sentiment."""

    metrics = (
        ("employment_rate", "Tasso di occupazione"),
        ("cpi", "Indice CPI"),
        ("sentiment", "Sentiment medio"),
    )
    columns = st.columns(2)
    for (metric, title), container in zip(metrics[:2], columns):
        chart = _macro_chart(base, reform, metric=metric, title=title)
        with container:
            st.altair_chart(chart, use_container_width=True)
    chart = _macro_chart(base, reform, metric=metrics[2][0], title=metrics[2][1])
    st.altair_chart(chart, use_container_width=True)


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
    "render_executive_panel",
    "render_equity_section",
    "render_macro_section",
    "render_timeline",
    "render_winners_section",
    "render_downloads",
    "comparison_payload",
]

