"""Reporting utilities for exporting simulation results as HTML and PDF."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .results import SimulationComparison, SimulationResults


def _fig_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _timeline_chart(
    metric: str,
    base: SimulationResults,
    reform: SimulationResults | None = None,
) -> tuple[str, plt.Figure]:
    label = metric.replace("_", " ").title()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(base.timeline["tick"], base.timeline[metric], label=f"Base – {base.scenario_name}")
    if reform is not None:
        ax.plot(
            reform.timeline["tick"],
            reform.timeline[metric],
            label=f"Reform – {reform.scenario_name}",
        )
    ax.set_xlabel("Tick (month)")
    ax.set_ylabel(label)
    ax.set_title(f"{label} timeline")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return label, fig


def _winners_table(results: SimulationResults) -> pd.DataFrame:
    winners = results.winners_losers()
    winners["cluster"] = winners["cluster"].map(lambda value: f"Cluster {int(value) + 1}")
    winners["mean_delta"] = winners["mean_delta"].map(lambda value: round(float(value), 2))
    return winners


def _kpi_pairs(
    base: Dict[str, float],
    comparison: Optional[Dict[str, float]] = None,
) -> Iterable[str]:
    tracked = [
        "tax_revenue",
        "transfer_spending",
        "budget_balance",
        "employment_rate",
        "unemployment_rate",
        "sentiment",
        "gini_post",
    ]
    labels = {
        "tax_revenue": "Gettito fiscale",
        "transfer_spending": "Spesa trasferimenti",
        "budget_balance": "Saldo PA",
        "employment_rate": "Tasso occupazione",
        "unemployment_rate": "Tasso disoccupazione",
        "sentiment": "Sentiment medio",
        "gini_post": "Indice di Gini (post)",
    }
    for key in tracked:
        label = labels[key]
        base_value = base.get(key)
        if base_value is None:
            continue
        if comparison and key in comparison:
            delta = comparison[key]
            yield f"<li><strong>{label}</strong>: {base_value:.2f} (Δ riforma {delta:+.2f})</li>"
        else:
            yield f"<li><strong>{label}</strong>: {base_value:.2f}</li>"


def build_html_report(
    base: SimulationResults,
    reform: SimulationResults | None = None,
    comparison: SimulationComparison | None = None,
) -> str:
    """Generate an HTML report for a base run and optional reform."""

    kpis = base.kpis()
    comp_payload = comparison.delta() if comparison else None

    charts: Dict[str, str] = {}
    for metric in ["tax_revenue", "budget_balance", "sentiment"]:
        label, fig = _timeline_chart(metric, base, reform)
        charts[label] = _fig_to_base64(fig)

    winners_table = _winners_table(base)
    winners_html = winners_table.to_html(index=False, classes="winners-table")

    reform_summary = (
        "<p>Confronto attivo tra scenario base e riforma fiscale.</p>"
        if reform is not None
        else "<p>Nessuna riforma aggiuntiva selezionata.</p>"
    )

    kpi_lines = "\n".join(_kpi_pairs(kpis, comp_payload))

    chart_images = "".join(
        f'<div class="chart"><img src="data:image/png;base64,{data}" alt="{label}"/></div>'
        for label, data in charts.items()
    )

    html = f"""
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <title>PREACT – Report fiscale</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #1f2933; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ background: #f3f4f6; padding: 20px; border-radius: 12px; margin-bottom: 30px; }}
    .charts {{ display: flex; flex-wrap: wrap; gap: 24px; }}
    .chart {{ flex: 1 1 320px; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #fff; }}
    .chart img {{ width: 100%; height: auto; }}
    .winners-table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    .winners-table th, .winners-table td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }}
    .winners-table th {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <h1>PREACT – Report fiscale</h1>
  <div class="summary">
    <h2>Executive summary</h2>
    <p>Scenario: <strong>{base.scenario_name}</strong></p>
    {reform_summary}
    <ul>
      {kpi_lines}
    </ul>
  </div>
  <h2>Timeline KPI</h2>
  <div class="charts">
    {chart_images}
  </div>
  <h2>Winners &amp; losers</h2>
  {winners_html}
</body>
</html>
"""
    return html


def build_pdf_report(
    target: Path,
    base: SimulationResults,
    reform: SimulationResults | None = None,
    comparison: SimulationComparison | None = None,
) -> None:
    """Persist a PDF report summarising the simulation results."""

    from matplotlib.backends.backend_pdf import PdfPages

    comparison_values = comparison.delta() if comparison else None

    with PdfPages(target) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait in inches
        ax.axis("off")
        lines = [
            "PREACT – Report fiscale",
            "",
            f"Scenario base: {base.scenario_name}",
        ]
        if reform is not None:
            lines.append(f"Scenario riforma: {reform.scenario_name}")
        lines.append("")
        for line in _kpi_pairs(base.kpis(), comparison_values):
            lines.append(_strip_html(line))
        text = "\n".join(lines)
        ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        for metric in ["tax_revenue", "budget_balance", "sentiment"]:
            _, fig = _timeline_chart(metric, base, reform)
            pdf.savefig(fig)
            plt.close(fig)

        winners = _winners_table(base)
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        table = ax.table(
            cellText=winners.values,
            colLabels=winners.columns,
            cellLoc="center",
            loc="center",
        )
        table.scale(1, 1.5)
        ax.set_title("Winners & losers", fontsize=12, pad=20)
        pdf.savefig(fig)
        plt.close(fig)


def _strip_html(value: str) -> str:
    """Remove simple HTML tags used in the KPI bullet list."""

    replacements = {
        "<li>": "- ",
        "</li>": "",
        "<strong>": "",
        "</strong>": "",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    return value


__all__ = ["build_html_report", "build_pdf_report"]

