"""Reusable layout primitives for the PREACT Streamlit dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd
import streamlit as st


@dataclass
class AlertSummary:
    country: str
    probability: float
    trend: float
    sparkline: Iterable[float]

    @property
    def direction(self) -> str:
        if self.trend > 0.02:
            return "increasing"
        if self.trend < -0.02:
            return "decreasing"
        return "stable"


def predictions_to_frame(predictions: Dict[str, pd.Series]) -> pd.DataFrame:
    """Combine a dict of series into a unified dataframe for plotting."""

    frame = pd.DataFrame({name: series for name, series in predictions.items()})
    frame.index.name = "date"
    return frame


def compute_alert_summaries(predictions: Dict[str, pd.Series], window: int = 7) -> list[AlertSummary]:
    summaries: list[AlertSummary] = []
    for country, series in predictions.items():
        tail = series.tail(window)
        trend = float(tail.iloc[-1] - tail.iloc[0]) if len(tail) >= 2 else 0.0
        summaries.append(
            AlertSummary(
                country=country,
                probability=float(series.iloc[-1]),
                trend=trend,
                sparkline=series.tail(20).tolist(),
            )
        )
    summaries.sort(key=lambda alert: alert.probability, reverse=True)
    return summaries


def render_header(predictions: Dict[str, pd.Series]) -> None:
    latest = {name: float(series.iloc[-1]) for name, series in predictions.items()}
    highest_country = max(latest, key=latest.get)

    col1, col2, col3 = st.columns(3)
    col1.metric("Countries monitored", len(predictions))
    col2.metric("Highest alert probability", f"{latest[highest_country]*100:.1f}%", highest_country)

    combined = predictions_to_frame(predictions)
    rolling = combined.rolling(window=7).mean().iloc[-1].mean()
    col3.metric("7-day rolling mean", f"{rolling*100:.1f}%")


def render_alert_overview(predictions: Dict[str, pd.Series], threshold: float) -> None:
    st.subheader("Alert overview")
    summaries = compute_alert_summaries(predictions)
    alerts_df = pd.DataFrame(
        [
            {
                "Country": summary.country,
                "Probability": summary.probability,
                "Trend (Δ)": summary.trend,
                "Status": summary.direction,
            }
            for summary in summaries
        ]
    )
    alerts_df["Probability"] = alerts_df["Probability"].map(lambda value: f"{value*100:.1f}%")
    alerts_df["Trend (Δ)"] = alerts_df["Trend (Δ)"].map(lambda value: f"{value*100:.1f}%")
    alerts_df["Status"] = alerts_df["Status"].str.title()

    st.dataframe(alerts_df, width="stretch", hide_index=True)

    chart_data = predictions_to_frame(predictions)
    highlighted = chart_data[[summary.country for summary in summaries[:5]]]
    st.area_chart(highlighted, width="stretch")

    st.caption(
        f"Threshold set at {threshold*100:.0f}%. Alerts above this line require immediate review."
    )


def render_country_detail(predictions: Dict[str, pd.Series], default_country: str | None = None) -> None:
    st.subheader("Country deep dive")
    countries = sorted(predictions.keys())
    default = default_country or countries[0]
    country = st.selectbox("Select a country", options=countries, index=countries.index(default))
    series = predictions[country]

    cols = st.columns(2)
    cols[0].metric("Latest probability", f"{series.iloc[-1]*100:.1f}%")
    change = float(series.iloc[-1] - series.iloc[-8]) if len(series) > 7 else float(series.iloc[-1] - series.iloc[0])
    cols[1].metric("7-day change", f"{change*100:.1f}%")

    st.line_chart(series, width="stretch")

    st.markdown(
        """
        **Interpretation guidance**

        - Probabilities are model-derived and represent the chance of crisis escalation.
        - Sudden increases signal the need for analyst review and potential response planning.
        - Combine this signal with qualitative field reports before acting.
        """
    )


def render_diagnostics(
    predictions: Dict[str, pd.Series], outcomes: pd.Series | None, threshold: float, summary
) -> None:
    st.subheader("Model diagnostics")
    if outcomes is None:
        st.info("Upload observed outcomes to unlock precision and recall diagnostics.")
        return

    st.dataframe(summary, width="stretch")

    comparison = pd.DataFrame(
        {
            "Probability": {country: float(series.iloc[-1]) for country, series in predictions.items()},
            "Observed": outcomes.astype(float),
        }
    )
    st.bar_chart(comparison, width="stretch")

    st.caption(
        "The chart contrasts the latest model probabilities with observed outcomes to help calibrate the threshold."
        f" Current alert threshold: {threshold*100:.0f}%."
    )


def render_evidence(evidence: pd.DataFrame | None) -> None:
    st.subheader("Evidence feed")
    if evidence is None or evidence.empty:
        st.info("No Bayesian evidence available yet. Once generated it will appear here with signal strength cues.")
        return

    st.dataframe(evidence, width="stretch")


def render_dashboard(
    predictions: Dict[str, pd.Series],
    threshold: float,
    outcomes: pd.Series | None = None,
    summary: pd.DataFrame | None = None,
    evidence: pd.DataFrame | None = None,
) -> None:
    render_header(predictions)
    overview_tab, country_tab, diagnostics_tab, evidence_tab = st.tabs(
        ["Overview", "Country detail", "Diagnostics", "Evidence"]
    )

    with overview_tab:
        render_alert_overview(predictions, threshold)

    with country_tab:
        highest_country = max(predictions, key=lambda key: predictions[key].iloc[-1])
        render_country_detail(predictions, default_country=highest_country)

    with diagnostics_tab:
        if summary is not None:
            render_diagnostics(predictions, outcomes, threshold, summary)
        else:
            render_diagnostics(predictions, None, threshold, summary)

    with evidence_tab:
        render_evidence(evidence)


__all__ = [
    "render_dashboard",
    "render_alert_overview",
    "render_country_detail",
    "render_diagnostics",
    "render_evidence",
]

