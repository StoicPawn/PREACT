"""Mobile-first world intelligence landing page.

The World Explorer is evidence-first: unknown data stay unknown, dynamic relationship
signals are explicitly separated from documented structural relationships, and every
future live observation is expected to retain point-in-time provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

import pandas as pd
import plotly.graph_objects as go
import pycountry
import streamlit as st

from preact.analytics.relationship_signals import (
    RelationshipSignal,
    build_relationship_layer,
)


@dataclass(frozen=True)
class CountryBrief:
    iso3: str
    name: str
    flag: str
    region: str = "Global"
    population_m: float | None = None
    gdp_bn_usd: float | None = None
    unemployment_pct: float | None = None
    life_expectancy: float | None = None
    internet_pct: float | None = None


def _flag(alpha2: str) -> str:
    code = str(alpha2).upper()
    if len(code) != 2 or not code.isalpha():
        return "🌐"
    return "".join(chr(127397 + ord(char)) for char in code)


def _country_catalogue() -> tuple[CountryBrief, ...]:
    rows: list[CountryBrief] = []
    for country in pycountry.countries:
        iso3 = getattr(country, "alpha_3", "")
        alpha2 = getattr(country, "alpha_2", "")
        name = getattr(country, "common_name", getattr(country, "name", iso3))
        if len(iso3) == 3:
            rows.append(CountryBrief(iso3=iso3, name=name, flag=_flag(alpha2)))
    return tuple(sorted(rows, key=lambda item: item.name))


COUNTRIES: tuple[CountryBrief, ...] = _country_catalogue()
COUNTRY_BY_ISO3 = {country.iso3: country for country in COUNTRIES}


def _catalogue_frame(countries: Iterable[CountryBrief] = COUNTRIES) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Country": f"{c.flag} {c.name}", "ISO3": c.iso3} for c in countries]
    )


def _metric(label: str, value: float | None, suffix: str = "") -> None:
    st.metric(label, "Data pending" if value is None else f"{value:,.1f}{suffix}")


_STATUS_LABEL = {
    "selected": "Selected country",
    "documented_alliance": "Documented alliance",
    "strong_cooperation": "Strong recent cooperation",
    "affinity": "Recent cooperative signal",
    "mixed": "Mixed interaction",
    "insufficient_evidence": "Insufficient evidence",
    "tension": "Recent tension",
    "conflict": "Recent conflict signal",
    "no_evidence": "No evidence",
}

_STATUS_COLOR = {
    "selected": "#F2C14E",
    "documented_alliance": "#2962FF",
    "strong_cooperation": "#2E7D32",
    "affinity": "#81C784",
    "mixed": "#B0BEC5",
    "insufficient_evidence": "#CFD8DC",
    "tension": "#FB8C00",
    "conflict": "#C62828",
    "no_evidence": "#ECEFF1",
}


def relationship_map_frame(
    focal_iso3: str,
    signals: Mapping[str, RelationshipSignal],
    countries: Iterable[CountryBrief] = COUNTRIES,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for country in countries:
        if country.iso3 == focal_iso3:
            status = "selected"
            score = 0.0
            confidence = 1.0
            events = 0
        else:
            signal = signals.get(country.iso3)
            status = signal.status if signal else "no_evidence"
            score = signal.score if signal else 0.0
            confidence = signal.confidence if signal else 0.0
            events = signal.event_count if signal else 0
        rows.append(
            {
                "iso3": country.iso3,
                "country": country.name,
                "flag": country.flag,
                "status": status,
                "status_label": _STATUS_LABEL[status],
                "score": score,
                "confidence": confidence,
                "event_count": events,
            }
        )
    return pd.DataFrame(rows)


def build_world_map(frame: pd.DataFrame) -> go.Figure:
    """Build a click-ready choropleth without external GeoJSON dependencies."""

    figure = go.Figure()
    order = [
        "no_evidence",
        "insufficient_evidence",
        "mixed",
        "affinity",
        "strong_cooperation",
        "documented_alliance",
        "tension",
        "conflict",
        "selected",
    ]
    for status in order:
        subset = frame.loc[frame["status"] == status]
        if subset.empty:
            continue
        customdata = subset[
            ["iso3", "country", "status_label", "score", "confidence", "event_count"]
        ].to_numpy()
        figure.add_trace(
            go.Choropleth(
                locations=subset["iso3"],
                z=[1] * len(subset),
                locationmode="ISO-3",
                colorscale=[
                    [0.0, _STATUS_COLOR[status]],
                    [1.0, _STATUS_COLOR[status]],
                ],
                showscale=False,
                marker_line_color="rgba(255,255,255,0.45)",
                marker_line_width=0.35,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "%{customdata[2]}<br>"
                    "Signal: %{customdata[3]:.2f}<br>"
                    "Confidence: %{customdata[4]:.0%}<br>"
                    "Events: %{customdata[5]}<extra></extra>"
                ),
                name=_STATUS_LABEL[status],
            )
        )
    figure.update_geos(
        projection_type="natural earth",
        showframe=False,
        showcoastlines=True,
        coastlinecolor="rgba(120,120,120,0.35)",
        bgcolor="rgba(0,0,0,0)",
    )
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        height=520,
        legend={"orientation": "h", "y": -0.04, "x": 0},
        geo={"fitbounds": False},
    )
    return figure


def _extract_selected_iso3(event: object) -> str | None:
    """Extract the ISO3 customdata value from a Streamlit Plotly selection event."""

    if event is None:
        return None
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, Mapping):
        selection = event.get("selection")
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, Mapping):
        points = selection.get("points")
    if not points:
        return None
    point = points[0]
    customdata = (
        point.get("customdata")
        if isinstance(point, Mapping)
        else getattr(point, "customdata", None)
    )
    if not customdata:
        return None
    iso3 = str(customdata[0]).upper()
    return iso3 if iso3 in COUNTRY_BY_ISO3 else None


def _relationship_edges_from_session() -> pd.DataFrame:
    value = st.session_state.get("world_relationship_edges")
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame()


def _render_mobile_css() -> None:
    st.markdown(
        """
        <style>
        @media (max-width: 700px) {
            .block-container {padding-top: 1rem; padding-left: .8rem; padding-right: .8rem;}
            div[data-testid="stMetric"] {padding: .25rem 0;}
            .stPlotlyChart {margin-left: -.5rem; margin-right: -.5rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_world_explorer(sidebar) -> None:
    """Render the responsive entry point for country intelligence."""

    _render_mobile_css()
    sidebar.caption("World intelligence · evidence-first")
    st.title("PREACT · World Explorer")
    st.caption(
        "Select a country from the list or the map. Dynamic colours represent recent "
        "evidence, not political judgments; structural alliances remain a separate layer."
    )

    query = st.text_input("Search countries", placeholder="Italy, Ukraine, Brazil…")
    catalogue = _catalogue_frame()
    if query.strip():
        mask = catalogue["Country"].str.contains(query.strip(), case=False, regex=False)
        catalogue = catalogue.loc[mask]

    names = catalogue["Country"].tolist()
    if not names:
        st.warning("No country matches the search.")
        return

    selected_iso3 = st.session_state.get("world_selected_iso3", "ITA")
    if selected_iso3 not in set(catalogue["ISO3"]):
        selected_iso3 = catalogue["ISO3"].iloc[0]
    selected_country = COUNTRY_BY_ISO3[selected_iso3]
    selected_label = f"{selected_country.flag} {selected_country.name}"
    index = names.index(selected_label) if selected_label in names else 0

    chosen_label = st.selectbox("Country", names, index=index, key="world_country_select")
    chosen_iso3 = catalogue.loc[catalogue["Country"] == chosen_label, "ISO3"].iloc[0]
    if chosen_iso3 != selected_iso3:
        selected_iso3 = chosen_iso3
        st.session_state["world_selected_iso3"] = selected_iso3

    edges = _relationship_edges_from_session()
    as_of = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    try:
        signals = build_relationship_layer(
            edges,
            focal_iso3=selected_iso3,
            as_of=as_of,
            min_events=1,
        )
    except ValueError as exc:
        st.error(f"Relationship layer rejected: {exc}")
        signals = {}

    map_frame = relationship_map_frame(selected_iso3, signals)
    event = st.plotly_chart(
        build_world_map(map_frame),
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="world_relationship_map",
    )
    clicked = _extract_selected_iso3(event)
    if clicked and clicked != selected_iso3:
        st.session_state["world_selected_iso3"] = clicked
        st.rerun()

    country = COUNTRY_BY_ISO3[selected_iso3]
    st.markdown(f"## {country.flag} {country.name}")
    st.caption(f"{country.iso3} · as of {as_of:%Y-%m-%d %H:%M} UTC")

    economic, social, relations, sources = st.tabs(
        ["Economy", "Society", "Relationships", "Sources"]
    )
    with economic:
        cols = st.columns(2)
        with cols[0]:
            _metric("Population", country.population_m, " M")
        with cols[1]:
            _metric("GDP", country.gdp_bn_usd, " B USD")
        _metric("Unemployment", country.unemployment_pct, "%")
        st.info("Versioned World Bank/official-series ingestion will populate this panel.")

    with social:
        cols = st.columns(2)
        with cols[0]:
            _metric("Life expectancy", country.life_expectancy, " years")
        with cols[1]:
            _metric("Internet use", country.internet_pct, "%")
        st.info("Social, demographic, governance and human-development series are next.")

    with relations:
        st.markdown("#### Relationship evidence")
        if not signals:
            st.info(
                "No relationship evidence is loaded in this runtime yet. The map remains "
                "neutral rather than inventing geopolitical relationships."
            )
        else:
            rows = map_frame.loc[
                ~map_frame["status"].isin(["no_evidence", "selected"]),
                ["flag", "country", "status_label", "score", "confidence", "event_count"],
            ].sort_values(["confidence", "event_count"], ascending=False)
            st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption(
            "News/event interactions use recency decay and confidence. A formal alliance "
            "can only be labelled from a structural source such as a treaty/alliance dataset."
        )

    with sources:
        st.write(
            "Required provenance per observation: source, source reference, observed-at, "
            "known-at, dataset version and point-in-time fingerprint."
        )
        st.write(
            "Current relationship runtime input is the world_relationship_edges session "
            "DataFrame using the PREACT state-interaction edge schema. Persistent hub "
            "integration follows."
        )

    st.divider()
    st.markdown("#### Countries")
    st.dataframe(catalogue[["Country"]], hide_index=True, use_container_width=True)
