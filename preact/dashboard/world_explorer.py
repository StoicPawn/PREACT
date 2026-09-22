"""Mobile-first world intelligence landing page.

The World Explorer is evidence-first: unknown data stay unknown, dynamic relationship
signals are explicitly separated from documented structural relationships, and every
future live observation is expected to retain point-in-time provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
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
from preact.data_hub.gateway import SharedProviderGateway
from preact.intelligence.country_profile import IndicatorSnapshot, fetch_current_country_profile
from preact.intelligence.gdelt_relationships import (
    GDELTRelationshipBatch,
    load_recent_relationship_edges,
    relationship_evidence,
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

    relationship_batches = st.session_state.setdefault("world_relationship_batches", {})
    lookback_days = st.selectbox(
        "Relationship evidence window",
        options=[7, 14, 30, 60],
        index=1,
        format_func=lambda days: f"Last {days} days",
        key="world_relationship_lookback",
    )
    if st.button("Refresh recent relationship evidence", use_container_width=True):
        with st.spinner("Reading archived GDELT realtime snapshots…"):
            try:
                batch = load_recent_relationship_edges(
                    os.getenv("SHARED_DATA_HUB_ROOT", "data/shared_hub"),
                    lookback_days=int(lookback_days),
                    min_events=1,
                )
                relationship_batches["latest"] = batch
                st.session_state["world_relationship_edges"] = batch.edges
                if batch.snapshot_count == 0:
                    st.warning(
                        "No archived realtime GDELT event snapshots are available yet. "
                        "Start the shared collector on the PREACT lab first."
                    )
            except Exception as exc:
                st.error(f"Recent relationship evidence unavailable: {exc}")

    batch: GDELTRelationshipBatch | None = relationship_batches.get("latest")
    if batch is not None and batch.snapshot_count:
        st.caption(
            f"Relationship evidence: {batch.resolved_interaction_count:,} resolved "
            f"interactions from {batch.snapshot_count:,} archived GDELT snapshots · "
            f"strict ISO-3 coverage {batch.resolution_rate:.1%}"
        )

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

    profiles = st.session_state.setdefault("world_country_profiles", {})
    if st.button("Load / refresh current indicators", use_container_width=True):
        with st.spinner("Fetching versioned World Bank indicators…"):
            try:
                gateway = SharedProviderGateway(
                    os.getenv("SHARED_DATA_HUB_ROOT", "data/shared_hub")
                )
                profiles[selected_iso3] = fetch_current_country_profile(
                    selected_iso3,
                    gateway=gateway,
                )
            except Exception as exc:
                st.error(f"Country indicators unavailable: {exc}")
    profile: Mapping[str, IndicatorSnapshot] = profiles.get(selected_iso3, {})

    economic, social, relations, sources = st.tabs(
        ["Economy", "Society", "Relationships", "Sources"]
    )
    with economic:
        cols = st.columns(2)
        population = profile.get("population")
        gdp = profile.get("gdp")
        unemployment = profile.get("unemployment")
        with cols[0]:
            _metric("Population", population.display_value if population else None, " M")
        with cols[1]:
            _metric("GDP", gdp.display_value if gdp else None, " B USD")
        _metric(
            "Unemployment",
            unemployment.display_value if unemployment else None,
            "%",
        )
        if profile:
            years = sorted(
                {
                    item.year
                    for item in (population, gdp, unemployment)
                    if item is not None and item.year is not None
                }
            )
            st.caption(
                "Latest available World Bank observations"
                + (f" · years {years[0]}–{years[-1]}" if years else "")
            )
        else:
            st.info("Load current indicators to populate the economic brief.")

    with social:
        cols = st.columns(2)
        life = profile.get("life_expectancy")
        internet = profile.get("internet_use")
        urban = profile.get("urban_population")
        with cols[0]:
            _metric("Life expectancy", life.display_value if life else None, " years")
        with cols[1]:
            _metric("Internet use", internet.display_value if internet else None, "%")
        _metric("Urban population", urban.display_value if urban else None, "%")
        if not profile:
            st.info("Load current indicators to populate the social brief.")

    with relations:
        st.markdown("#### Relationship evidence")
        relationship_rows = map_frame.loc[
            ~map_frame["status"].isin(["no_evidence", "selected"]),
            ["iso3", "flag", "country", "status_label", "score", "confidence", "event_count"],
        ].sort_values(["confidence", "event_count"], ascending=False)
        if not signals:
            st.info(
                "No relationship evidence is loaded in this runtime yet. The map remains "
                "neutral rather than inventing geopolitical relationships."
            )
        else:
            st.dataframe(
                relationship_rows.drop(columns=["iso3"]),
                hide_index=True,
                use_container_width=True,
            )

        if batch is not None and not batch.events.empty:
            counterpart_options = ["All"]
            counterpart_options.extend(relationship_rows["iso3"].astype(str).tolist())
            selected_counterpart = st.selectbox(
                "Underlying event evidence",
                options=counterpart_options,
                format_func=lambda code: (
                    "All counterparts"
                    if code == "All"
                    else (
                        f"{COUNTRY_BY_ISO3[code].flag} {COUNTRY_BY_ISO3[code].name}"
                        if code in COUNTRY_BY_ISO3
                        else code
                    )
                ),
                key="world_relationship_counterpart",
            )
            evidence = relationship_evidence(
                batch,
                focal_iso3=selected_iso3,
                counterpart_iso3=None if selected_counterpart == "All" else selected_counterpart,
                limit=100,
            )
            if evidence.empty:
                st.caption("No event-level evidence for this selection.")
            else:
                evidence = evidence.copy()
                evidence["counterpart"] = evidence["counterpart_iso3"].map(
                    lambda code: (
                        f"{COUNTRY_BY_ISO3[code].flag} {COUNTRY_BY_ISO3[code].name}"
                        if code in COUNTRY_BY_ISO3
                        else code
                    )
                )
                evidence["event_date"] = pd.to_datetime(
                    evidence["event_date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d")
                display = evidence[
                    [
                        "event_date",
                        "counterpart",
                        "goldstein",
                        "tone",
                        "num_articles",
                        "source_url",
                    ]
                ].rename(
                    columns={
                        "event_date": "Date",
                        "counterpart": "Counterpart",
                        "goldstein": "Goldstein",
                        "tone": "Tone",
                        "num_articles": "Articles",
                        "source_url": "Source",
                    }
                )
                st.dataframe(
                    display,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Source": st.column_config.LinkColumn(
                            "Source",
                            display_text="open",
                        )
                    },
                )
                st.caption(
                    "These are provider event records behind the aggregate map signal; "
                    "open the source before drawing a substantive conclusion."
                )

        st.caption(
            "News/event interactions use recency decay and confidence. A formal alliance "
            "can only be labelled from a structural source such as a treaty/alliance dataset."
        )

    with sources:
        st.write(
            "Required provenance per observation: source, source reference, observed-at, "
            "known-at, dataset version and point-in-time fingerprint."
        )
        if profile:
            source_rows = []
            for key, snapshot in profile.items():
                source_rows.append(
                    {
                        "metric": snapshot.label,
                        "indicator": snapshot.code,
                        "year": snapshot.year,
                        "retrieved_at": snapshot.retrieved_at,
                        "snapshot": snapshot.snapshot_checksum,
                        "replay_safe_before_retrieval": False,
                    }
                )
            st.dataframe(pd.DataFrame(source_rows), hide_index=True, use_container_width=True)
            st.caption(
                "World Bank pulls are current-vintage snapshots. Historical replay must "
                "use snapshots that were actually known by the replay cutoff."
            )
        if batch is not None and batch.snapshot_count:
            st.markdown("#### Relationship snapshot provenance")
            st.write(
                {
                    "snapshot_count": batch.snapshot_count,
                    "raw_event_rows": batch.raw_event_count,
                    "resolved_interactions": batch.resolved_interaction_count,
                    "strict_iso3_resolution_rate": round(batch.resolution_rate, 4),
                    "newest_retrieved_at": batch.newest_retrieved_at,
                    "snapshot_checksums": [
                        checksum[:16] + "…" for checksum in batch.snapshot_checksums[-8:]
                    ],
                }
            )
        st.write(
            "Relationship runtime input is the world_relationship_edges session DataFrame "
            "built from immutable GDELT realtime snapshots in the shared data hub."
        )

    st.divider()
    st.markdown("#### Countries")
    st.dataframe(catalogue[["Country"]], hide_index=True, use_container_width=True)
