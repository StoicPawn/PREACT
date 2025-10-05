"""Streamlit module to explore GDELT events and the interaction graph."""

from __future__ import annotations

import io
import math
from typing import Any, Dict, Iterable, Tuple

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from preact.analytics.gdelt_graphs import StateGraph, build_state_graph
from preact.config import DataSourceConfig
from preact.data_ingestion.sources.gdelt import GDELTQuery, GDELTSource

DEFAULT_ENDPOINT = "https://api.gdeltproject.org/api/v2/events"
HIDDEN_COLUMNS = ["_actor1_country", "_actor2_country", "_event_id"]
TONE_POSITIVE_THRESHOLD = 1.0
TONE_NEGATIVE_THRESHOLD = -1.0


@st.cache_resource(show_spinner=False)
def _gdelt_source() -> GDELTSource:
    """Return a cached instance of the GDELT source client."""

    config = DataSourceConfig(
        name="GDELT",
        endpoint=DEFAULT_ENDPOINT,
        params={
            "mode": "Events",
            "format": "json",
            "maxrecords": "250",
            "sort": "DateDesc",
        },
    )
    return GDELTSource(config)


def _column_or_default(frame: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    """Return an existing column or a series filled with the default value."""

    if column in frame.columns:
        series = frame[column]
        if isinstance(series, pd.Series):
            return series
        return pd.Series(series, index=frame.index)
    if frame.empty:
        return pd.Series(dtype="object")
    return pd.Series([default] * len(frame), index=frame.index)


@st.cache_data(show_spinner=False, ttl=900)
def _load_events(
    *,
    lookback_days: int,
    limit: int,
    country: str | None,
    theme: str | None,
    keyword: str | None,
    tone_range: Tuple[float | None, float | None],
) -> tuple[pd.DataFrame, Dict[str, str]]:
    """Fetch recent GDELT events applying the provided filters."""

    source = _gdelt_source()
    tone_min, tone_max = tone_range
    query = GDELTQuery(
        keywords=(keyword,) if keyword else (),
        countries=(country,) if country else (),
        themes=(theme,) if theme else (),
        tone_min=tone_min,
        tone_max=tone_max,
    )
    result = source.recent_events(days=lookback_days, limit=limit, query=query)
    events = result.data.copy()
    metadata = dict(result.metadata)
    if not events.empty:
        events["event_date"] = pd.to_datetime(events.get("event_date"), errors="coerce")
        events["tone"] = pd.to_numeric(events.get("tone"), errors="coerce")
        events["goldstein"] = pd.to_numeric(events.get("goldstein"), errors="coerce")
        events["num_articles"] = pd.to_numeric(
            events.get("num_articles"), errors="coerce"
        ).fillna(0)
        events["actor1"] = _column_or_default(events, "actor1").fillna("")
        events["actor2"] = _column_or_default(events, "actor2").fillna("")
        events["themes"] = _column_or_default(events, "themes").fillna("")
        events["country"] = (
            _column_or_default(events, "country")
            .fillna("")
            .astype(str)
            .str.upper()
            .replace({"": "GLOBAL"})
        )
        events["actor1_country"] = (
            _column_or_default(events, "actor1_country")
            .fillna("")
            .astype(str)
            .str.upper()
            .replace({"": pd.NA})
        )
        events["actor2_country"] = (
            _column_or_default(events, "actor2_country")
            .fillna("")
            .astype(str)
            .str.upper()
            .replace({"": pd.NA})
        )
    return events, metadata


def _tone_bucket(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Non disponibile"
    if value >= 3:
        return "Molto positivo"
    if value >= 1:
        return "Positivo"
    if value <= -3:
        return "Molto negativo"
    if value <= -1:
        return "Negativo"
    return "Neutro"


def _prepare_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "Data",
                "Paese",
                "Interazione",
                "Tema principale",
                "Articoli",
                "Tone medio",
                "Sentiment",
                "Indice Goldstein",
                "Fonte",
            ]
        )

    frame = events.copy()
    frame["Data"] = frame.get("event_date").dt.strftime("%Y-%m-%d %H:%M")
    frame["Paese"] = frame.get("country").fillna("GLOBAL")
    frame["Attore 1"] = frame.get("actor1").replace({"": "Sconosciuto"})
    frame["Attore 2"] = frame.get("actor2").replace({"": "Sconosciuto"})
    frame["Interazione"] = frame["Attore 1"] + " ↔ " + frame["Attore 2"]
    frame["Tema principale"] = (
        frame.get("themes")
        .fillna("")
        .astype(str)
        .apply(lambda value: value.split(";")[0].strip() if value else "")
    )
    frame["Articoli"] = frame.get("num_articles").fillna(0).astype(int)
    frame["Tone medio"] = frame.get("tone").round(2)
    frame["Sentiment"] = frame["Tone medio"].apply(lambda v: _tone_bucket(v if pd.notna(v) else None))
    frame["Indice Goldstein"] = frame.get("goldstein").round(2)
    frame["Fonte"] = frame.get("source_url").replace({"": pd.NA})
    frame["_actor1_country"] = (
        events.get("actor1_country", pd.Series(index=events.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    frame["_actor2_country"] = (
        events.get("actor2_country", pd.Series(index=events.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    frame["_event_id"] = events.get("event_id")

    preferred = [
        "Data",
        "Paese",
        "Interazione",
        "Tema principale",
        "Articoli",
        "Tone medio",
        "Sentiment",
        "Indice Goldstein",
        "Fonte",
    ]
    return frame.loc[:, preferred]


def _filter_table(
    table: pd.DataFrame,
    *,
    countries: Iterable[str] | None,
    sentiment: Iterable[str] | None,
    search: str | None,
    tone_range: Tuple[float, float] | None,
) -> pd.DataFrame:
    frame = table.copy()
    if countries:
        normalized = [value.upper() for value in countries]
        actor1 = frame.get("_actor1_country", pd.Series(index=frame.index, dtype="object"))
        actor2 = frame.get("_actor2_country", pd.Series(index=frame.index, dtype="object"))
        mask = (
            frame["Paese"].isin(normalized)
            | actor1.isin(normalized)
            | actor2.isin(normalized)
        )
        frame = frame[mask]
    if sentiment:
        frame = frame[frame["Sentiment"].isin(list(sentiment))]
    if search:
        needle = search.strip().lower()
        if needle:
            mask = (
                frame["Interazione"].str.lower().str.contains(needle)
                | frame["Tema principale"].str.lower().str.contains(needle)
            )
            frame = frame[mask]
    if tone_range:
        low, high = tone_range
        frame = frame[(frame["Tone medio"] >= low) & (frame["Tone medio"] <= high)]
    return frame


def _prepare_download(table: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    export_table = table.drop(columns=[col for col in table.columns if col in HIDDEN_COLUMNS], errors="ignore")
    export_table.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _compute_layout(nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame(columns=["state", "x", "y", "total_events", "total_weight"])

    positions = nodes.copy().reset_index(drop=True)
    count = len(positions)
    angles = np.linspace(0, 2 * math.pi, count, endpoint=False)
    positions["x"] = np.cos(angles)
    positions["y"] = np.sin(angles)
    return positions


def _edge_segments(edges: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    if edges.empty or positions.empty:
        return pd.DataFrame(
            columns=["edge_id", "order", "x", "y", "source", "target", "events", "weight", "avg_tone"]
        )

    merged = edges.merge(
        positions.rename(columns={"state": "source"})[["source", "x", "y"]],
        on="source",
        how="left",
        suffixes=("", "_source"),
    ).rename(columns={"x": "x_source", "y": "y_source"})
    merged = merged.merge(
        positions.rename(columns={"state": "target"})[["target", "x", "y"]],
        on="target",
        how="left",
        suffixes=("", "_target"),
    ).rename(columns={"x": "x_target", "y": "y_target"})
    merged = merged.dropna(subset=["x_source", "y_source", "x_target", "y_target"])
    segments: list[dict[str, Any]] = []
    for idx, row in merged.iterrows():
        edge_id = f"{row['source']}→{row['target']}"
        segments.append(
            {
                "edge_id": edge_id,
                "order": 0,
                "x": row["x_source"],
                "y": row["y_source"],
                "source": row["source"],
                "target": row["target"],
                "events": row.get("events", 0),
                "weight": row.get("weight", 0.0),
                "avg_tone": row.get("avg_tone", np.nan),
            }
        )
        segments.append(
            {
                "edge_id": edge_id,
                "order": 1,
                "x": row["x_target"],
                "y": row["y_target"],
                "source": row["source"],
                "target": row["target"],
                "events": row.get("events", 0),
                "weight": row.get("weight", 0.0),
                "avg_tone": row.get("avg_tone", np.nan),
            }
        )
    return pd.DataFrame(segments)


def _prepare_country_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "events",
                "articles",
                "avg_tone",
                "positive",
                "negative",
                "latitude",
                "longitude",
            ]
        )

    frame = events.copy()
    frame["country"] = frame.get("country", pd.Series(dtype="object")).fillna("GLOBAL")
    frame["tone"] = pd.to_numeric(frame.get("tone"), errors="coerce")
    frame["num_articles"] = pd.to_numeric(frame.get("num_articles"), errors="coerce").fillna(0)
    frame["latitude"] = pd.to_numeric(frame.get("latitude"), errors="coerce")
    frame["longitude"] = pd.to_numeric(frame.get("longitude"), errors="coerce")

    grouped = (
        frame.groupby("country")
        .agg(
            events=("event_id", "nunique"),
            articles=("num_articles", "sum"),
            avg_tone=("tone", "mean"),
            positive=("tone", lambda series: int((series > TONE_POSITIVE_THRESHOLD).sum())),
            negative=("tone", lambda series: int((series < TONE_NEGATIVE_THRESHOLD).sum())),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.replace({np.inf: np.nan, -np.inf: np.nan})
    grouped = grouped.dropna(subset=["latitude", "longitude"], how="any")
    return grouped


def _tone_to_color(tone: float | None) -> list[int]:
    if tone is None or pd.isna(tone):
        return [180, 180, 180, 180]
    if tone >= TONE_POSITIVE_THRESHOLD:
        return [34, 139, 34, 220]
    if tone <= TONE_NEGATIVE_THRESHOLD:
        return [220, 53, 69, 220]
    return [255, 193, 7, 220]


def _build_arc_data(
    edges: pd.DataFrame, summary: pd.DataFrame, focus: str | None
) -> pd.DataFrame:
    if edges.empty or summary.empty or not focus:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "avg_tone",
                "weight",
                "events",
                "source_lat",
                "source_lon",
                "target_lat",
                "target_lon",
            ]
        )

    coords = summary.set_index("country")["latitude"].to_frame()
    coords["longitude"] = summary.set_index("country")["longitude"]
    focus_edges = edges[(edges["source"] == focus) | (edges["target"] == focus)].copy()
    if focus_edges.empty:
        return focus_edges

    focus_edges = focus_edges.join(
        coords.rename(columns={"latitude": "source_lat", "longitude": "source_lon"}),
        on="source",
    ).join(
        coords.rename(columns={"latitude": "target_lat", "longitude": "target_lon"}),
        on="target",
    )
    focus_edges = focus_edges.dropna(subset=["source_lat", "source_lon", "target_lat", "target_lon"])
    return focus_edges.reset_index(drop=True)


def _render_globe_map(
    container: DeltaGenerator,
    summary: pd.DataFrame,
    arcs: pd.DataFrame,
    *,
    focus_country: str | None,
) -> None:
    if summary.empty:
        container.info("Nessun dato geolocalizzato disponibile per i filtri correnti.")
        return

    display = summary.copy()
    display["color"] = display["avg_tone"].apply(_tone_to_color)
    display["radius"] = (
        display["articles"].fillna(0).astype(float).clip(lower=1.0) * 8000.0
    )
    if focus_country and focus_country in display["country"].values:
        display.loc[display["country"] == focus_country, "radius"] *= 1.4

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=display,
            get_position="[longitude, latitude]",
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
    ]

    if not arcs.empty:
        arc_data = arcs.copy()
        arc_data["color"] = arc_data["avg_tone"].apply(_tone_to_color)
        arc_data["width"] = arc_data.get("weight", 1.0).astype(float).clip(lower=1.0)
        layers.append(
            pdk.Layer(
                "ArcLayer",
                data=arc_data,
                get_source_position="[source_lon, source_lat]",
                get_target_position="[target_lon, target_lat]",
                get_source_color="color",
                get_target_color="color",
                get_width="width",
                pickable=True,
            )
        )

    view_state = pdk.ViewState(
        latitude=float(display["latitude"].mean()),
        longitude=float(display["longitude"].mean()),
        zoom=1.2,
        min_zoom=1,
        max_zoom=6,
    )
    tooltip = {
        "html": "<b>{country}</b><br/>Eventi: {events}<br/>Articoli: {articles}<br/>Tone medio: {avg_tone}",
        "style": {"backgroundColor": "#1f1f1f", "color": "white"},
    }
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip)
    container.pydeck_chart(deck, use_container_width=True)


def _render_graph(graph: StateGraph) -> None:
    if graph.edges.empty and graph.nodes.empty:
        st.info("Nessuna relazione tra stati da visualizzare con i filtri correnti.")
        return

    positions = _compute_layout(graph.nodes)
    segments = _edge_segments(graph.edges, positions)
    layers = []
    if not segments.empty:
        layers.append(
            alt.Chart(segments)
            .mark_line(opacity=0.35)
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                detail="edge_id:N",
                color=alt.Color("weight:Q", title="Peso (articoli)", scale=alt.Scale(scheme="blues")),
                tooltip=[
                    alt.Tooltip("source:N", title="Origine"),
                    alt.Tooltip("target:N", title="Destinazione"),
                    alt.Tooltip("events:Q", title="Eventi"),
                    alt.Tooltip("weight:Q", title="Articoli", format=",.0f"),
                    alt.Tooltip("avg_tone:Q", title="Tone medio", format=",.2f"),
                ],
            )
        )

    if not positions.empty:
        layers.append(
            alt.Chart(positions)
            .mark_circle()
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                size=alt.Size("total_weight:Q", title="Peso complessivo", scale=alt.Scale(range=[80, 1200])),
                color=alt.Color("total_events:Q", title="Eventi", scale=alt.Scale(scheme="reds")),
                tooltip=[
                    alt.Tooltip("state:N", title="Stato"),
                    alt.Tooltip("total_events:Q", title="Eventi"),
                    alt.Tooltip("total_weight:Q", title="Articoli", format=",.0f"),
                ],
            )
        )
        layers.append(
            alt.Chart(positions)
            .mark_text(baseline="middle", dx=0, dy=-12)
            .encode(x="x:Q", y="y:Q", text="state:N")
        )

    if not layers:
        st.info("Nessuna relazione tra stati da visualizzare con i filtri correnti.")
        return

    chart = alt.layer(*layers).properties(height=520)
    st.altair_chart(chart, use_container_width=True)


def render_gdelt_module(sidebar: DeltaGenerator) -> None:
    """Render the GDELT intelligence module inside the dashboard."""

    st.title("Modulo intelligence GDELT")
    st.caption(
        "Esplora gli ultimi eventi GDELT con filtri interattivi e una vista del grafo"
        " delle relazioni fra stati e attori."
    )

    sidebar.subheader("Parametri sorgente")
    lookback_days = sidebar.slider("Intervallo di analisi (giorni)", min_value=1, max_value=180, value=14)
    limit = sidebar.slider(
        "Numero massimo di eventi", min_value=10, max_value=250, value=150, step=10
    )
    country_filter = sidebar.text_input(
        "Paese (codice ISO3)",
        help="Filtra gli eventi principali per paese di riferimento (es. BFA, MLI)",
    ).strip().upper()
    theme_filter = sidebar.text_input(
        "Tema GDELT",
        help="Identificatore di tema GDELT, ad esempio 'TAX_FNCAID'",
    ).strip()
    keyword_filter = sidebar.text_input(
        "Parola chiave",
        help="Parola chiave da ricercare nel titolo/descrizione degli eventi",
    ).strip()
    tone_bounds = sidebar.slider(
        "Intervallo tone",
        min_value=-10.0,
        max_value=10.0,
        value=(-10.0, 10.0),
        step=0.5,
    )
    tone_min = None if tone_bounds[0] <= -10 else float(tone_bounds[0])
    tone_max = None if tone_bounds[1] >= 10 else float(tone_bounds[1])

    sidebar.subheader("Impostazioni grafo")
    min_events = sidebar.slider("Eventi minimi per arco", min_value=1, max_value=10, value=2)
    min_weight_slider = sidebar.slider(
        "Peso minimo (articoli)", min_value=0, max_value=50, value=5, step=1
    )
    min_weight = float(min_weight_slider) if min_weight_slider > 0 else None
    include_self_loops = sidebar.checkbox("Includi self-loop", value=False)

    with st.spinner("Recupero degli eventi GDELT..."):
        events, metadata = _load_events(
            lookback_days=lookback_days,
            limit=limit,
            country=country_filter or None,
            theme=theme_filter or None,
            keyword=keyword_filter or None,
            tone_range=(tone_min, tone_max),
        )

    total_events = int(len(events))
    distinct_countries = int(events.get("country", pd.Series(dtype=str)).nunique()) if total_events else 0
    avg_tone = float(events["tone"].mean()) if total_events and "tone" in events else float("nan")

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Eventi", f"{total_events:,}".replace(",", "."))
    kpi_cols[1].metric("Paesi coinvolti", f"{distinct_countries:,}".replace(",", "."))
    kpi_cols[2].metric(
        "Tone medio",
        f"{avg_tone:.2f}" if not math.isnan(avg_tone) else "n/d",
    )

    metadata_items = {
        "Finestra": f"{metadata.get('start_iso', '')} → {metadata.get('end_iso', '')}",
        "Query": metadata.get("query", ""),
        "Fallback": metadata.get("fallback", "false"),
    }
    with st.expander("Dettagli metadata", expanded=False):
        st.json(metadata_items)

    table = _prepare_table(events)

    map_container = st.container()
    with map_container:
        st.subheader("Mappa delle relazioni globali")

    country_options = (
        sorted(table["Paese"].dropna().unique().tolist()) if not table.empty else []
    )
    selected_country: str | None = None
    with map_container:
        if country_options:
            selector_options = ["Nessun filtro"] + country_options
            stored_selection = st.session_state.get("gdelt_selected_country")
            default_index = 0
            if stored_selection and stored_selection in country_options:
                default_index = selector_options.index(stored_selection)
            choice = st.selectbox(
                "Paese di interesse",
                options=selector_options,
                index=default_index,
                key="gdelt_map_focus",
            )
            selected_country = None if choice == "Nessun filtro" else choice
            if selected_country:
                st.session_state["gdelt_selected_country"] = selected_country
            else:
                st.session_state.pop("gdelt_selected_country", None)
        else:
            st.info("Nessun evento disponibile per la mappa.")

    st.subheader("Eventi recenti")
    filter_cols = st.columns(3)
    sentiment_options = sorted(table["Sentiment"].dropna().unique().tolist()) if not table.empty else []
    filter_key = "gdelt_country_filter"
    if filter_key not in st.session_state:
        st.session_state[filter_key] = []
    current_countries = [
        value for value in st.session_state.get(filter_key, []) if value in country_options
    ]
    if selected_country and selected_country not in current_countries:
        current_countries = sorted(set(current_countries + [selected_country]))
    st.session_state[filter_key] = current_countries
    selected_countries = filter_cols[0].multiselect(
        "Filtra per paese",
        options=country_options,
        default=current_countries,
        key=filter_key,
    )
    selected_sentiment = filter_cols[1].multiselect(
        "Sentiment", options=sentiment_options, default=[]
    )
    tone_filter = filter_cols[2].slider(
        "Range tone tabella",
        min_value=-10.0,
        max_value=10.0,
        value=(-2.0, 2.0),
        step=0.5,
    ) if not table.empty else None
    search_term = st.text_input(
        "Ricerca libera", value="", help="Cerca fra attori e temi principali"
    )

    filtered_table = _filter_table(
        table,
        countries=selected_countries,
        sentiment=selected_sentiment,
        search=search_term,
        tone_range=tone_filter if tone_filter else None,
    )

    filtered_events = (
        events.loc[filtered_table.index.unique()] if not filtered_table.empty else events.iloc[0:0]
    )
    display_table = filtered_table.drop(
        columns=[col for col in filtered_table.columns if col in HIDDEN_COLUMNS],
        errors="ignore",
    )
    st.dataframe(
        display_table,
        use_container_width=True,
        column_config={
            "Articoli": st.column_config.NumberColumn(format="%d"),
            "Tone medio": st.column_config.NumberColumn(format="%.2f"),
            "Indice Goldstein": st.column_config.NumberColumn(format="%.2f"),
            "Fonte": st.column_config.LinkColumn("Fonte", display_text="Apri"),
        },
        hide_index=True,
    )

    download_data = _prepare_download(display_table)
    st.download_button(
        "Scarica CSV",
        data=download_data,
        file_name="gdelt_eventi.csv",
        mime="text/csv",
        use_container_width=False,
    )

    summary = _prepare_country_summary(filtered_events)
    with map_container:
        graph_for_map = build_state_graph(
            filtered_events,
            weight_column="num_articles",
            min_events=min_events,
            min_weight=min_weight,
            include_self_loops=include_self_loops,
        )
        arcs = _build_arc_data(graph_for_map.edges, summary, selected_country)
        _render_globe_map(map_container, summary, arcs, focus_country=selected_country)

    st.subheader("Grafo interazioni fra stati")
    if filtered_events.empty:
        st.info("Nessun evento disponibile per costruire il grafo con i filtri correnti.")
        return

    graph = graph_for_map
    _render_graph(graph)

    if not graph.nodes.empty:
        st.markdown("#### Stati principali")
        st.dataframe(
            graph.nodes,
            use_container_width=True,
            column_config={
                "in_degree": st.column_config.NumberColumn("In-degree", format="%d"),
                "out_degree": st.column_config.NumberColumn("Out-degree", format="%d"),
                "degree": st.column_config.NumberColumn("Degree", format="%d"),
                "in_events": st.column_config.NumberColumn("Eventi in ingresso", format="%d"),
                "out_events": st.column_config.NumberColumn("Eventi in uscita", format="%d"),
                "total_events": st.column_config.NumberColumn("Totale eventi", format="%d"),
                "in_weight": st.column_config.NumberColumn("Peso in ingresso", format="%.0f"),
                "out_weight": st.column_config.NumberColumn("Peso in uscita", format="%.0f"),
                "total_weight": st.column_config.NumberColumn("Peso complessivo", format="%.0f"),
            },
            hide_index=True,
        )

    if not graph.edges.empty:
        st.markdown("#### Relazioni principali")
        st.dataframe(
            graph.edges,
            use_container_width=True,
            column_config={
                "events": st.column_config.NumberColumn("Eventi", format="%d"),
                "weight": st.column_config.NumberColumn("Peso", format="%.0f"),
                "avg_tone": st.column_config.NumberColumn("Tone medio", format="%.2f"),
                "avg_goldstein": st.column_config.NumberColumn("Goldstein medio", format="%.2f"),
            },
            hide_index=True,
        )
