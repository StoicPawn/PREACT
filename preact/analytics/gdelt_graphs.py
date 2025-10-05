"""Utilities to derive state interaction graphs from GDELT events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


def _normalise_country(series: pd.Series) -> pd.Series:
    values = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, None: pd.NA})
    )
    return values


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce") if series is not None else series


@dataclass(slots=True)
class StateGraph:
    """Container for the aggregated state-to-state interaction graph."""

    edges: pd.DataFrame
    nodes: pd.DataFrame

    def is_empty(self) -> bool:
        return self.edges.empty and self.nodes.empty


def _prepare_events(events: pd.DataFrame, weight_column: str) -> pd.DataFrame:
    frame = events.copy()
    for column in ("actor1_country", "actor2_country", "country"):
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = _normalise_country(frame[column])

    frame["event_date"] = _ensure_datetime(frame.get("event_date"))

    numeric_candidates: Iterable[str] = ("tone", "goldstein", weight_column)
    for column in numeric_candidates:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if weight_column in frame.columns:
        frame["weight"] = frame[weight_column].fillna(1.0)
    else:
        frame["weight"] = 1.0

    frame["source"] = frame["actor1_country"].combine_first(frame["country"])
    frame["target"] = frame["actor2_country"].combine_first(frame["country"])
    frame = frame.dropna(subset=["source", "target"])
    return frame


def _aggregate_edges(
    events: pd.DataFrame,
    *,
    min_events: int,
    min_weight: float | None,
    include_self_loops: bool,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "events",
                "weight",
                "avg_tone",
                "avg_goldstein",
                "first_seen",
                "last_seen",
            ]
        )

    frame = events.copy()
    if not include_self_loops:
        frame = frame[frame["source"] != frame["target"]]

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "events",
                "weight",
                "avg_tone",
                "avg_goldstein",
                "first_seen",
                "last_seen",
            ]
        )

    agg_kwargs: dict[str, tuple[str, str]] = {
        "weight": ("weight", "sum"),
        "first_seen": ("event_date", "min"),
        "last_seen": ("event_date", "max"),
    }

    if "event_id" in frame.columns:
        agg_kwargs["events"] = ("event_id", "nunique")
    else:
        agg_kwargs["events"] = ("weight", "count")

    if "tone" in frame.columns:
        agg_kwargs["avg_tone"] = ("tone", "mean")
    if "goldstein" in frame.columns:
        agg_kwargs["avg_goldstein"] = ("goldstein", "mean")

    grouped = frame.groupby(["source", "target"], dropna=False).agg(**agg_kwargs).reset_index()

    if min_events > 1 and "events" in grouped.columns:
        grouped = grouped[grouped["events"] >= min_events]

    if min_weight is not None and "weight" in grouped.columns:
        grouped = grouped[grouped["weight"] >= min_weight]

    grouped = grouped.sort_values(["weight", "events"], ascending=[False, False])
    return grouped.reset_index(drop=True)


def _summarise_nodes(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(
            columns=[
                "state",
                "in_degree",
                "out_degree",
                "degree",
                "in_events",
                "out_events",
                "total_events",
                "in_weight",
                "out_weight",
                "total_weight",
            ]
        )

    out_stats = (
        edges.groupby("source")
        .agg(out_degree=("target", "nunique"), out_events=("events", "sum"), out_weight=("weight", "sum"))
        .rename_axis("state")
    )
    in_stats = (
        edges.groupby("target")
        .agg(in_degree=("source", "nunique"), in_events=("events", "sum"), in_weight=("weight", "sum"))
        .rename_axis("state")
    )

    nodes = out_stats.join(in_stats, how="outer").fillna(0)
    nodes["degree"] = nodes.get("out_degree", 0) + nodes.get("in_degree", 0)
    nodes["total_events"] = nodes.get("out_events", 0) + nodes.get("in_events", 0)
    nodes["total_weight"] = nodes.get("out_weight", 0.0) + nodes.get("in_weight", 0.0)
    nodes = nodes.reset_index()
    preferred = [
        "state",
        "in_degree",
        "out_degree",
        "degree",
        "in_events",
        "out_events",
        "total_events",
        "in_weight",
        "out_weight",
        "total_weight",
    ]
    nodes = nodes.loc[:, preferred]
    nodes = nodes.sort_values("total_weight", ascending=False).reset_index(drop=True)
    return nodes


def build_state_graph(
    events: pd.DataFrame,
    *,
    weight_column: str = "num_articles",
    min_events: int = 1,
    min_weight: float | None = None,
    include_self_loops: bool = False,
) -> StateGraph:
    """Create a state interaction graph from GDELT events."""

    if events.empty:
        empty_edges = _aggregate_edges(
            events,
            min_events=min_events,
            min_weight=min_weight,
            include_self_loops=include_self_loops,
        )
        empty_nodes = _summarise_nodes(empty_edges)
        return StateGraph(edges=empty_edges, nodes=empty_nodes)

    prepared = _prepare_events(events, weight_column)
    edges = _aggregate_edges(
        prepared,
        min_events=min_events,
        min_weight=min_weight,
        include_self_loops=include_self_loops,
    )
    nodes = _summarise_nodes(edges)
    return StateGraph(edges=edges, nodes=nodes)


__all__ = ["StateGraph", "build_state_graph"]

