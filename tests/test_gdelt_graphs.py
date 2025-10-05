from datetime import datetime

import pandas as pd
import pytest

from preact.analytics import build_state_graph


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "1",
                "event_date": datetime(2024, 3, 1),
                "actor1_country": "ITA",
                "actor2_country": "FRA",
                "num_articles": 4,
                "tone": -2.0,
                "goldstein": 1.5,
            },
            {
                "event_id": "2",
                "event_date": datetime(2024, 3, 2),
                "actor1_country": "FRA",
                "actor2_country": "DEU",
                "num_articles": 2,
                "tone": -1.0,
                "goldstein": 3.0,
            },
            {
                "event_id": "3",
                "event_date": datetime(2024, 3, 3),
                "actor1_country": "ITA",
                "actor2_country": "FRA",
                "num_articles": 1,
                "tone": -3.0,
                "goldstein": 2.0,
            },
        ]
    )


def test_build_state_graph_aggregates_edges() -> None:
    events = _sample_events()
    graph = build_state_graph(events, min_events=1)
    assert not graph.edges.empty
    italy_france = graph.edges.iloc[0]
    assert italy_france["source"] == "ITA"
    assert italy_france["target"] == "FRA"
    assert italy_france["events"] == 2
    assert italy_france["weight"] == 5
    assert italy_france["avg_tone"] == pytest.approx(-2.5)
    assert italy_france["first_seen"].date().isoformat() == "2024-03-01"
    assert not graph.nodes.empty
    italy_node = graph.nodes.loc[graph.nodes["state"] == "ITA"].iloc[0]
    assert italy_node["out_events"] == 2
    assert italy_node["total_weight"] == pytest.approx(5)


def test_build_state_graph_filters_by_thresholds() -> None:
    events = _sample_events()
    graph = build_state_graph(events, min_events=2, min_weight=4.5)
    assert len(graph.edges) == 1
    assert graph.edges.iloc[0]["source"] == "ITA"


def test_build_state_graph_handles_self_loops() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "4",
                "event_date": datetime(2024, 3, 4),
                "actor1_country": "ITA",
                "actor2_country": "ITA",
                "num_articles": 2,
            }
        ]
    )
    graph = build_state_graph(events, include_self_loops=False)
    assert graph.edges.empty
    graph_with_loops = build_state_graph(events, include_self_loops=True)
    assert len(graph_with_loops.edges) == 1
