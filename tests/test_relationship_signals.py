import pandas as pd
import pytest

from preact.analytics.relationship_signals import build_relationship_layer, relationship_rows


def test_relationship_layer_separates_structural_alliance_from_news_signal():
    edges = pd.DataFrame(
        [
            {
                "source": "ITA",
                "target": "FRA",
                "events": 20,
                "avg_goldstein": 6.0,
                "avg_tone": 4.0,
                "last_seen": "2026-09-20",
            },
            {
                "source": "DEU",
                "target": "ITA",
                "events": 10,
                "avg_goldstein": -5.0,
                "avg_tone": -3.0,
                "last_seen": "2026-09-21",
            },
        ]
    )
    signals = build_relationship_layer(
        edges,
        focal_iso3="ITA",
        as_of="2026-09-22",
        structural_allies=["FRA"],
    )

    assert signals["FRA"].structural_alliance is True
    assert signals["FRA"].status == "documented_alliance"
    assert signals["FRA"].score > 0
    assert signals["DEU"].structural_alliance is False
    assert signals["DEU"].score < 0
    assert signals["DEU"].status in {"tension", "conflict"}


def test_relationship_layer_rejects_future_aggregated_edges():
    edges = pd.DataFrame(
        [
            {
                "source": "ITA",
                "target": "FRA",
                "events": 3,
                "avg_goldstein": 1.0,
                "last_seen": "2026-09-23",
            }
        ]
    )

    with pytest.raises(ValueError, match="after as_of"):
        build_relationship_layer(edges, focal_iso3="ITA", as_of="2026-09-22")


def test_relationship_confidence_decays_and_rows_are_stable():
    edges = pd.DataFrame(
        [
            {
                "source": "ITA",
                "target": "FRA",
                "events": 12,
                "avg_goldstein": 4.0,
                "last_seen": "2026-09-01",
            }
        ]
    )
    recent = build_relationship_layer(
        edges,
        focal_iso3="ITA",
        as_of="2026-09-02",
        half_life_days=30,
    )
    old = build_relationship_layer(
        edges,
        focal_iso3="ITA",
        as_of="2026-12-01",
        half_life_days=30,
    )

    assert recent["FRA"].confidence > old["FRA"].confidence
    rows = relationship_rows(recent)
    assert rows[0]["iso3"] == "FRA"
    assert 0 <= rows[0]["confidence"] <= 1
