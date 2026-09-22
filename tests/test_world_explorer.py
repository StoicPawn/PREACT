import pandas as pd

from preact.analytics.relationship_signals import RelationshipSignal
from preact.dashboard.world_explorer import (
    COUNTRIES,
    _catalogue_frame,
    build_world_map,
    relationship_map_frame,
)


def test_country_catalogue_has_unique_iso3_and_names():
    assert len(COUNTRIES) >= 200
    assert len({country.iso3 for country in COUNTRIES}) == len(COUNTRIES)
    assert len({country.name for country in COUNTRIES}) == len(COUNTRIES)
    assert all(len(country.iso3) == 3 and country.iso3.isupper() for country in COUNTRIES)
    assert all(country.flag for country in COUNTRIES)


def test_country_catalogue_frame_is_stable_and_selectable():
    frame = _catalogue_frame()
    assert list(frame.columns) == ["Country", "ISO3"]
    assert set(frame["ISO3"]) == {country.iso3 for country in COUNTRIES}
    assert frame["Country"].str.len().gt(3).all()


def test_relationship_map_marks_selected_and_signal_countries():
    signals = {
        "FRA": RelationshipSignal(
            focal_iso3="ITA",
            counterpart_iso3="FRA",
            score=0.7,
            confidence=0.8,
            status="strong_cooperation",
            event_count=12,
            last_seen=pd.Timestamp("2026-09-20"),
        )
    }
    frame = relationship_map_frame("ITA", signals)
    italy = frame.loc[frame["iso3"] == "ITA"].iloc[0]
    france = frame.loc[frame["iso3"] == "FRA"].iloc[0]
    germany = frame.loc[frame["iso3"] == "DEU"].iloc[0]

    assert italy["status"] == "selected"
    assert france["status"] == "strong_cooperation"
    assert germany["status"] == "no_evidence"

    figure = build_world_map(frame)
    assert len(figure.data) >= 3
    locations = {location for trace in figure.data for location in trace.locations}
    assert {"ITA", "FRA", "DEU"}.issubset(locations)
