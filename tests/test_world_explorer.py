from preact.dashboard.world_explorer import COUNTRIES, _catalogue_frame


def test_country_catalogue_has_unique_iso3_and_names():
    assert len(COUNTRIES) >= 20
    assert len({country.iso3 for country in COUNTRIES}) == len(COUNTRIES)
    assert len({country.name for country in COUNTRIES}) == len(COUNTRIES)
    assert all(len(country.iso3) == 3 and country.iso3.isupper() for country in COUNTRIES)
    assert all(country.flag and country.region for country in COUNTRIES)


def test_country_catalogue_frame_is_stable_and_selectable():
    frame = _catalogue_frame()
    assert list(frame.columns) == ["Country", "Region", "ISO3"]
    assert set(frame["ISO3"]) == {country.iso3 for country in COUNTRIES}
    assert frame["Country"].str.len().gt(3).all()
