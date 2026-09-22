"""Mobile-first world intelligence landing page.

The first iteration deliberately keeps relationship semantics evidence-neutral: the UI
shows structural/economic/social indicators and exposes an explicit relationship layer
contract, but it does not infer political affinity from headlines without provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class CountryBrief:
    iso3: str
    name: str
    flag: str
    region: str
    population_m: float | None = None
    gdp_bn_usd: float | None = None
    unemployment_pct: float | None = None
    life_expectancy: float | None = None
    internet_pct: float | None = None


# Bootstrap catalogue. Values are intentionally sparse: unknown is preferable to a
# silently stale or fabricated observation. Live/versioned sources will replace these.
COUNTRIES: tuple[CountryBrief, ...] = (
    CountryBrief("ARG", "Argentina", "🇦🇷", "Americas"),
    CountryBrief("BRA", "Brazil", "🇧🇷", "Americas"),
    CountryBrief("CAN", "Canada", "🇨🇦", "Americas"),
    CountryBrief("CHN", "China", "🇨🇳", "Asia"),
    CountryBrief("DEU", "Germany", "🇩🇪", "Europe"),
    CountryBrief("EGY", "Egypt", "🇪🇬", "Africa"),
    CountryBrief("FRA", "France", "🇫🇷", "Europe"),
    CountryBrief("GBR", "United Kingdom", "🇬🇧", "Europe"),
    CountryBrief("IND", "India", "🇮🇳", "Asia"),
    CountryBrief("ISR", "Israel", "🇮🇱", "Asia"),
    CountryBrief("ITA", "Italy", "🇮🇹", "Europe"),
    CountryBrief("JPN", "Japan", "🇯🇵", "Asia"),
    CountryBrief("MEX", "Mexico", "🇲🇽", "Americas"),
    CountryBrief("NGA", "Nigeria", "🇳🇬", "Africa"),
    CountryBrief("RUS", "Russia", "🇷🇺", "Europe / Asia"),
    CountryBrief("SAU", "Saudi Arabia", "🇸🇦", "Asia"),
    CountryBrief("TUR", "Türkiye", "🇹🇷", "Europe / Asia"),
    CountryBrief("UKR", "Ukraine", "🇺🇦", "Europe"),
    CountryBrief("USA", "United States", "🇺🇸", "Americas"),
    CountryBrief("ZAF", "South Africa", "🇿🇦", "Africa"),
)


def _catalogue_frame(countries: Iterable[CountryBrief] = COUNTRIES) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Country": f"{c.flag} {c.name}", "Region": c.region, "ISO3": c.iso3} for c in countries]
    )


def _metric(label: str, value: float | None, suffix: str = "") -> None:
    st.metric(label, "Data pending" if value is None else f"{value:,.1f}{suffix}")


def render_world_explorer(sidebar) -> None:
    """Render the responsive entry point for country intelligence."""

    sidebar.caption("World intelligence · evidence-first")
    st.title("PREACT · World Explorer")
    st.caption(
        "Select a country to inspect economic, social and geopolitical evidence. "
        "Every live indicator will carry source, observation date and data vintage."
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

    selected_label = st.selectbox("Country", names, index=0)
    iso3 = catalogue.loc[catalogue["Country"] == selected_label, "ISO3"].iloc[0]
    country = next(item for item in COUNTRIES if item.iso3 == iso3)

    st.markdown(f"## {country.flag} {country.name}")
    st.caption(f"{country.region} · {country.iso3}")

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
        st.markdown("#### Relationship map")
        st.write(
            "The map layer will distinguish documented alliances/treaties from "
            "time-varying news-derived affinity or tension. News signals will never "
            "overwrite structural relationships and will expose confidence and recency."
        )
        st.warning(
            "Relationship scoring is not populated yet: PREACT will not fabricate "
            "alliances, sympathies or hostility when evidence is missing."
        )

    with sources:
        st.write(
            "Planned provenance per observation: source, source reference, observed-at, "
            "known-at, dataset version and point-in-time fingerprint."
        )

    st.divider()
    st.markdown("#### Countries")
    st.dataframe(catalogue[["Country", "Region"]], hide_index=True, use_container_width=True)
