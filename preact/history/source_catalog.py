"""Reviewed source registry for PREACT historical/geopolitical ingestion.

The registry is deliberately descriptive: access and licence constraints are part of the
data model so connectors cannot silently treat "publicly visible" as "freely redistributable".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AccessClass = Literal[
    "open",
    "free_registration",
    "noncommercial",
    "mixed_rights",
    "research_request",
]
ReplayPolicy = Literal[
    "native_vintages",
    "snapshot_required",
    "event_time_native",
    "publication_release",
]
Priority = Literal["core", "high", "specialist"]


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    name: str
    domains: tuple[str, ...]
    temporal_coverage: str
    update_cadence: str
    access: AccessClass
    replay_policy: ReplayPolicy
    priority: Priority
    url: str
    licence_note: str = ""
    notes: str = ""


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "gdelt", "GDELT", ("news_events", "media"), "1979-present",
        "15 minutes", "open", "event_time_native", "core",
        "https://www.gdeltproject.org/",
        notes="Global event/news-derived signals; keep raw daily/15-minute files and source URLs."
    ),
    SourceSpec(
        "google_news_rss", "Google News RSS", ("media", "news_discovery"),
        "current/recent search feed", "continuous", "mixed_rights",
        "snapshot_required", "high", "https://news.google.com/",
        licence_note="RSS metadata is used for discovery; linked article content retains publisher-specific rights.",
        notes="Shared with GoldenBull. Persist feed XML/metadata and URLs, not unrestricted article bodies."
    ),
    SourceSpec(
        "mediacloud", "Media Cloud", ("media", "news_archive"), "source-dependent",
        "continuous", "open", "snapshot_required", "high",
        "https://www.mediacloud.org/",
        notes="Large online-news archive; coverage begins when a source entered its collection."
    ),
    SourceSpec(
        "commoncrawl", "Common Crawl", ("web_archive", "media"), "2008-present",
        "periodic crawls", "open", "native_vintages", "high",
        "https://commoncrawl.org/",
        licence_note="Crawl access is open; underlying page copyright remains source-specific."
    ),
    SourceSpec(
        "guardian_open_platform", "Guardian Open Platform", ("media",), "1999-present",
        "continuous", "noncommercial", "snapshot_required", "specialist",
        "https://open-platform.theguardian.com/",
        licence_note="Free developer access is for non-commercial use; commercial/text-mining use has separate terms."
    ),
    SourceSpec(
        "chronicling_america", "Library of Congress Chronicling America",
        ("historical_press",), "historic US newspapers", "archive", "open",
        "native_vintages", "high", "https://www.loc.gov/collections/chronicling-america/",
        notes="Public API/datasets; OCR and page-level resources."
    ),
    SourceSpec(
        "europeana_newspapers", "Europeana Newspapers",
        ("historical_press", "cultural_heritage"), "1618-1996 collection coverage",
        "archive", "mixed_rights", "native_vintages", "high",
        "https://www.europeana.eu/",
        licence_note="Metadata/API access is open; item reuse rights vary by provider/object."
    ),
    SourceSpec(
        "gallica", "BnF Gallica", ("historical_press", "books"), "historical",
        "archive", "mixed_rights", "native_vintages", "high",
        "https://gallica.bnf.fr/",
        notes="Provides APIs including OCR extraction in ALTO XML."
    ),
    SourceSpec(
        "delpher", "Delpher", ("historical_press",), "1618-1995 newspapers",
        "archive", "mixed_rights", "native_vintages", "high",
        "https://www.delpher.nl/",
        licence_note="Open bulk OCR is available for public-domain newspaper archive through 1879; later material has reuse constraints."
    ),
    SourceSpec(
        "trove", "National Library of Australia Trove", ("historical_press", "archives"),
        "historical", "archive", "free_registration", "native_vintages", "high",
        "https://trove.nla.gov.au/",
        notes="API supports newspapers/gazettes and other cultural collections."
    ),
    SourceSpec(
        "ucdp", "Uppsala Conflict Data Program", ("conflict", "violence"),
        "1946-present; georeferenced events 1989-present", "monthly/yearly",
        "free_registration", "native_vintages", "core", "https://ucdp.uu.se/",
        notes="Free token-protected API. Every call must pin a dataset version; UCDP documents versioned URLs as reproducible indefinitely."
    ),
    SourceSpec(
        "acled", "ACLED", ("conflict", "protest", "political_violence"),
        "country-dependent, 1997-present at longest", "near-real-time",
        "free_registration", "snapshot_required", "core", "https://acleddata.com/",
        licence_note="Account/authentication and ACLED EULA/attribution requirements apply."
    ),
    SourceSpec(
        "cow", "Correlates of War", ("war", "alliances", "diplomacy", "trade", "capabilities", "borders"),
        "mostly 1816 onward", "versioned releases", "noncommercial", "native_vintages",
        "core", "https://correlatesofwar.org/data-sets/",
        licence_note=(
            "COW terms prohibit commercial use and third-party redistribution "
            "without written permission; dataset-specific citation is required."
        ),
        notes="Use sub-datasets separately: wars, MIDs, alliances, NMC, diplomatic exchange, trade, IGO, contiguity, territorial change."
    ),
    SourceSpec(
        "vdem", "V-Dem", ("institutions", "democracy", "civil_society", "political_parties"),
        "many units 1789-present", "annual", "free_registration", "native_vintages",
        "core", "https://v-dem.net/data/",
        notes="Version archive is especially valuable for point-in-time replay."
    ),
    SourceSpec(
        "qog", "Quality of Government Institute", ("governance", "institutions", "country_panel"),
        "1946-present in standard TS", "annual", "noncommercial", "native_vintages",
        "high", "https://www.gu.se/en/quality-government/qog-data",
        licence_note="Free academic/non-commercial use; redistribution/commercial use restricted."
    ),
    SourceSpec(
        "world_bank", "World Bank Indicators API", ("macro", "development", "debt", "demography"),
        "many series 50+ years", "source-dependent", "open", "snapshot_required",
        "core", "https://api.worldbank.org/v2/",
        notes="No API key; revisions mean raw retrieval vintages must be stored."
    ),
    SourceSpec(
        "imf", "IMF Data", ("macro", "fiscal", "balance_of_payments", "financial"),
        "dataset-dependent", "dataset-dependent", "free_registration",
        "snapshot_required", "high", "https://data.imf.org/",
        notes="SDMX APIs; current portal may require an account for API exploration."
    ),
    SourceSpec(
        "oecd", "OECD Data Explorer", ("macro", "social", "trade", "institutions"),
        "dataset-dependent", "dataset-dependent", "open", "snapshot_required",
        "high", "https://data-explorer.oecd.org/",
        notes="Free SDMX API with rate limiting."
    ),
    SourceSpec(
        "eurostat", "Eurostat", ("macro", "social", "demography", "trade"),
        "dataset-dependent", "twice daily when updated", "open", "snapshot_required",
        "high", "https://ec.europa.eu/eurostat/",
        notes="API exposes latest dataset versions; source explicitly notes lack of historical versioning."
    ),
    SourceSpec(
        "un_comtrade", "UN Comtrade", ("trade",), "long-run, dataset-dependent",
        "monthly/annual", "free_registration", "snapshot_required", "core",
        "https://comtradeplus.un.org/",
        notes="Free account/API tier supports substantial programmatic access."
    ),
    SourceSpec(
        "unhcr", "UNHCR Refugee Data Finder", ("migration", "refugees", "humanitarian"),
        "historical to present", "annual/periodic", "open", "snapshot_required",
        "core", "https://www.unhcr.org/refugee-statistics/",
        notes="Open JSON API, no special credentials."
    ),
    SourceSpec(
        "un_wpp", "UN World Population Prospects", ("demography",),
        "1950-present estimates; projections to 2100", "revision releases",
        "open", "native_vintages", "core", "https://population.un.org/wpp/",
        notes="Bulk CSV and open API; keep revision identifier."
    ),
    SourceSpec(
        "faostat", "FAOSTAT", ("food", "agriculture", "land", "prices"),
        "1961-present", "periodic", "open", "snapshot_required", "high",
        "https://www.fao.org/faostat/",
        notes="Global country coverage and official API developer portal."
    ),
    SourceSpec(
        "who_gho", "WHO Global Health Observatory", ("health", "mortality", "disease"),
        "indicator-dependent", "periodic", "open", "snapshot_required", "high",
        "https://www.who.int/data/gho",
        notes="WHO estimates are revised; use the current World Health Data Hub interface when available."
    ),
    SourceSpec(
        "maddison", "Maddison Project Database", ("historical_macro", "population"),
        "1 AD-2022 for parts of dataset", "release-based", "open",
        "native_vintages", "core",
        "https://www.rug.nl/ggdc/historicaldevelopment/maddison/",
        licence_note="MPD 2023 is CC BY 4.0 with citation requirements."
    ),
    SourceSpec(
        "pwt", "Penn World Table", ("macro", "productivity", "capital", "labour"),
        "1950-2023", "release-based", "open", "native_vintages", "high",
        "https://www.rug.nl/ggdc/productivity/pwt/",
    ),
    SourceSpec(
        "clio_infra", "Clio Infra", ("historical_demography", "historical_economy", "institutions"),
        "often 1500 onward; variable-dependent", "archive/research updates",
        "open", "native_vintages", "high", "https://clio-infra.eu/",
    ),
    SourceSpec(
        "seshat", "Seshat Global History Databank", ("deep_history", "institutions", "social_complexity"),
        "deep historical; polity/variable-dependent", "snapshot/releases/API",
        "open", "native_vintages", "high", "https://seshatdatabank.info/data",
        notes="Replication datasets, periodic snapshots and API."
    ),
    SourceSpec(
        "sipri", "SIPRI Databases", ("military_spending", "arms_transfers", "embargoes", "peace_operations"),
        "military spending 1949-present; arms transfers 1950-present",
        "annual/periodic", "open", "publication_release", "core",
        "https://www.sipri.org/databases",
        notes="Historical values can be revised; archive every published edition used."
    ),
    SourceSpec(
        "epr", "Ethnic Power Relations", ("ethnicity", "political_power"),
        "1946-2021", "release-based", "open", "native_vintages", "high",
        "https://icr.ethz.ch/data/epr/core/",
    ),
    SourceSpec(
        "cshapes", "CShapes 2.0", ("historical_boundaries", "capitals", "geospatial"),
        "1886-2019; Europe from 1816", "release-based", "noncommercial",
        "native_vintages", "core", "https://icr.ethz.ch/data/cshapes/",
        licence_note="Dataset reuse is CC BY-NC-SA 4.0."
    ),
    SourceSpec(
        "powell_thyne_coups", "Powell & Thyne Coup Dataset", ("coups", "leadership_change"),
        "1950-present", "irregular updates", "mixed_rights", "native_vintages", "high",
        "https://jonathanmpowell.com/coups/",
        licence_note=(
            "Publicly downloadable academic dataset; no broad open-content licence "
            "is assumed. Preserve citation, provider link and exact vintage."
        ),
        notes="Current and archived published versions are available."
    ),
    SourceSpec(
        "idea_turnout", "International IDEA Voter Turnout Database", ("elections", "participation"),
        "1945-present", "event-driven", "open", "snapshot_required", "high",
        "https://www.idea.int/data-tools/data/voter-turnout-database",
    ),
    SourceSpec(
        "kof_globalisation", "KOF Globalisation Index", ("globalisation", "economic_links", "political_links"),
        "1970s-present", "annual", "open", "native_vintages", "high",
        "https://kof.ethz.ch/en/forecasts-and-indicators/indicators/kof-globalisation-index.html",
    ),
    SourceSpec(
        "era5", "Copernicus ERA5", ("climate", "weather"),
        "1940-present", "continuous/reanalysis", "open", "snapshot_required",
        "core", "https://cds.climate.copernicus.eu/",
        licence_note="ERA5 catalogue states CC BY 4.0.",
    ),
    SourceSpec(
        "ibtracs", "NOAA IBTrACS", ("tropical_cyclones", "natural_hazards"),
        "1840s-present", "annual/agency updates", "open", "native_vintages",
        "high", "https://www.ncei.noaa.gov/products/international-best-track-archive",
    ),
    SourceSpec(
        "usgs_earthquakes", "USGS Earthquake Catalog", ("earthquakes", "natural_hazards"),
        "catalog-dependent; long historical coverage", "real-time", "open",
        "event_time_native", "high", "https://earthquake.usgs.gov/fdsnws/event/1/",
    ),
    SourceSpec(
        "gdacs", "Global Disaster Alert and Coordination System", ("disasters", "hazards"),
        "recent/historical API coverage", "near-real-time", "open",
        "event_time_native", "high", "https://www.gdacs.org/",
        notes="Free API; attribution requested."
    ),
    SourceSpec(
        "bis", "Bank for International Settlements Statistics",
        ("banking", "credit", "debt", "exchange_rates", "property_prices"),
        "dataset-dependent", "periodic", "open", "snapshot_required", "core",
        "https://stats.bis.org/",
        notes="Official public SDMX REST API for BIS statistical data and metadata."
    ),
    SourceSpec(
        "ilostat", "ILOSTAT", ("labour", "employment", "wages", "working_conditions"),
        "country/indicator-dependent", "periodic", "open", "snapshot_required", "core",
        "https://ilostat.ilo.org/data/bulk/",
        notes="Programmatic bulk CSV by indicator and reference area with dictionaries/metadata."
    ),
    SourceSpec(
        "unesco_uis", "UNESCO Institute for Statistics",
        ("education", "science", "culture", "demography"),
        "indicator-dependent", "periodic", "open", "snapshot_required", "high",
        "https://databrowser.uis.unesco.org/resources",
        notes="Official API plus bulk CSV releases."
    ),
    SourceSpec(
        "pax", "PA-X Peace Agreements Database", ("peace_agreements", "ceasefires", "transitions"),
        "1990-present", "versioned releases", "open", "native_vintages", "core",
        "https://www.peaceagreements.org/downloads/",
        notes="Versioned CSV/Excel datasets and corpus; archives of previous releases support replay."
    ),
    SourceSpec(
        "prio_grid", "PRIO-GRID", ("spatial_panel", "socioeconomic", "environment", "conflict_covariates"),
        "1946-2014 (v2.0)", "release-based", "open", "native_vintages", "high",
        "https://www.prio.org/data/9",
        notes="Global 0.5-degree cell-year structure; useful as a historical spatial backbone but current release is old."
    ),
    SourceSpec(
        "fred_alfred", "FRED / ALFRED", ("macro", "finance", "realtime_vintages"),
        "series-dependent", "source-dependent", "open", "native_vintages", "high",
        "https://fred.stlouisfed.org/docs/api/fred/alfred.html",
        notes="ALFRED explicitly preserves real-time periods showing what values were known before later revisions."
    ),
    SourceSpec(
        "eia", "U.S. Energy Information Administration Open Data",
        ("energy", "oil", "gas", "electricity", "emissions"),
        "series-dependent; many series from 1960s", "monthly/annual/real-time",
        "free_registration", "snapshot_required", "high",
        "https://www.eia.gov/opendata/",
        notes="Free API key for API access; bulk files do not require a key."
    ),
    SourceSpec(
        "wid", "World Inequality Database", ("income", "wealth", "inequality"),
        "country/series-dependent; some very long-run series", "research updates",
        "open", "snapshot_required", "high", "https://wid.world/data/",
        notes="Direct dataset downloads plus replication packages/methodology."
    ),
    SourceSpec(
        "wvs", "World Values Survey", ("public_opinion", "values", "institutions", "society"),
        "1981-present", "wave-based", "noncommercial", "native_vintages", "high",
        "https://www.worldvaluessurvey.org/",
        licence_note="Free registration; non-profit use and no redistribution of data files."
    ),
    SourceSpec(
        "ipums_international", "IPUMS International", ("census_microdata", "demography", "households"),
        "mainly 1960-present plus historical censuses", "release-based",
        "research_request", "native_vintages", "specialist",
        "https://international.ipums.org/international/",
        licence_note="Free to qualified researchers; scholarly/educational use, individual registration, no redistribution."
    ),
    SourceSpec(
        "dhs", "The DHS Program", ("health", "demography", "households", "geospatial_surveys"),
        "survey/country-dependent", "survey releases", "free_registration",
        "native_vintages", "specialist", "https://www.dhsprogram.com/data/",
        licence_note="Survey microdata are distributed at no cost for legitimate academic research after registration/approval."
    ),
    SourceSpec(
        "openalex", "OpenAlex", ("scholarly_literature", "citations", "institutions", "topics"),
        "multi-century bibliographic coverage", "API/live; public snapshot quarterly",
        "open", "native_vintages", "core", "https://openalex.org/",
        licence_note="Metadata is released under CC0; full-text works retain their own licences.",
        notes="Free public complete snapshot plus API; ideal for historiography and research-evidence graph."
    ),
    SourceSpec(
        "crossref", "Crossref", ("scholarly_metadata", "citations", "retractions", "funding"),
        "publisher-dependent", "continuous", "open", "snapshot_required", "high",
        "https://api.crossref.org/",
        notes="Public REST API requires no signup; bibliographic metadata is broadly reusable."
    ),
    SourceSpec(
        "aiddata", "AidData", ("development_finance", "aid", "geospatial_development"),
        "dataset-dependent; core releases include 1947 onward", "release-based",
        "open", "native_vintages", "high", "https://www.aiddata.org/datasets",
        notes="Project-level and geocoded development-finance datasets plus free GeoQuery aggregation."
    ),
    SourceSpec(
        "geonames", "GeoNames", ("gazetteer", "place_names", "geocoding", "entity_resolution"),
        "current gazetteer with daily exports", "daily", "open", "snapshot_required", "core",
        "https://www.geonames.org/export/",
        licence_note="CC BY; free downloadable global gazetteer and web services."
    ),
    SourceSpec(
        "wikidata", "Wikidata", ("knowledge_graph", "entity_resolution", "biographical", "historical_context"),
        "multi-period knowledge graph", "continuous", "open", "snapshot_required", "high",
        "https://www.wikidata.org/wiki/Wikidata:Data_access",
        licence_note="Wikidata data is CC0; linked media/content may carry separate licences.",
        notes="Use for entity linking and crosswalk support, not as an authoritative primary historical source."
    ),
    SourceSpec(
        "osm_full_history", "OpenStreetMap Full History", ("geospatial", "infrastructure", "map_history"),
        "mostly October 2007-present", "weekly full-history dump", "open",
        "native_vintages", "specialist", "https://planet.openstreetmap.org/planet/full-history/",
        notes="Full-history dumps contain successive versions of OSM objects; very large and best used selectively."
    ),
    SourceSpec(
        "gtd", "Global Terrorism Database", ("terrorism", "political_violence"),
        "historical", "release-based", "noncommercial", "native_vintages", "specialist",
        "https://www.start.umd.edu/gtd/",
        licence_note="Use is licensed for non-commercial research/analysis; redistribution is restricted."
    ),

)


SOURCE_BY_ID = {source.source_id: source for source in SOURCES}
