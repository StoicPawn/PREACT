# PREACT data-source universe

This registry is the first reviewed source map for rebuilding PREACT as a genuine historical-geopolitical platform. It is intentionally stricter than a list of useful websites: a source must have a credible programmatic/bulk path or clear research reuse path, and its access/licensing constraints must be recorded.

## Rule zero: free access is not the same as free redistribution

PREACT must keep four things separate for every source:

- access rights;
- right to cache raw content;
- right to derive features/models;
- right to redistribute original content.

For news in particular, the default product should store metadata, timestamps, source URLs and derived event/entity features. Full article text is stored only where the source licence permits it.

## Rule one: every ingestion creates a vintage

For historical replay, `event_time` is not enough. Many official sources revise historical values. Therefore each raw acquisition is written to immutable storage with:

- `source_id`
- `retrieved_at`
- `source_release/version`
- checksum
- request/query parameters
- original URL
- licence snapshot/reference

Sources are assigned one of four replay policies:

- **event_time_native**: individual events have stable occurrence/publication times; raw records are still archived.
- **native_vintages**: the provider publishes versioned releases/archives; use those versions explicitly.
- **publication_release**: the source revises history at each edition; the exact published edition is part of the record.
- **snapshot_required**: the public API mainly exposes latest/revised values, so PREACT must preserve every raw pull.

## Temporal layers

### Layer A — live / contemporary world

**GDELT** is the main open global event/news signal backbone: large historical event files exist back to 1979 and the modern stream updates every 15 minutes.

**ACLED** supplies high-resolution political violence/protest events with country-specific coverage dates and authenticated API access.

**UCDP Candidate Events** gives a second, independently coded violence stream with monthly releases; the annual GED provides stable research vintages.

**Media Cloud** supplies a large searchable online-news archive and continuously collected media sources.

**Common Crawl** is the raw web fallback from 2008 onward. It is valuable for recovering pages and testing source survivorship, but underlying page copyright remains source-specific.

**Official RSS/API feeds** from governments, parliaments, central banks, international organisations and statistical agencies should form a separate primary-source collection rather than be blended invisibly with journalism.

### Layer B — 1945 to present

This is the richest period for rigorous country panels.

Core structured sources:

- UCDP conflict/violence
- ACLED political violence/protest
- V-Dem institutions and political variables
- QoG multi-source governance panel
- World Bank indicators
- IMF data
- UN Comtrade
- UNHCR displacement
- UN World Population Prospects
- FAOSTAT
- WHO Global Health Observatory / World Health Data Hub
- SIPRI military expenditure, arms transfers and related security databases
- EPR ethnic political power
- International IDEA elections/turnout
- Powell-Thyne coups
- KOF Globalisation Index
- Penn World Table

These should never be collapsed into one opaque country score. They populate separate latent/observed dimensions.

### Layer C — 1816 to 1945

**Correlates of War** is the backbone for the international system. PREACT should ingest each COW family independently:

- State System Membership
- Wars
- Militarized Interstate Disputes
- Formal Alliances
- National Material Capabilities
- Diplomatic Exchange
- Bilateral/National Trade
- Intergovernmental Organizations
- Direct Contiguity
- Colonial/Dependency Contiguity
- Territorial Change

**CShapes 2.0** supplies time-varying borders/capitals (1886-2019; Europe earlier), enabling historically correct maps.

**Maddison Project Database** supplies very long-run GDP/population estimates.

**V-Dem** reaches 1789 for many country units, providing an unusually deep institutional layer.

Historical press must be used as primary evidence rather than forcing modern structured indicators backwards.

### Layer D — before 1816 / deep history

**Seshat Global History Databank**: polity-level institutional, social-complexity and historical variables; replication datasets, snapshots and API.

**Clio Infra**: long-run demographic, economic and institutional indicators, often reaching back to 1500 depending on variable/region.

**Maddison Project Database**: parts of the dataset extend to year 1.

For this period PREACT should explicitly increase uncertainty and avoid pretending that annual modern nation-state panels exist where they do not.

## Historical press and source archives

A separate evidence layer is required because historians need to see the source material, not only model features.

- **Library of Congress Chronicling America** — machine-readable US historical newspapers; public API without a key.
- **Europeana Newspapers** — cross-European historical newspaper material, including raw/API access; object reuse rights vary.
- **BnF Gallica** — French and francophone historical collections with programmatic OCR (ALTO).
- **Delpher** — Dutch-language newspapers, 1618-1995; public-domain bulk OCR archive through 1879 and broader research access under conditions.
- **Trove** — Australian newspapers, gazettes and other collections through an API.
- **Guardian Open Platform** — useful modern editorial archive from 1999, but free tier is non-commercial.
- **Common Crawl** — archived web pages from 2008 onward.
- **Wikimedia projects** — open APIs/page history can add contextual timelines and entity-resolution evidence, but should never be treated as an authoritative primary source.

Future expansion should add national-library newspaper APIs country by country rather than scraping commercial archives.

## Geography and borders

Historical analysis must not geocode 1910 events against 2026 borders.

Priority geometry:
1. CShapes for historical state borders.
2. COW state membership/territorial-change data for political-entity validity.
3. provider-native coordinates for event datasets.
4. a canonical polity/entity crosswalk mapping ISO, COW, Gleditsch-Ward, V-Dem, UCDP, ACLED and historical entity IDs.

The entity crosswalk is a prerequisite for serious replay.

## Climate and natural shocks

- **Copernicus ERA5** — global reanalysis from 1940 onward; CC BY; weather/climate covariates.
- **NOAA IBTrACS** — global tropical cyclone tracks from the 1840s onward.
- **USGS ComCat** — earthquake event API.
- **GDACS** — free disaster-alert API for major hazards.

Natural shocks must be kept exogenous where appropriate in causal/scenario models rather than learned as political labels.

## Immediate ingestion order

### Wave 1 — backbone
1. COW
2. CShapes
3. V-Dem
4. World Bank
5. UCDP
6. GDELT
7. UNHCR
8. UN WPP
9. Maddison
10. SIPRI

This already gives PREACT a defensible 1816-present world model, with selected earlier series.

### Wave 2 — richer modern state
ACLED, QoG, UN Comtrade, IMF, FAOSTAT, WHO, EPR, IDEA, Powell-Thyne, KOF, PWT.

### Wave 3 — primary historical evidence
Chronicling America, Europeana Newspapers, Gallica, Delpher, Trove, Seshat, Clio Infra, Common Crawl, Media Cloud.

## What not to make core dependencies

Commercial or restrictive news sources such as Reuters, AP, Bloomberg, Financial Times and similar archives may be excellent evidence for a human user but should not be foundational ingestion dependencies without an explicit licence agreement.

Likewise, a public web page is not automatically a dataset PREACT may redistribute.

## Next technical milestone

Build a `SourceSnapshotStore` that writes immutable raw payloads plus retrieval metadata, then implement the first historical entity crosswalk and connectors for COW + CShapes + V-Dem. Only after that should model work resume.
