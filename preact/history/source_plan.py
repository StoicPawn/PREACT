"""Ordered acquisition roadmap for PREACT's evidence universe."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannedSource:
    source_id:str
    wave:int
    order:int
    role:str
    dependencies:tuple[str,...]=()
    required_for:tuple[str,...]=("atlas",)


SOURCE_PLAN:tuple[PlannedSource,...]=(
    PlannedSource("geonames",0,1,"entity/code normalization",required_for=("atlas","replay","scenario")),
    PlannedSource("gdelt",0,2,"live global events/news",("geonames",),("atlas","replay")),
    PlannedSource("google_news_rss",0,3,"current news discovery",(),("atlas",)),
    PlannedSource("cow",1,1,"historical state system, alliances, disputes, capabilities",("geonames",),("atlas","replay","scenario")),
    PlannedSource("cshapes",1,2,"historical borders/capitals",("cow",),("atlas",)),
    PlannedSource("vdem",1,3,"long-run political institutions",("geonames",),("atlas","replay","scenario")),
    PlannedSource("world_bank",1,4,"modern macro/development",("geonames",),("atlas","replay","scenario")),
    PlannedSource("ucdp",1,5,"organized violence outcomes/events",("geonames",),("atlas","replay")),
    PlannedSource("unhcr",1,6,"displacement/humanitarian pressure",("geonames",),("atlas","replay","scenario")),
    PlannedSource("un_wpp",1,7,"population/demography",("geonames",),("atlas","replay","scenario")),
    PlannedSource("maddison",1,8,"long-run GDP/population estimates",("geonames",),("atlas","replay","scenario")),
    PlannedSource("sipri",1,9,"military expenditure/security capacity",("geonames",),("atlas","replay","scenario")),
    PlannedSource("powell_thyne_coups",1,10,"coup attempts/success outcomes",("geonames",),("atlas","replay")),\n    PlannedSource("fred_alfred",2,1,"real-time-vintage macro evidence",(),("replay","scenario")),
    PlannedSource("bis",2,2,"banking/credit/financial stress",(),("atlas","replay","scenario")),
    PlannedSource("imf",2,3,"fiscal/external/financial macro",(),("atlas","replay","scenario")),
    PlannedSource("un_comtrade",2,4,"bilateral trade dependencies",("geonames",),("atlas","replay","scenario")),
    PlannedSource("pax",2,5,"peace agreements/transitions",("geonames",),("atlas","replay")),
    PlannedSource("epr",2,6,"ethnic political power",("geonames",),("atlas","replay")),
    PlannedSource("ilostat",2,7,"labour market/social stress",("geonames",),("atlas","replay","scenario")),
    PlannedSource("faostat",2,8,"food/agriculture/resource stress",("geonames",),("atlas","replay","scenario")),
    PlannedSource("who_gho",2,9,"health/mortality shocks",("geonames",),("atlas","replay","scenario")),
    PlannedSource("era5",2,10,"climate/weather exogenous shocks",(),("replay","scenario")),
    PlannedSource("chronicling_america",3,1,"primary historical press",(),("atlas",)),
    PlannedSource("europeana_newspapers",3,2,"European historical press",(),("atlas",)),
    PlannedSource("gallica",3,3,"French/francophone OCR archive",(),("atlas",)),
    PlannedSource("delpher",3,4,"Dutch historical press",(),("atlas",)),
    PlannedSource("trove",3,5,"Australian historical press/archive",(),("atlas",)),
    PlannedSource("seshat",3,6,"deep historical polity structure",(),("atlas","scenario")),
    PlannedSource("clio_infra",3,7,"pre-modern socioeconomic indicators",(),("atlas","scenario")),
    PlannedSource("openalex",3,8,"historiography/research evidence graph",(),("atlas",)),
    PlannedSource("commoncrawl",3,9,"web archive fallback",(),("atlas",)),
    PlannedSource("mediacloud",3,10,"online media archive",(),("atlas",)),
)

PLAN_BY_SOURCE={item.source_id:item for item in SOURCE_PLAN}
