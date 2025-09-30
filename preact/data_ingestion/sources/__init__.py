"""Data source implementations for PREACT."""
from .acled import ACLEDSource
from .base import DataSource, HTTPJSONSource, IngestionResult
from .economic import EconomicIndicatorSource
from .gdelt import GDELTSource
from .hdx import HDXSource
from .registry import SOURCE_REGISTRY, build_sources, fetch_all
from .unhcr import UNHCRSource

__all__ = [
    "ACLEDSource",
    "DataSource",
    "EconomicIndicatorSource",
    "GDELTSource",
    "HDXSource",
    "HTTPJSONSource",
    "IngestionResult",
    "SOURCE_REGISTRY",
    "UNHCRSource",
    "build_sources",
    "fetch_all",
]
