"""Simulation engine for PREACT MVP (v1).

This package implements the high-level architecture outlined in
``building_map.md``.  It provides a composable toolkit made of
``policy_core``, ``economy_core`` and ``sentiment_core`` modules together with
scenario building utilities and result summarisation helpers.
"""

from .engine import SimulationEngine, SimulationConfig
from .policy import PolicyCore, PolicyParameters, TaxBracket
from .economy import EconomyCore, EconomyParameters, EconomyState, Shock
from .sentiment import SentimentCore, SentimentWeights
from .scenario import (
    Scenario,
    ScenarioBuilder,
    PopulationParameters,
    FirmParameters,
)
from .results import SimulationResults, SimulationComparison
from .storage import SimulationRepository
from .service import SimulationService, SimulationRunSummary
from .templates import ScenarioTemplate, default_templates

__all__ = [
    "SimulationEngine",
    "SimulationConfig",
    "PolicyCore",
    "PolicyParameters",
    "TaxBracket",
    "EconomyCore",
    "EconomyParameters",
    "EconomyState",
    "Shock",
    "SentimentCore",
    "SentimentWeights",
    "Scenario",
    "ScenarioBuilder",
    "PopulationParameters",
    "FirmParameters",
    "SimulationResults",
    "SimulationComparison",
    "SimulationRepository",
    "SimulationService",
    "SimulationRunSummary",
    "ScenarioTemplate",
    "default_templates",
]
