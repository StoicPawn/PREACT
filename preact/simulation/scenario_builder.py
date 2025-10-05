"""Compatibility layer exposing the scenario builder under the documented name."""

from .scenario import (
    Scenario,
    ScenarioBuilder,
    PopulationParameters,
    FirmParameters,
)

__all__ = [
    "Scenario",
    "ScenarioBuilder",
    "PopulationParameters",
    "FirmParameters",
]
