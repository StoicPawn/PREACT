"""Compatibility layer exposing the economy core under the documented name."""

from .economy import EconomyCore, EconomyParameters, EconomyState, Shock

__all__ = ["EconomyCore", "EconomyParameters", "EconomyState", "Shock"]
