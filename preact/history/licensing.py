"""Minimal policy guard for source access and raw redistribution."""

from __future__ import annotations

from dataclasses import dataclass

from .source_catalog import SourceSpec


@dataclass(frozen=True)
class DataUseContext:
    commercial: bool = False
    redistribute_raw: bool = False


@dataclass(frozen=True)
class DataUseDecision:
    allowed: bool
    reason: str


def evaluate_source_use(source: SourceSpec, context: DataUseContext) -> DataUseDecision:
    """Conservative guard derived from catalog access class.

    This is not legal advice. It prevents accidental treatment of restricted
    research sources as unrestricted product data until source-specific terms
    are explicitly reviewed.
    """

    if source.access == "open":
        return DataUseDecision(True, "catalogued as open; preserve attribution/licence metadata")

    if source.access in {"noncommercial", "research_request"} and context.commercial:
        return DataUseDecision(
            False,
            "source is catalogued for non-commercial/research use; commercial use needs separate review",
        )

    if source.access in {"noncommercial", "research_request", "free_registration"} and context.redistribute_raw:
        return DataUseDecision(
            False,
            "raw redistribution is not assumed for this access class; source-specific permission required",
        )

    if source.access == "mixed_rights" and context.redistribute_raw:
        return DataUseDecision(
            False,
            "item-level rights vary; raw redistribution requires per-item rights evaluation",
        )

    return DataUseDecision(
        True,
        "access may be usable under constraints; preserve account/licence provenance and review before release",
    )
