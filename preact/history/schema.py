"""Canonical temporal and provenance schema for historical/geopolitical analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, Optional


class KnowledgeMode(str, Enum):
    """How point-in-time queries treat publication/knowledge time."""

    STRICT_AS_KNOWN = "strict_as_known"
    RETROSPECTIVE = "retrospective"


class EvidenceClass(str, Enum):
    """Semantic class of a record exposed to users and models."""

    OBSERVATION = "observation"
    DERIVED = "derived"
    ESTIMATE = "estimate"
    FORECAST = "forecast"
    COUNTERFACTUAL = "counterfactual"
    INTERPRETATION = "interpretation"


@dataclass(frozen=True)
class Provenance:
    """Source lineage for a single temporal record."""

    source: str
    source_ref: str
    retrieved_at: datetime
    dataset_version: Optional[str] = None
    licence: Optional[str] = None
    transform: Optional[str] = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class TemporalRecord:
    """A fact-like object with separate event and knowledge clocks.

    valid_from / valid_to describe when the record applies in the world.
    known_at describes when PREACT could legitimately know the record.

    Historical replay must filter on known_at to prevent hindsight leakage.
    """

    record_id: str
    entity_id: str
    variable: str
    value: Any
    valid_from: datetime
    known_at: datetime
    provenance: Provenance
    evidence_class: EvidenceClass = EvidenceClass.OBSERVATION
    valid_to: Optional[datetime] = None
    uncertainty: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def is_known_as_of(self, cutoff: datetime) -> bool:
        """Return whether the record was knowable by cutoff."""
        return self.known_at <= cutoff

    def is_valid_at(self, when: datetime) -> bool:
        """Return whether the record applies at a point in valid time."""
        if when < self.valid_from:
            return False
        return self.valid_to is None or when < self.valid_to


@dataclass(frozen=True)
class HistoricalQuery:
    """Point-in-time query specification.

    knowledge_cutoff is the anti-leakage boundary. valid_at allows a
    user to ask about a world-state time that may be earlier than the cutoff,
    while still limiting the evidence to what had become known by the cutoff.
    """

    knowledge_cutoff: datetime
    valid_at: Optional[datetime] = None
    entity_ids: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()
    evidence_classes: tuple[EvidenceClass, ...] = ()
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN

    def filter(self, records: Iterable[TemporalRecord]) -> list[TemporalRecord]:
        """Filter records according to temporal and semantic constraints."""
        valid_at = self.valid_at or self.knowledge_cutoff
        output: list[TemporalRecord] = []

        for record in records:
            if (
                self.knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN
                and not record.is_known_as_of(self.knowledge_cutoff)
            ):
                continue
            if not record.is_valid_at(valid_at):
                continue
            if self.entity_ids and record.entity_id not in self.entity_ids:
                continue
            if self.variables and record.variable not in self.variables:
                continue
            if self.evidence_classes and record.evidence_class not in self.evidence_classes:
                continue
            output.append(record)

        return output
