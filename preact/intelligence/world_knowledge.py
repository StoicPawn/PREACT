"""Contracts and fail-closed policy for autonomous World Knowledge updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Optional, Sequence, Tuple


class KnowledgeUpdateKind(str, Enum):
    FACT = "fact"
    NARRATIVE = "narrative"
    INTERPRETATION = "interpretation"
    FORECAST = "forecast"


class KnowledgeDomain(str, Enum):
    GOVERNANCE = "governance"
    POLITICS = "politics"
    DIPLOMACY = "diplomacy"
    CONFLICT = "conflict"
    SOCIETY = "society"
    ECONOMY = "economy"
    HISTORY = "history"


class PromotionAction(str, Enum):
    AUTO_PROMOTE_FACT = "auto_promote_fact"
    HOLD_FOR_MORE_EVIDENCE = "hold_for_more_evidence"
    STORE_INTERPRETATION_ONLY = "store_interpretation_only"
    STORE_FORECAST_ONLY = "store_forecast_only"
    QUEUE_NARRATIVE_REGENERATION = "queue_narrative_regeneration"
    REJECT = "reject"


@dataclass(frozen=True)
class SourceEvidence:
    source: str
    source_ref: str
    published_at: datetime
    retrieved_at: datetime
    independent_group: str
    authoritative: bool = False
    excerpt_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source is required")
        if not self.source_ref.strip():
            raise ValueError("source_ref is required")
        if not self.independent_group.strip():
            raise ValueError("independent_group is required")
        if self.retrieved_at < self.published_at:
            # Retrieval before stated publication is almost certainly a timestamp error.
            raise ValueError("retrieved_at cannot precede published_at")


@dataclass(frozen=True)
class KnowledgeUpdateCandidate:
    entity_id: str
    field: str
    value: Any
    valid_from: datetime
    detected_at: datetime
    domain: KnowledgeDomain
    kind: KnowledgeUpdateKind
    confidence: float
    evidence: Tuple[SourceEvidence, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity_id is required")
        if not self.field.strip():
            raise ValueError("field is required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.detected_at < self.valid_from and self.kind is KnowledgeUpdateKind.FACT:
            # Facts may become known after they become valid, but not before their effective date
            # unless explicitly modelled as a scheduled future fact.
            raise ValueError("detected_at cannot precede valid_from for factual updates")
        object.__setattr__(self, "candidate_id", self.candidate_id or self.fingerprint())

    def fingerprint(self) -> str:
        payload = {
            "entity_id": self.entity_id,
            "field": self.field,
            "value": self.value,
            "valid_from": self.valid_from.isoformat(),
            "domain": self.domain.value,
            "kind": self.kind.value,
            "evidence": sorted((item.source_ref for item in self.evidence)),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return "wkc_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class PromotionPolicy:
    min_confidence: float = 0.75
    min_independent_source_groups: int = 2
    min_authoritative_sources: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.min_independent_source_groups < 1:
            raise ValueError("min_independent_source_groups must be >= 1")
        if self.min_authoritative_sources < 1:
            raise ValueError("min_authoritative_sources must be >= 1")


@dataclass(frozen=True)
class PromotionDecision:
    candidate_id: str
    action: PromotionAction
    reason: str
    independent_source_groups: int
    authoritative_sources: int
    confidence: float


def evaluate_candidate(
    candidate: KnowledgeUpdateCandidate,
    policy: PromotionPolicy = PromotionPolicy(),
) -> PromotionDecision:
    """Apply a fail-closed update policy.

    Facts can change the current world state only after corroboration. Interpretations,
    narratives and forecasts remain separate semantic objects.
    """

    groups = {item.independent_group for item in candidate.evidence}
    authoritative = sum(1 for item in candidate.evidence if item.authoritative)

    if candidate.kind is KnowledgeUpdateKind.FORECAST:
        return PromotionDecision(
            candidate_id=str(candidate.candidate_id),
            action=PromotionAction.STORE_FORECAST_ONLY,
            reason="Forecasts cannot mutate factual world state.",
            independent_source_groups=len(groups),
            authoritative_sources=authoritative,
            confidence=float(candidate.confidence),
        )

    if candidate.kind is KnowledgeUpdateKind.INTERPRETATION:
        return PromotionDecision(
            candidate_id=str(candidate.candidate_id),
            action=PromotionAction.STORE_INTERPRETATION_ONLY,
            reason="Interpretations are retained separately from factual assertions.",
            independent_source_groups=len(groups),
            authoritative_sources=authoritative,
            confidence=float(candidate.confidence),
        )

    if candidate.kind is KnowledgeUpdateKind.NARRATIVE:
        return PromotionDecision(
            candidate_id=str(candidate.candidate_id),
            action=PromotionAction.QUEUE_NARRATIVE_REGENERATION,
            reason="Narratives are regenerated from structured assertions; they do not mutate them.",
            independent_source_groups=len(groups),
            authoritative_sources=authoritative,
            confidence=float(candidate.confidence),
        )

    if candidate.confidence < policy.min_confidence:
        return PromotionDecision(
            candidate_id=str(candidate.candidate_id),
            action=PromotionAction.HOLD_FOR_MORE_EVIDENCE,
            reason="Candidate confidence is below the factual promotion threshold.",
            independent_source_groups=len(groups),
            authoritative_sources=authoritative,
            confidence=float(candidate.confidence),
        )

    corroborated = (
        authoritative >= policy.min_authoritative_sources
        or len(groups) >= policy.min_independent_source_groups
    )
    if not corroborated:
        return PromotionDecision(
            candidate_id=str(candidate.candidate_id),
            action=PromotionAction.HOLD_FOR_MORE_EVIDENCE,
            reason="Factual update lacks authoritative or independent corroboration.",
            independent_source_groups=len(groups),
            authoritative_sources=authoritative,
            confidence=float(candidate.confidence),
        )

    return PromotionDecision(
        candidate_id=str(candidate.candidate_id),
        action=PromotionAction.AUTO_PROMOTE_FACT,
        reason="Factual update passed confidence and corroboration gates.",
        independent_source_groups=len(groups),
        authoritative_sources=authoritative,
        confidence=float(candidate.confidence),
    )


def affected_narrative_sections(candidate: KnowledgeUpdateCandidate) -> Tuple[str, ...]:
    """Return narrative sections that should be regenerated after a promoted fact."""

    field_name = candidate.field.lower()
    sections = {"country_overview"}

    if candidate.domain in {KnowledgeDomain.GOVERNANCE, KnowledgeDomain.POLITICS}:
        sections.update({"political_system", "current_government", "recent_history"})
    if candidate.domain is KnowledgeDomain.DIPLOMACY:
        sections.update({"foreign_relations", "recent_history"})
    if candidate.domain is KnowledgeDomain.CONFLICT:
        sections.update({"security_conflict", "foreign_relations", "recent_history"})
    if candidate.domain is KnowledgeDomain.SOCIETY:
        sections.add("society")
    if candidate.domain is KnowledgeDomain.ECONOMY:
        sections.add("economy")
    if candidate.domain is KnowledgeDomain.HISTORY:
        sections.add("historical_timeline")

    if any(token in field_name for token in ("president", "prime_minister", "government", "coalition")):
        sections.update({"current_government", "political_system", "recent_history"})

    return tuple(sorted(sections))


__all__ = [
    "KnowledgeDomain",
    "KnowledgeUpdateCandidate",
    "KnowledgeUpdateKind",
    "PromotionAction",
    "PromotionDecision",
    "PromotionPolicy",
    "SourceEvidence",
    "affected_narrative_sections",
    "evaluate_candidate",
]
