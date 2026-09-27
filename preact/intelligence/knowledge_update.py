"""Orchestration for autonomous source-to-world-state updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from preact.history.world_knowledge_store import WorldKnowledgeStore
from preact.intelligence.world_knowledge import (
    KnowledgeUpdateCandidate,
    PromotionAction,
    PromotionDecision,
    PromotionPolicy,
    evaluate_candidate,
)


@dataclass(frozen=True)
class KnowledgeUpdateResult:
    candidate_id: str
    decision: PromotionDecision
    assertion_id: Optional[str] = None
    candidate_was_new: bool = True


class AutonomousKnowledgeUpdater:
    """Fail-closed bridge between extracted claims and versioned world state.

    Source collectors and future LLM extractors produce KnowledgeUpdateCandidate objects.
    This service does not fetch news and does not generate claims itself; it only governs
    whether a candidate may mutate factual world state.
    """

    def __init__(
        self,
        store: WorldKnowledgeStore,
        policy: PromotionPolicy = PromotionPolicy(),
    ) -> None:
        self.store = store
        self.policy = policy

    def process(self, candidate: KnowledgeUpdateCandidate) -> KnowledgeUpdateResult:
        """Persist one observation, aggregate its claim, then evaluate corroboration."""
        is_new = self.store.record_candidate(candidate)
        aggregated = self.store.aggregate_claim(candidate.claim_key())
        decision = evaluate_candidate(aggregated, self.policy)
        self.store.record_decision(decision)

        assertion_id: Optional[str] = None
        if decision.action is PromotionAction.AUTO_PROMOTE_FACT:
            assertion_id = self.store.promote_fact(aggregated, decision)

        return KnowledgeUpdateResult(
            candidate_id=str(aggregated.candidate_id),
            decision=decision,
            assertion_id=assertion_id,
            candidate_was_new=is_new,
        )


__all__ = ["AutonomousKnowledgeUpdater", "KnowledgeUpdateResult"]
