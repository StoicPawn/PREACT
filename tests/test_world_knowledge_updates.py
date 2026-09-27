from datetime import datetime, timezone

from preact.history.world_knowledge_store import WorldKnowledgeStore
from preact.intelligence.knowledge_update import AutonomousKnowledgeUpdater
from preact.intelligence.world_knowledge import (
    KnowledgeDomain,
    KnowledgeUpdateCandidate,
    KnowledgeUpdateKind,
    PromotionAction,
    SourceEvidence,
    evaluate_candidate,
)


def dt(day: int) -> datetime:
    return datetime(2026, 9, day, 12, tzinfo=timezone.utc)


def evidence(
    source: str,
    group: str,
    *,
    authoritative: bool = False,
    day: int = 27,
) -> SourceEvidence:
    return SourceEvidence(
        source=source,
        source_ref=f"https://example.test/{source}/{day}",
        published_at=dt(day),
        retrieved_at=dt(day),
        independent_group=group,
        authoritative=authoritative,
    )


def candidate(value, *, day=27, sources=(), confidence=0.9, kind=KnowledgeUpdateKind.FACT):
    return KnowledgeUpdateCandidate(
        entity_id="country:ITA",
        field="head_of_government",
        value=value,
        valid_from=dt(day),
        detected_at=dt(day),
        domain=KnowledgeDomain.GOVERNANCE,
        kind=kind,
        confidence=confidence,
        evidence=tuple(sources),
    )


def test_single_non_authoritative_story_cannot_rewrite_world_state():
    item = candidate("Person B", sources=(evidence("wire", "wire"),))
    decision = evaluate_candidate(item)
    assert decision.action is PromotionAction.HOLD_FOR_MORE_EVIDENCE


def test_authoritative_source_can_promote_high_confidence_fact():
    item = candidate(
        "Person B",
        sources=(evidence("official_gazette", "government", authoritative=True),),
    )
    decision = evaluate_candidate(item)
    assert decision.action is PromotionAction.AUTO_PROMOTE_FACT


def test_two_independent_sources_can_promote_fact():
    item = candidate(
        "Person B",
        sources=(
            evidence("wire_a", "agency_a"),
            evidence("paper_b", "publisher_b"),
        ),
    )
    assert evaluate_candidate(item).action is PromotionAction.AUTO_PROMOTE_FACT


def test_forecast_never_mutates_factual_state():
    item = candidate(
        "Person B",
        sources=(evidence("model", "preact"),),
        kind=KnowledgeUpdateKind.FORECAST,
    )
    assert evaluate_candidate(item).action is PromotionAction.STORE_FORECAST_ONLY


def test_promoted_change_closes_prior_assertion_and_queues_narratives(tmp_path):
    store = WorldKnowledgeStore(tmp_path / "world.duckdb")
    updater = AutonomousKnowledgeUpdater(store)

    first = candidate(
        "Person A",
        day=26,
        sources=(evidence("gazette_a", "government", authoritative=True, day=26),),
    )
    second = candidate(
        "Person B",
        day=27,
        sources=(evidence("gazette_b", "government", authoritative=True, day=27),),
    )

    first_result = updater.process(first)
    second_result = updater.process(second)

    assert first_result.assertion_id
    assert second_result.assertion_id
    current = store.current_state("country:ITA")
    assert current["head_of_government"]["value"] == "Person B"

    old_state = store.state_as_of("country:ITA", dt(26), known_cutoff=dt(26))
    assert old_state["head_of_government"]["value"] == "Person A"

    jobs = store.queued_narrative_jobs("country:ITA")
    sections = {job["section"] for job in jobs}
    assert {"country_overview", "current_government", "political_system", "recent_history"}.issubset(sections)


def test_low_confidence_candidate_is_stored_but_not_promoted(tmp_path):
    store = WorldKnowledgeStore(tmp_path / "world.duckdb")
    updater = AutonomousKnowledgeUpdater(store)

    item = candidate(
        "Person X",
        confidence=0.4,
        sources=(evidence("gazette", "government", authoritative=True),),
    )
    result = updater.process(item)

    assert result.decision.action is PromotionAction.HOLD_FOR_MORE_EVIDENCE
    assert result.assertion_id is None
    assert store.current_state("country:ITA") == {}


def test_independent_reports_across_cycles_corroborate_same_claim(tmp_path):
    store = WorldKnowledgeStore(tmp_path / "corroboration.duckdb")
    updater = AutonomousKnowledgeUpdater(store)

    valid_from = datetime(2026, 9, 27, 8, 0, tzinfo=timezone.utc)
    first_seen = datetime(2026, 9, 27, 9, 0, tzinfo=timezone.utc)
    second_seen = datetime(2026, 9, 27, 9, 15, tzinfo=timezone.utc)

    first = KnowledgeUpdateCandidate(
        entity_id="country:ITA",
        field="head_of_government",
        value="Person B",
        valid_from=valid_from,
        detected_at=first_seen,
        domain=KnowledgeDomain.GOVERNANCE,
        kind=KnowledgeUpdateKind.FACT,
        confidence=0.9,
        evidence=(
            SourceEvidence(
                source="wire_a",
                source_ref="https://a.example/story",
                published_at=first_seen,
                retrieved_at=first_seen,
                independent_group="agency_a",
            ),
        ),
    )
    second = KnowledgeUpdateCandidate(
        entity_id="country:ITA",
        field="head_of_government",
        value="Person B",
        valid_from=valid_from,
        detected_at=second_seen,
        domain=KnowledgeDomain.GOVERNANCE,
        kind=KnowledgeUpdateKind.FACT,
        confidence=0.88,
        evidence=(
            SourceEvidence(
                source="paper_b",
                source_ref="https://b.example/story",
                published_at=second_seen,
                retrieved_at=second_seen,
                independent_group="publisher_b",
            ),
        ),
    )

    first_result = updater.process(first)
    assert first_result.decision.action is PromotionAction.HOLD_FOR_MORE_EVIDENCE
    assert store.current_state("country:ITA") == {}

    second_result = updater.process(second)
    assert second_result.candidate_id == first.claim_key() == second.claim_key()
    assert second_result.decision.action is PromotionAction.AUTO_PROMOTE_FACT
    assert second_result.decision.independent_source_groups == 2
    assert store.current_state("country:ITA")["head_of_government"]["value"] == "Person B"

    status = store.status()
    assert status["candidates"] == 2
    assert status["distinct_claims"] == 1
    assert status["evidence_observations"] == 2


def test_repeated_reports_from_same_source_group_do_not_fake_corroboration(tmp_path):
    store = WorldKnowledgeStore(tmp_path / "same-source.duckdb")
    updater = AutonomousKnowledgeUpdater(store)

    valid_from = datetime(2026, 9, 27, 8, 0, tzinfo=timezone.utc)
    observations = []
    for minute, source in ((0, "paper_a"), (15, "paper_a_update")):
        seen = datetime(2026, 9, 27, 9, minute, tzinfo=timezone.utc)
        observations.append(
            KnowledgeUpdateCandidate(
                entity_id="country:ITA",
                field="head_of_government",
                value="Person B",
                valid_from=valid_from,
                detected_at=seen,
                domain=KnowledgeDomain.GOVERNANCE,
                kind=KnowledgeUpdateKind.FACT,
                confidence=0.95,
                evidence=(
                    SourceEvidence(
                        source=source,
                        source_ref=f"https://same.example/{minute}",
                        published_at=seen,
                        retrieved_at=seen,
                        independent_group="same_publisher_group",
                    ),
                ),
            )
        )

    updater.process(observations[0])
    result = updater.process(observations[1])

    assert result.decision.action is PromotionAction.HOLD_FOR_MORE_EVIDENCE
    assert result.decision.independent_source_groups == 1
    assert store.current_state("country:ITA") == {}
