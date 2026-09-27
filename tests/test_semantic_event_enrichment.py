from dataclasses import replace
from datetime import datetime, timezone

from preact.intelligence.geopolitical_state import EventImpact
from preact.intelligence.semantic_event_enrichment import (
    SemanticEventEnrichment,
    apply_semantic_enrichment,
)


NOW = datetime(2026, 9, 27, 16, tzinfo=timezone.utc)


def impact():
    return EventImpact(
        provider_event_id="1",
        source_iso3="ITA",
        target_iso3="FRA",
        event_time=NOW,
        known_at=NOW,
        event_code="190",
        event_root_code="19",
        semantic_tags=("conflict",),
        vector={
            "diplomatic_alignment": -0.8,
            "security_alignment": -0.9,
            "economic_alignment": 0.0,
            "institutional_alignment": 0.0,
            "conflict_intensity": 0.9,
        },
        severity=0.9,
        confidence=0.9,
        half_life_days=30,
        persistence="high",
        source_count=5,
        article_count=10,
    )


def test_weak_semantic_review_has_no_effect():
    base = impact()
    review = SemanticEventEnrichment(
        provider_event_id="1",
        model="test",
        event_type="other",
        direction="neutral",
        severity=0.1,
        persistence="short",
        confidence=0.3,
        dimension_modifiers={"security_alignment": 0.2},
        rationale="weak review",
    )
    assert apply_semantic_enrichment(base, review) == base


def test_semantic_review_is_bounded_and_cannot_flip_strong_signal():
    base = impact()
    review = SemanticEventEnrichment(
        provider_event_id="1",
        model="test",
        event_type="armed_conflict",
        direction="cooperative",
        severity=0.8,
        persistence="high",
        confidence=1.0,
        dimension_modifiers={
            "security_alignment": 0.2,
            "diplomatic_alignment": 0.2,
            "economic_alignment": 0.2,
        },
        rationale="bounded",
    )
    enriched = apply_semantic_enrichment(base, review)
    assert enriched.vector["security_alignment"] < 0
    assert enriched.vector["diplomatic_alignment"] < 0
    assert 0 < enriched.vector["economic_alignment"] <= 0.1
    assert enriched.half_life_days >= base.half_life_days
