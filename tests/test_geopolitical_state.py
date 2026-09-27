from datetime import datetime, timedelta, timezone

from preact.intelligence.geopolitical_state import (
    build_states,
    estimate_pair_state,
    interpret_event,
)


NOW = datetime(2026, 9, 27, 16, tzinfo=timezone.utc)


def event(
    *,
    event_id="1",
    root="04",
    base="040",
    goldstein=3.0,
    quad=1,
    themes=(),
    event_time=NOW,
    sources=4,
    articles=6,
    mention_sources=4,
    mention_confidence=90.0,
    is_root=True,
    source_url=None,
):
    return {
        "provider_event_id": event_id,
        "event_time": event_time,
        "known_at": NOW,
        "actor1_entity_id": "country:ITA",
        "actor2_entity_id": "country:FRA",
        "event_code": base,
        "event_base_code": base,
        "event_root_code": root,
        "quad_class": quad,
        "goldstein": goldstein,
        "tone": 0.0,
        "num_sources": sources,
        "num_articles": articles,
        "corroborating_mentions": articles,
        "mention_source_count": mention_sources,
        "mention_max_confidence": mention_confidence,
        "is_root_event": is_root,
        "source_url": source_url or f"https://wire.example/{event_id}",
        "themes": list(themes),
    }


def test_diplomatic_meeting_creates_positive_diplomatic_impact():
    impact = interpret_event(event())
    assert impact is not None
    assert impact.vector["diplomatic_alignment"] > 0
    assert impact.confidence > 0.5
    assert impact.half_life_days >= 7


def test_sanctions_target_economic_and_diplomatic_dimensions():
    impact = interpret_event(
        event(
            root="16",
            base="163",
            goldstein=-5.0,
            quad=3,
            themes=("SANCTIONS", "ECONOMIC"),
        )
    )
    assert impact is not None
    assert impact.vector["economic_alignment"] <= -0.7
    assert impact.vector["diplomatic_alignment"] < 0
    assert impact.half_life_days >= 90
    assert impact.persistence == "high"


def test_fighting_produces_high_conflict_intensity():
    impact = interpret_event(
        event(
            root="19",
            base="190",
            goldstein=-10.0,
            quad=4,
            themes=("ARMED_CONFLICT", "MILITARY"),
        )
    )
    assert impact is not None
    state = estimate_pair_state([impact], as_of=NOW, mode="canonical")
    assert state.vector["security_alignment"] < -0.2
    assert state.vector["conflict_intensity"] > 0.3
    assert state.status in {"tension", "conflict"}


def test_live_pulse_forgets_old_transient_event_faster_than_canonical():
    impact = interpret_event(
        event(
            event_time=NOW - timedelta(days=6),
            root="04",
            goldstein=3.0,
        )
    )
    assert impact is not None
    canonical = estimate_pair_state([impact], as_of=NOW, mode="canonical")
    live = estimate_pair_state(
        [impact],
        as_of=NOW,
        mode="live",
        canonical_reference=canonical,
    )
    assert abs(live.overall_score) < abs(canonical.overall_score)


def test_structural_alliance_can_anchor_state_without_recent_news():
    states = build_states(
        [],
        as_of=NOW,
        mode="canonical",
        anchor_pairs={"FRA|ITA": ("formal_alliance",)},
    )
    assert len(states) == 1
    state = states[0]
    assert state.status == "documented_alliance"
    assert state.vector["security_alignment"] > 0.5
    assert state.confidence > 0.4
    assert state.event_count == 0


def test_live_trend_compares_against_canonical_state():
    good = interpret_event(event(event_id="good", root="05", goldstein=5.0))
    bad = interpret_event(
        event(
            event_id="bad",
            root="19",
            goldstein=-10.0,
            quad=4,
            themes=("ARMED_CONFLICT",),
        )
    )
    assert good and bad
    canonical = estimate_pair_state([good], as_of=NOW, mode="canonical")
    live = estimate_pair_state(
        [bad],
        as_of=NOW,
        mode="live",
        canonical_reference=canonical,
    )
    assert live.live_delta is not None
    assert live.live_delta < -0.12
    assert live.trend == "deteriorating"



def test_single_source_extreme_event_is_damped_until_corroborated():
    weak = interpret_event(
        event(
            event_id="weak",
            root="19",
            base="190",
            goldstein=-10.0,
            quad=4,
            themes=("ARMED_CONFLICT", "MILITARY"),
            sources=1,
            articles=1,
            mention_sources=1,
            mention_confidence=50.0,
        )
    )
    strong = interpret_event(
        event(
            event_id="strong",
            root="19",
            base="190",
            goldstein=-10.0,
            quad=4,
            themes=("ARMED_CONFLICT", "MILITARY"),
            sources=4,
            articles=8,
            mention_sources=4,
            mention_confidence=95.0,
        )
    )
    assert weak and strong
    assert weak.vector["conflict_intensity"] < strong.vector["conflict_intensity"]
    assert weak.severity < strong.severity


def test_non_root_single_source_background_event_is_rejected():
    impact = interpret_event(
        event(
            event_id="background",
            root="19",
            base="190",
            goldstein=-10.0,
            quad=4,
            themes=("ARMEDCONFLICT", "MILITARY"),
            sources=1,
            articles=1,
            mention_sources=1,
            is_root=False,
        )
    )
    assert impact is None


def test_non_root_event_can_survive_after_independent_corroboration():
    root_version = interpret_event(
        event(
            event_id="root",
            root="16",
            base="163",
            goldstein=-5.0,
            sources=3,
            articles=4,
            mention_sources=3,
            is_root=True,
        )
    )
    non_root = interpret_event(
        event(
            event_id="corroborated",
            root="16",
            base="163",
            goldstein=-5.0,
            sources=3,
            articles=4,
            mention_sources=3,
            is_root=False,
        )
    )
    assert root_version is not None and non_root is not None
    assert non_root.confidence < root_version.confidence


def test_one_article_with_multiple_extractions_does_not_multiply_state_shock():
    first = interpret_event(
        event(
            event_id="a",
            root="19",
            base="190",
            goldstein=-10.0,
            source_url="https://one.example/story",
        )
    )
    second = interpret_event(
        event(
            event_id="b",
            root="19",
            base="190",
            goldstein=-10.0,
            source_url="https://one.example/story",
        )
    )
    assert first is not None and second is not None
    single = estimate_pair_state([first], as_of=NOW, mode="canonical")
    duplicated = estimate_pair_state([first, second], as_of=NOW, mode="canonical")
    assert abs(
        duplicated.vector["diplomatic_alignment"]
        - single.vector["diplomatic_alignment"]
    ) < 1e-9
    assert abs(
        duplicated.vector["conflict_intensity"]
        - single.vector["conflict_intensity"]
    ) < 1e-9
