"""Evidence-weighted geopolitical relationship state estimation.

This module estimates a latent bilateral state from provider observations. It does
not turn media-derived signals into factual assertions. Structural facts remain
separate anchors and every estimated state retains its evidence and model version.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from math import exp, log1p, tanh
from typing import Any, Iterable, Mapping, Sequence

MODEL_VERSION = "geopolitical-state-v3"

SIGNED_DIMENSIONS = (
    "diplomatic_alignment",
    "security_alignment",
    "economic_alignment",
    "institutional_alignment",
)
ALL_DIMENSIONS = SIGNED_DIMENSIONS + ("conflict_intensity",)


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _pair_key(a: str, b: str) -> str:
    left, right = sorted((str(a).upper(), str(b).upper()))
    return f"{left}|{right}"


@dataclass(frozen=True)
class EventImpact:
    provider_event_id: str
    source_iso3: str
    target_iso3: str
    event_time: datetime
    known_at: datetime
    event_code: str | None
    event_root_code: str | None
    semantic_tags: tuple[str, ...]
    vector: Mapping[str, float]
    severity: float
    confidence: float
    half_life_days: float
    persistence: str
    source_count: int
    article_count: int
    source_url: str | None = None
    evidence_event_ids: tuple[str, ...] = ()
    cluster_size: int = 1
    model_version: str = MODEL_VERSION

    @property
    def pair_key(self) -> str:
        return _pair_key(self.source_iso3, self.target_iso3)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event_time"] = self.event_time.isoformat()
        payload["known_at"] = self.known_at.isoformat()
        payload["vector"] = dict(self.vector)
        payload["semantic_tags"] = list(self.semantic_tags)
        return payload


@dataclass(frozen=True)
class RelationshipState:
    pair_key: str
    source_iso3: str
    target_iso3: str
    as_of: datetime
    known_at: datetime
    mode: str
    vector: Mapping[str, float]
    overall_score: float
    confidence: float
    coverage: float
    status: str
    trend: str
    live_delta: float | None
    event_count: int
    source_count: int
    last_event_at: datetime | None
    structural_anchors: tuple[str, ...] = ()
    evidence_event_ids: tuple[str, ...] = ()
    model_version: str = MODEL_VERSION

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["as_of"] = self.as_of.isoformat()
        payload["known_at"] = self.known_at.isoformat()
        payload["last_event_at"] = (
            self.last_event_at.isoformat() if self.last_event_at else None
        )
        payload["vector"] = dict(self.vector)
        payload["structural_anchors"] = list(self.structural_anchors)
        payload["evidence_event_ids"] = list(self.evidence_event_ids)
        return payload


_ROOT_VECTOR: dict[str, dict[str, float]] = {
    "01": {"diplomatic_alignment": 0.05},
    "02": {"diplomatic_alignment": 0.10},
    "03": {"diplomatic_alignment": 0.25},
    "04": {"diplomatic_alignment": 0.35},
    "05": {"diplomatic_alignment": 0.55, "institutional_alignment": 0.15},
    "06": {"diplomatic_alignment": 0.45},
    "07": {"diplomatic_alignment": 0.35},
    "08": {"diplomatic_alignment": 0.20},
    "09": {"diplomatic_alignment": -0.05},
    "10": {"diplomatic_alignment": -0.20},
    "11": {"diplomatic_alignment": -0.30},
    "12": {"diplomatic_alignment": -0.35},
    "13": {
        "diplomatic_alignment": -0.50,
        "security_alignment": -0.25,
        "conflict_intensity": 0.05,
    },
    "14": {"diplomatic_alignment": -0.30, "conflict_intensity": 0.05},
    "15": {
        "diplomatic_alignment": -0.55,
        "security_alignment": -0.55,
        "conflict_intensity": 0.25,
    },
    "16": {"diplomatic_alignment": -0.65, "economic_alignment": -0.15},
    "17": {
        "diplomatic_alignment": -0.70,
        "security_alignment": -0.35,
        "conflict_intensity": 0.25,
    },
    "18": {
        "diplomatic_alignment": -0.85,
        "security_alignment": -0.80,
        "conflict_intensity": 0.65,
    },
    "19": {
        "diplomatic_alignment": -0.95,
        "security_alignment": -0.95,
        "conflict_intensity": 0.90,
    },
    "20": {
        "diplomatic_alignment": -1.00,
        "security_alignment": -1.00,
        "conflict_intensity": 1.00,
    },
}

_ROOT_SEVERITY = {
    "01": 0.10, "02": 0.12, "03": 0.20, "04": 0.25, "05": 0.35,
    "06": 0.40, "07": 0.35, "08": 0.25, "09": 0.15, "10": 0.25,
    "11": 0.30, "12": 0.35, "13": 0.45, "14": 0.35, "15": 0.60,
    "16": 0.65, "17": 0.70, "18": 0.82, "19": 0.95, "20": 1.00,
}

_ROOT_HALF_LIFE = {
    "01": 3.0, "02": 3.0, "03": 7.0, "04": 7.0, "05": 14.0,
    "06": 30.0, "07": 30.0, "08": 14.0, "09": 7.0, "10": 7.0,
    "11": 7.0, "12": 10.0, "13": 14.0, "14": 7.0, "15": 21.0,
    "16": 60.0, "17": 45.0, "18": 30.0, "19": 30.0, "20": 45.0,
}


def _themes(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw = row.get("themes") or []
    if isinstance(raw, str):
        raw = [item for item in raw.split(";") if item]
    return tuple(sorted({str(item).strip().upper() for item in raw if str(item).strip()}))


def _contains_theme(themes: Sequence[str], *needles: str) -> bool:
    haystack = " ".join(themes)
    return any(needle in haystack for needle in needles)


def _semantic_tags(row: Mapping[str, Any], root: str, base: str) -> tuple[str, ...]:
    themes = _themes(row)
    tags: set[str] = set()

    if root in {"03", "04", "05"} or _contains_theme(themes, "DIPLO", "NEGOTIAT"):
        tags.add("diplomacy")
    if root in {"15", "17", "18", "19", "20"} or _contains_theme(
        themes, "MILIT", "ARMED", "WAR", "DEFEN", "SECURITY"
    ):
        tags.add("security")
    if base.startswith("163") or _contains_theme(
        themes, "SANCTION", "EMBARGO", "BOYCOTT"
    ):
        tags.add("sanctions")
    if _contains_theme(themes, "TRADE", "ECONOM", "INVEST", "TARIFF"):
        tags.add("economic")
    if _contains_theme(themes, "ALLIANCE", "TREATY", "NATO", "DEFEN"):
        tags.add("alliance_or_treaty")
    if _contains_theme(
        themes, "UNITED_NATIONS", "INTERNATIONAL_ORGAN", "EUROPEAN_UNION",
        "MULTILATER", "DIPLOMACY"
    ):
        tags.add("institutional")
    if root in {"18", "19", "20"} or _contains_theme(
        themes, "CONFLICT", "WAR", "TERROR", "VIOLENCE", "MILITARY"
    ):
        tags.add("conflict")
    return tuple(sorted(tags))


def interpret_event(row: Mapping[str, Any]) -> EventImpact | None:
    """Convert one latest-known GDELT event observation into an audit-ready impact."""

    source = str(row.get("actor1_entity_id") or "").removeprefix("country:").upper()
    target = str(row.get("actor2_entity_id") or "").removeprefix("country:").upper()
    event_id = str(row.get("provider_event_id") or "").strip()
    if len(source) != 3 or len(target) != 3 or source == target or not event_id:
        return None

    event_time = row.get("event_time")
    known_at = row.get("known_at")
    if not isinstance(event_time, datetime) or not isinstance(known_at, datetime):
        return None

    root = str(row.get("event_root_code") or "").strip().zfill(2)[:2]
    base = str(row.get("event_base_code") or "").strip()
    vector = {name: 0.0 for name in ALL_DIMENSIONS}
    for key, value in _ROOT_VECTOR.get(root, {}).items():
        vector[key] = float(value)

    goldstein = row.get("goldstein")
    try:
        goldstein_value = float(goldstein)
    except (TypeError, ValueError):
        goldstein_value = 0.0

    tags = _semantic_tags(row, root, base)
    signed_hint = _clip(goldstein_value / 10.0)
    if signed_hint == 0.0:
        quad = int(row.get("quad_class") or 0)
        signed_hint = {1: 0.20, 2: 0.45, 3: -0.25, 4: -0.70}.get(quad, 0.0)

    if "sanctions" in tags:
        vector["economic_alignment"] = min(vector["economic_alignment"], -0.75)
        vector["diplomatic_alignment"] = min(vector["diplomatic_alignment"], -0.45)
    elif "economic" in tags:
        vector["economic_alignment"] = _clip(
            signed_hint * max(0.30, abs(signed_hint))
        )

    if "security" in tags:
        candidate = _clip(signed_hint * max(0.35, abs(signed_hint)))
        if candidate < 0:
            vector["security_alignment"] = min(vector["security_alignment"], candidate)
        else:
            vector["security_alignment"] = max(vector["security_alignment"], candidate)

    if "alliance_or_treaty" in tags and signed_hint > 0:
        vector["security_alignment"] = max(vector["security_alignment"], 0.55)
        vector["institutional_alignment"] = max(
            vector["institutional_alignment"], 0.25
        )

    if "institutional" in tags:
        candidate = _clip(signed_hint * 0.35)
        vector["institutional_alignment"] = (
            max(vector["institutional_alignment"], candidate)
            if candidate >= 0
            else min(vector["institutional_alignment"], candidate)
        )

    if "conflict" in tags and signed_hint < 0:
        vector["conflict_intensity"] = max(
            vector["conflict_intensity"], min(1.0, abs(signed_hint))
        )

    rule_severity = _ROOT_SEVERITY.get(root, min(1.0, abs(signed_hint)))
    severity = _clip(
        0.55 * rule_severity + 0.45 * min(1.0, abs(goldstein_value) / 10.0),
        0.05,
        1.0,
    )

    event_sources = int(float(row.get("num_sources") or 0))
    mention_sources = int(float(row.get("mention_source_count") or 0))
    source_count = max(event_sources, mention_sources)
    article_count = max(
        int(float(row.get("num_articles") or 0)),
        int(float(row.get("corroborating_mentions") or 0)),
    )
    source_domains = {
        str(item).strip().lower()
        for item in (row.get("source_domains") or [])
        if str(item).strip()
    }
    independent_source_count = max(source_count, len(source_domains))

    # Extreme CAMEO codes can occasionally come from a tangential reference in
    # one article. Preserve the signal but stop one document defining a conflict.
    if root in {"18", "19", "20"} and independent_source_count < 2:
        vector["conflict_intensity"] *= 0.25
        vector["security_alignment"] *= 0.45
        vector["diplomatic_alignment"] *= 0.45
        severity *= 0.45

    mention_conf = row.get("mention_max_confidence")
    try:
        mention_confidence = _clip(float(mention_conf) / 100.0, 0.0, 1.0)
    except (TypeError, ValueError):
        mention_confidence = 0.0

    # GDELT can code background/historical mentions as bilateral events. For the
    # relationship state we fail closed on a non-root, single-source observation.
    # It remains in World Knowledge evidence and can become eligible later if
    # additional independent Mentions arrive.
    raw_root = row.get("is_root_event")
    is_root_event = None if raw_root is None else bool(raw_root)
    if is_root_event is False and source_count < 2 and article_count < 2:
        return None

    source_strength = 1.0 - exp(-max(0, source_count) / 4.0)
    article_strength = min(1.0, log1p(max(0, article_count)) / log1p(20.0))
    confidence = _clip(
        0.35 + 0.30 * source_strength + 0.20 * article_strength
        + 0.15 * mention_confidence,
        0.20,
        0.98,
    )
    if is_root_event is False:
        confidence *= 0.70

    half_life = _ROOT_HALF_LIFE.get(root, 7.0)
    persistence = "short"
    if "sanctions" in tags:
        half_life = max(half_life, 90.0)
        persistence = "high"
    elif "alliance_or_treaty" in tags and signed_hint > 0:
        half_life = max(half_life, 90.0)
        persistence = "high"
    elif root in {"16", "17", "18", "19", "20"}:
        persistence = "medium" if root in {"17", "18"} else "high"
    elif half_life >= 30:
        persistence = "medium"

    return EventImpact(
        provider_event_id=event_id,
        source_iso3=source,
        target_iso3=target,
        event_time=event_time,
        known_at=known_at,
        event_code=str(row.get("event_code") or "").strip() or None,
        event_root_code=root or None,
        semantic_tags=tags,
        vector={key: _clip(value, 0.0, 1.0) if key == "conflict_intensity" else _clip(value)
                for key, value in vector.items()},
        severity=severity,
        confidence=confidence,
        half_life_days=float(half_life),
        persistence=persistence,
        source_count=source_count,
        article_count=article_count,
        source_url=str(row.get("source_url") or "").strip() or None,
        evidence_event_ids=tuple(
            str(item).strip()
            for item in (row.get("cluster_event_ids") or [event_id])
            if str(item).strip()
        ),
        cluster_size=max(1, int(row.get("cluster_size") or 1)),
    )


def _anchor_contribution(
    anchors: Sequence[str],
) -> tuple[dict[str, float], float]:
    accum = {name: 0.0 for name in ALL_DIMENSIONS}
    mass = 0.0
    if "formal_alliance" in anchors:
        anchor_mass = 2.5
        accum["diplomatic_alignment"] += 0.60 * anchor_mass
        accum["security_alignment"] += 0.85 * anchor_mass
        accum["institutional_alignment"] += 0.35 * anchor_mass
        mass += anchor_mass
    if "active_conflict" in anchors:
        anchor_mass = 3.0
        accum["diplomatic_alignment"] -= 0.80 * anchor_mass
        accum["security_alignment"] -= 0.95 * anchor_mass
        accum["conflict_intensity"] += 1.00 * anchor_mass
        mass += anchor_mass
    return accum, mass


def estimate_pair_state(
    impacts: Sequence[EventImpact],
    *,
    as_of: datetime,
    mode: str,
    structural_anchors: Sequence[str] = (),
    canonical_reference: RelationshipState | None = None,
    pair_key_override: str | None = None,
) -> RelationshipState:
    """Estimate one bilateral latent state without letting media volume saturate it.

    Event volume increases confidence, not polarity. Each dimension is a decayed,
    evidence-weighted mean of relevant event impacts plus structural pseudo-evidence.
    """
    if mode not in {"live", "canonical"}:
        raise ValueError("mode must be 'live' or 'canonical'")
    if not impacts and not structural_anchors:
        raise ValueError("at least one impact or structural anchor is required")

    sample = impacts[0] if impacts else None
    if sample is not None:
        a, b = sorted((sample.source_iso3, sample.target_iso3))
    elif pair_key_override and "|" in pair_key_override:
        a, b = pair_key_override.split("|", 1)
        if len(a) != 3 or len(b) != 3:
            raise ValueError("invalid pair_key_override")
    else:
        raise ValueError("pair identity is required for structural-only state")

    numerators = {name: 0.0 for name in ALL_DIMENSIONS}
    masses = {name: 0.0 for name in ALL_DIMENSIONS}
    max_conflict = 0.0
    anchor_accum, anchor_mass = _anchor_contribution(tuple(structural_anchors))
    for dimension, value in anchor_accum.items():
        if abs(value) <= 1e-12:
            continue
        numerators[dimension] += value
        masses[dimension] += anchor_mass

    evidence_mass = anchor_mass
    source_groups: set[str] = set()
    last_event: datetime | None = None
    event_ids: list[str] = []

    # One article may yield several CAMEO events. Treat the article as one finite
    # unit of evidence rather than allowing extraction multiplicity to amplify it.
    url_counts: dict[str, int] = {}
    for impact in impacts:
        url = (impact.source_url or "").strip()
        if url:
            url_counts[url] = url_counts.get(url, 0) + 1

    for impact in impacts:
        age_days = max(0.0, (as_of - impact.event_time).total_seconds() / 86400.0)
        half_life = impact.half_life_days
        if mode == "live":
            half_life = min(2.0, max(0.35, half_life))
        decay = 2.0 ** (-age_days / max(0.1, half_life))
        source_strength = 1.0 - exp(-max(0, impact.source_count) / 4.0)
        raw_weight = (
            impact.confidence
            * (0.25 + 0.75 * impact.severity)
            * (0.65 + 0.35 * source_strength)
            * decay
        )
        weight = raw_weight
        url = (impact.source_url or "").strip()
        if url:
            weight /= max(1, url_counts.get(url, 1))
        if weight < 0.01:
            continue

        # Confidence may grow with independent evidence, but repeated media coverage
        # must not linearly push the latent state toward +/-1.
        evidence_mass += min(1.0, weight)
        event_ids.extend(
            impact.evidence_event_ids or (impact.provider_event_id,)
        )
        if impact.source_url:
            source_groups.add(
                impact.source_url.split("/", 3)[2].lower()
                if "://" in impact.source_url
                else impact.source_url
            )
        if last_event is None or impact.event_time > last_event:
            last_event = impact.event_time

        for dimension, raw_value in impact.vector.items():
            if dimension not in numerators:
                continue
            value = float(raw_value)
            if abs(value) <= 1e-12:
                continue
            numerators[dimension] += value * weight
            masses[dimension] += weight
            if dimension == "conflict_intensity":
                max_conflict = max(max_conflict, value * min(1.0, raw_weight))

    # Neutral prior prevents one observation from defining the relationship and
    # makes temporal decay pull old event-only states back toward zero.
    neutral_prior_mass = 0.30 if mode == "canonical" else 0.80
    vector: dict[str, float] = {}
    for dimension in SIGNED_DIMENSIONS:
        vector[dimension] = _clip(
            numerators[dimension]
            / (masses[dimension] + neutral_prior_mass)
            if masses[dimension] > 0
            else 0.0
        )

    mean_conflict = (
        numerators["conflict_intensity"]
        / (masses["conflict_intensity"] + neutral_prior_mass)
        if masses["conflict_intensity"] > 0
        else 0.0
    )
    # A serious recent incident matters even if many benign interactions coexist,
    # but a large count of small conflict-coded articles cannot saturate the state.
    vector["conflict_intensity"] = _clip(
        0.70 * mean_conflict + 0.30 * max_conflict,
        0.0,
        1.0,
    )

    weights = {
        "diplomatic_alignment": 0.35,
        "security_alignment": 0.30,
        "economic_alignment": 0.20,
        "institutional_alignment": 0.15,
    }
    active_weight = sum(
        dim_weight
        for dimension, dim_weight in weights.items()
        if masses[dimension] > 0
    )
    signed_score = (
        sum(
            vector[dimension] * weights[dimension]
            for dimension in SIGNED_DIMENSIONS
            if masses[dimension] > 0
        ) / active_weight
        if active_weight > 0
        else 0.0
    )
    overall = _clip(
        signed_score - 0.30 * vector["conflict_intensity"]
    )

    touched = sum(1 for dimension in ALL_DIMENSIONS if masses[dimension] > 0)
    coverage = touched / len(ALL_DIMENSIONS)
    confidence = _clip(
        (1.0 - exp(-evidence_mass / 3.5))
        * (0.68 + 0.32 * coverage),
        0.0,
        0.995,
    )

    if confidence < 0.15:
        status = "insufficient_evidence"
    elif vector["conflict_intensity"] >= 0.72 and overall <= -0.25:
        status = "conflict"
    elif overall <= -0.18 or vector["conflict_intensity"] >= 0.45:
        status = "tension"
    elif "formal_alliance" in structural_anchors and overall > -0.18:
        status = "documented_alliance"
    elif overall >= 0.55:
        status = "strong_cooperation"
    elif overall >= 0.18:
        status = "affinity"
    else:
        status = "mixed"

    live_delta: float | None = None
    trend = "stable"
    if mode == "live" and canonical_reference is not None:
        live_delta = overall - canonical_reference.overall_score
        if live_delta >= 0.12:
            trend = "improving"
        elif live_delta <= -0.12:
            trend = "deteriorating"

    known_at = max((impact.known_at for impact in impacts), default=as_of)
    return RelationshipState(
        pair_key=_pair_key(a, b),
        source_iso3=a,
        target_iso3=b,
        as_of=as_of,
        known_at=known_at,
        mode=mode,
        vector=vector,
        overall_score=overall,
        confidence=confidence,
        coverage=coverage,
        status=status,
        trend=trend,
        live_delta=live_delta,
        event_count=len(set(event_ids)),
        source_count=len(source_groups),
        last_event_at=last_event,
        structural_anchors=tuple(sorted(set(structural_anchors))),
        evidence_event_ids=tuple(dict.fromkeys(event_ids[-50:])),
    )

def build_states(
    impacts: Iterable[EventImpact],
    *,
    as_of: datetime,
    mode: str,
    anchor_pairs: Mapping[str, Sequence[str]] | None = None,
    canonical_states: Mapping[str, RelationshipState] | None = None,
) -> list[RelationshipState]:
    groups: dict[str, list[EventImpact]] = {}
    for impact in impacts:
        groups.setdefault(impact.pair_key, []).append(impact)

    anchors = dict(anchor_pairs or {})
    canonical = dict(canonical_states or {})
    states: list[RelationshipState] = []
    pair_keys = sorted(set(groups).union(anchors))
    for pair_key in pair_keys:
        pair_impacts = groups.get(pair_key, [])
        states.append(
            estimate_pair_state(
                pair_impacts,
                as_of=as_of,
                mode=mode,
                structural_anchors=anchors.get(pair_key, ()),
                canonical_reference=canonical.get(pair_key),
                pair_key_override=pair_key,
            )
        )
    return states


__all__ = [
    "ALL_DIMENSIONS",
    "EventImpact",
    "MODEL_VERSION",
    "RelationshipState",
    "SIGNED_DIMENSIONS",
    "build_states",
    "estimate_pair_state",
    "interpret_event",
]
