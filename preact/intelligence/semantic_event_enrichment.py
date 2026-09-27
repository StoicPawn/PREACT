"""Bounded local-LLM enrichment for material geopolitical events.

The LLM is a semantic reviewer, never an authority. Output is schema-validated and
can only make bounded adjustments to the deterministic GDELT impact.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import json
import os
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

from preact.intelligence.geopolitical_state import ALL_DIMENSIONS, EventImpact

SCHEMA_VERSION = "semantic-event-v1"
ALLOWED_EVENT_TYPES = {
    "diplomatic_meeting",
    "diplomatic_agreement",
    "security_cooperation",
    "trade_or_economic_agreement",
    "sanctions_or_embargo",
    "diplomatic_dispute",
    "threat_or_coercion",
    "military_posturing",
    "military_incident",
    "armed_conflict",
    "protest_or_domestic_event",
    "other",
}
ALLOWED_DIRECTIONS = {"cooperative", "adversarial", "mixed", "neutral"}
ALLOWED_PERSISTENCE = {"short", "medium", "high"}


@dataclass(frozen=True)
class SemanticEventEnrichment:
    provider_event_id: str
    model: str
    event_type: str
    direction: str
    severity: float
    persistence: str
    confidence: float
    dimension_modifiers: Mapping[str, float]
    rationale: str
    schema_version: str = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dimension_modifiers"] = dict(self.dimension_modifiers)
        return payload


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _event_payload(row: Mapping[str, Any], impact: EventImpact) -> dict[str, Any]:
    return {
        "provider_event_id": impact.provider_event_id,
        "source_country": impact.source_iso3,
        "target_country": impact.target_iso3,
        "event_code": impact.event_code,
        "event_root_code": impact.event_root_code,
        "goldstein": row.get("goldstein"),
        "quad_class": row.get("quad_class"),
        "provider_tone": row.get("tone"),
        "action_location": row.get("action_location"),
        "actor1_name": row.get("actor1_name"),
        "actor2_name": row.get("actor2_name"),
        "themes": list(row.get("themes") or [])[:40],
        "persons": list(row.get("persons") or [])[:20],
        "organizations": list(row.get("organizations") or [])[:20],
        "source_count": impact.source_count,
        "article_count": impact.article_count,
        "deterministic_tags": list(impact.semantic_tags),
        "deterministic_vector": dict(impact.vector),
        "deterministic_severity": impact.severity,
    }


def _prompt(payload: Mapping[str, Any]) -> str:
    return (
        "You are a geopolitical event classifier. Review only the structured "
        "evidence below. Do not infer motives or facts not present. Return one JSON "
        "object with keys event_type, direction, severity, persistence, confidence, "
        "dimension_modifiers, rationale. event_type must be one of: "
        + ", ".join(sorted(ALLOWED_EVENT_TYPES))
        + ". direction must be cooperative/adversarial/mixed/neutral. "
        "persistence must be short/medium/high. severity and confidence are 0..1. "
        "dimension_modifiers may contain only diplomatic_alignment, "
        "security_alignment, economic_alignment, institutional_alignment, "
        "conflict_intensity; every modifier must be between -0.20 and +0.20. "
        "Modifiers are small corrections to an existing deterministic model, not "
        "replacement scores. rationale must be <= 180 characters. Evidence: "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    )


def _validate(
    provider_event_id: str,
    model: str,
    raw: Mapping[str, Any],
) -> SemanticEventEnrichment:
    event_type = str(raw.get("event_type") or "other")
    if event_type not in ALLOWED_EVENT_TYPES:
        event_type = "other"
    direction = str(raw.get("direction") or "neutral")
    if direction not in ALLOWED_DIRECTIONS:
        direction = "neutral"
    persistence = str(raw.get("persistence") or "short")
    if persistence not in ALLOWED_PERSISTENCE:
        persistence = "short"

    try:
        severity = _clip(float(raw.get("severity", 0.5)), 0.0, 1.0)
    except (TypeError, ValueError):
        severity = 0.5
    try:
        confidence = _clip(float(raw.get("confidence", 0.0)), 0.0, 1.0)
    except (TypeError, ValueError):
        confidence = 0.0

    raw_modifiers = raw.get("dimension_modifiers")
    modifiers: dict[str, float] = {}
    if isinstance(raw_modifiers, Mapping):
        for key, value in raw_modifiers.items():
            if key not in ALL_DIMENSIONS:
                continue
            try:
                modifiers[str(key)] = _clip(float(value), -0.20, 0.20)
            except (TypeError, ValueError):
                continue

    rationale = " ".join(str(raw.get("rationale") or "").split())[:180]
    return SemanticEventEnrichment(
        provider_event_id=provider_event_id,
        model=model,
        event_type=event_type,
        direction=direction,
        severity=severity,
        persistence=persistence,
        confidence=confidence,
        dimension_modifiers=modifiers,
        rationale=rationale,
    )


def classify_with_ollama(
    row: Mapping[str, Any],
    impact: EventImpact,
    *,
    url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 60.0,
) -> SemanticEventEnrichment:
    endpoint = (url or os.getenv(
        "PREACT_OLLAMA_URL", "http://127.0.0.1:11434"
    )).rstrip("/") + "/api/generate"
    model_name = model or os.getenv("PREACT_SEMANTIC_MODEL", "qwen3:1.7b")
    body = json.dumps(
        {
            "model": model_name,
            "prompt": _prompt(_event_payload(row, impact)),
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": 160,
            },
        }
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    generated = payload.get("response")
    if not isinstance(generated, str):
        raise ValueError("Ollama response does not contain JSON text")
    parsed = json.loads(generated)
    if not isinstance(parsed, Mapping):
        raise ValueError("Ollama semantic response must be a JSON object")
    return _validate(impact.provider_event_id, model_name, parsed)


def apply_semantic_enrichment(
    impact: EventImpact,
    enrichment: SemanticEventEnrichment | None,
) -> EventImpact:
    """Apply a bounded reviewer adjustment; weak reviews have no effect."""

    if enrichment is None or enrichment.confidence < 0.55:
        return impact

    vector = dict(impact.vector)
    reviewer_weight = min(0.50, max(0.0, enrichment.confidence - 0.50))
    for dimension, modifier in enrichment.dimension_modifiers.items():
        base = float(vector.get(dimension, 0.0))
        delta = float(modifier) * reviewer_weight
        candidate = base + delta
        if dimension == "conflict_intensity":
            vector[dimension] = _clip(candidate, 0.0, 1.0)
            continue
        # A small model cannot flip a strong deterministic signal.
        if abs(base) >= 0.60 and base * candidate < 0:
            continue
        vector[dimension] = _clip(candidate, -1.0, 1.0)

    severity = _clip(
        0.85 * impact.severity + 0.15 * enrichment.severity,
        0.05,
        1.0,
    )
    half_life = impact.half_life_days
    if enrichment.persistence == "medium":
        half_life = max(half_life, 21.0)
    elif enrichment.persistence == "high":
        half_life = max(half_life, 60.0)

    tags = tuple(sorted(set(impact.semantic_tags).union(
        {f"semantic:{enrichment.event_type}"}
    )))
    return replace(
        impact,
        vector=vector,
        severity=severity,
        half_life_days=half_life,
        semantic_tags=tags,
    )


def materiality_score(impact: EventImpact) -> float:
    source_bonus = min(1.0, impact.source_count / 8.0)
    persistent_bonus = {
        "short": 0.0,
        "medium": 0.10,
        "high": 0.20,
    }.get(impact.persistence, 0.0)
    return _clip(
        impact.severity * impact.confidence * (0.75 + 0.25 * source_bonus)
        + persistent_bonus,
        0.0,
        1.2,
    )


__all__ = [
    "SCHEMA_VERSION",
    "SemanticEventEnrichment",
    "apply_semantic_enrichment",
    "classify_with_ollama",
    "materiality_score",
]
