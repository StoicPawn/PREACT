"""Populate and maintain current political reference facts.

Wikidata provides the global structured reference baseline. It can seed fields only
when PREACT has no current assertion. Any later change is treated as a candidate and
must pass normal corroboration; Wikidata by itself never overwrites an existing fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import pycountry

from preact.data_hub.news_store import SharedNewsStore
from preact.data_hub.wikidata import WikidataCountryReference
from preact.history.world_knowledge_store import WorldKnowledgeStore
from preact.intelligence.knowledge_update import AutonomousKnowledgeUpdater
from preact.intelligence.world_knowledge import (
    KnowledgeDomain,
    KnowledgeUpdateCandidate,
    KnowledgeUpdateKind,
    PromotionAction,
    PromotionPolicy,
    SourceEvidence,
)


_FIELD_DOMAINS: dict[str, KnowledgeDomain] = {
    "capital": KnowledgeDomain.GOVERNANCE,
    "government_form": KnowledgeDomain.GOVERNANCE,
    "head_of_state": KnowledgeDomain.GOVERNANCE,
    "head_of_government": KnowledgeDomain.GOVERNANCE,
    "official_languages": KnowledgeDomain.SOCIETY,
    "official_website": KnowledgeDomain.GOVERNANCE,
}

_OFFICE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "head_of_state": (
        "president", "head of state", "king", "queen", "monarch",
        "presidente", "capo dello stato", "re ", "regina",
    ),
    "head_of_government": (
        "prime minister", "premier", "chancellor", "head of government",
        "primo ministro", "presidente del consiglio", "cancelliere",
        "sworn in", "appointed", "elected", "resign", "resigns",
    ),
}


@dataclass(frozen=True)
class PoliticalReferenceRefreshResult:
    references: int
    countries_considered: int
    fields_seeded: int
    unchanged_fields: int
    change_candidates: int
    promoted_changes: int
    held_changes: int
    skipped_fields: int

    def as_dict(self) -> dict[str, int]:
        return {
            "references": self.references,
            "countries_considered": self.countries_considered,
            "fields_seeded": self.fields_seeded,
            "unchanged_fields": self.unchanged_fields,
            "change_candidates": self.change_candidates,
            "promoted_changes": self.promoted_changes,
            "held_changes": self.held_changes,
            "skipped_fields": self.skipped_fields,
        }


def _scalar_or_list(values: Iterable[str]) -> str | list[str] | None:
    cleaned = sorted({str(value).strip() for value in values if str(value).strip()})
    if not cleaned:
        return None
    return cleaned[0] if len(cleaned) == 1 else cleaned


def _reference_fields(reference: WikidataCountryReference) -> dict[str, Any]:
    return {
        "capital": _scalar_or_list(reference.capital),
        "government_form": _scalar_or_list(reference.government_forms),
        "head_of_state": _scalar_or_list(reference.heads_of_state),
        "head_of_government": _scalar_or_list(reference.heads_of_government),
        "official_languages": _scalar_or_list(reference.official_languages),
        "official_website": _scalar_or_list(reference.official_websites),
    }


def _canonical(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(_canonical(item) for item in value)
    if isinstance(value, tuple):
        return sorted(_canonical(item) for item in value)
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return value


def _utc(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        stamp = value
    else:
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return fallback
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _country_aliases(reference: WikidataCountryReference) -> tuple[str, ...]:
    aliases = {reference.country_label.casefold(), reference.iso3.casefold()}
    country = pycountry.countries.get(alpha_3=reference.iso3)
    if country is not None:
        for attr in ("name", "official_name", "common_name"):
            value = getattr(country, attr, None)
            if value:
                aliases.add(str(value).casefold())
    return tuple(sorted((item for item in aliases if len(item) >= 3), key=len, reverse=True))


def _value_labels(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value or "").strip()
    return (text,) if text else ()


def _news_evidence_for_change(
    news: SharedNewsStore | None,
    *,
    reference: WikidataCountryReference,
    field: str,
    value: Any,
    limit_per_label: int = 100,
) -> tuple[SourceEvidence, ...]:
    if news is None:
        return ()

    aliases = _country_aliases(reference)
    labels = _value_labels(value)
    if not labels:
        return ()

    keyword_set = tuple(item.casefold() for item in _OFFICE_KEYWORDS.get(field, ()))
    by_group: dict[str, SourceEvidence] = {}

    for label in labels:
        try:
            rows = news.latest(query=label, limit=limit_per_label)
        except Exception:
            continue
        for row in rows:
            title = str(row.get("title") or "")
            snippet = str(row.get("snippet") or "")
            text = f"{title} {snippet}".casefold()
            if label.casefold() not in text:
                continue
            if not any(alias in text for alias in aliases):
                continue
            if keyword_set and not any(keyword in text for keyword in keyword_set):
                continue

            domain = str(row.get("domain") or "").strip().lower().removeprefix("www.")
            publisher = str(row.get("publisher") or "").strip().casefold()
            provider = str(row.get("provider") or "news").strip()
            group = domain or publisher or provider
            if not group or group in by_group:
                continue

            retrieved = _utc(row.get("retrieved_at"), reference.retrieved_at)
            published = _utc(row.get("published_at"), retrieved)
            if published > retrieved:
                published = retrieved

            source_ref = str(row.get("url") or "").strip()
            if not source_ref:
                continue
            by_group[group] = SourceEvidence(
                source=provider,
                source_ref=source_ref,
                published_at=published,
                retrieved_at=retrieved,
                independent_group=group,
                authoritative=False,
            )

    return tuple(by_group[key] for key in sorted(by_group))


def _candidate(
    reference: WikidataCountryReference,
    *,
    field: str,
    value: Any,
    evidence: tuple[SourceEvidence, ...],
    confidence: float,
) -> KnowledgeUpdateCandidate:
    return KnowledgeUpdateCandidate(
        entity_id=f"country:{reference.iso3}",
        field=field,
        value=value,
        valid_from=reference.retrieved_at,
        detected_at=max(
            (item.retrieved_at for item in evidence),
            default=reference.retrieved_at,
        ),
        domain=_FIELD_DOMAINS[field],
        kind=KnowledgeUpdateKind.FACT,
        confidence=confidence,
        evidence=evidence,
        attributes={
            "reference_source": "wikidata",
            "wikidata_qid": reference.qid,
            "snapshot_checksum": reference.snapshot_checksum,
            "valid_time_precision": "observed_current_state",
            "reference_country_label": reference.country_label,
        },
    )


def refresh_political_reference(
    references: Iterable[WikidataCountryReference],
    *,
    store: WorldKnowledgeStore,
    news: SharedNewsStore | None = None,
) -> PoliticalReferenceRefreshResult:
    """Seed unknown fields and corroborate later reference changes with local news."""

    refs = list(references)
    seeded = unchanged = change_candidates = promoted = held = skipped = considered = 0

    # A changed current-office assertion needs Wikidata + two independent news
    # publisher groups. A future authoritative primary-source connector can still
    # promote alone through the authoritative-source gate.
    updater = AutonomousKnowledgeUpdater(
        store,
        policy=PromotionPolicy(
            min_confidence=0.80,
            min_independent_source_groups=3,
            min_authoritative_sources=1,
        ),
    )

    for reference in refs:
        if pycountry.countries.get(alpha_3=reference.iso3) is None:
            continue
        considered += 1
        entity_id = f"country:{reference.iso3}"
        current = store.current_state(entity_id)

        wikidata_evidence = SourceEvidence(
            source="wikidata",
            source_ref=reference.source_ref,
            published_at=reference.retrieved_at,
            retrieved_at=reference.retrieved_at,
            independent_group="wikidata",
            authoritative=False,
        )

        for field, value in _reference_fields(reference).items():
            if value is None:
                skipped += 1
                continue

            existing = current.get(field)
            if existing is None:
                candidate = _candidate(
                    reference,
                    field=field,
                    value=value,
                    evidence=(wikidata_evidence,),
                    confidence=0.82,
                )
                _assertion_id, was_seeded = store.seed_reference_fact(candidate)
                seeded += int(was_seeded)
                continue

            if _canonical(existing.get("value")) == _canonical(value):
                unchanged += 1
                continue

            news_evidence = _news_evidence_for_change(
                news,
                reference=reference,
                field=field,
                value=value,
            )
            evidence = (wikidata_evidence, *news_evidence)
            confidence = min(0.96, 0.82 + 0.04 * len(news_evidence))
            candidate = _candidate(
                reference,
                field=field,
                value=value,
                evidence=evidence,
                confidence=confidence,
            )
            change_candidates += 1
            result = updater.process(candidate)
            if result.decision.action is PromotionAction.AUTO_PROMOTE_FACT:
                promoted += 1
            else:
                held += 1

    return PoliticalReferenceRefreshResult(
        references=len(refs),
        countries_considered=considered,
        fields_seeded=seeded,
        unchanged_fields=unchanged,
        change_candidates=change_candidates,
        promoted_changes=promoted,
        held_changes=held,
        skipped_fields=skipped,
    )


__all__ = [
    "PoliticalReferenceRefreshResult",
    "refresh_political_reference",
]
