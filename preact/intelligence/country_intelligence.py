"""Assemble an evidence-first country intelligence profile from local PREACT stores.

The assembler never invents missing political facts. It exposes a stable schema,
the available point-in-time assertions, recent provider-derived events, locally
archived news metadata and socioeconomic observations already present in PREACT.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pycountry

from preact.data_hub.news_store import SharedNewsStore
from preact.history.warehouse import HistoricalWarehouse
from preact.history.world_knowledge_store import WorldKnowledgeStore
from preact.intelligence.country_profile import COUNTRY_INDICATORS


@dataclass(frozen=True)
class CountryFieldSpec:
    key: str
    label: str
    aliases: tuple[str, ...] = ()
    description: str | None = None


POLITICAL_SYSTEM_FIELDS: tuple[CountryFieldSpec, ...] = (
    CountryFieldSpec("state_form", "Form of state"),
    CountryFieldSpec("government_form", "Form of government"),
    CountryFieldSpec("constitutional_framework", "Constitutional framework", ("constitution",)),
    CountryFieldSpec("head_of_state_role", "Head of state: constitutional role"),
    CountryFieldSpec("head_of_government_role", "Head of government: constitutional role"),
    CountryFieldSpec("executive_selection", "How the executive is selected"),
    CountryFieldSpec("government_formation", "How a government is formed"),
    CountryFieldSpec("government_removal", "How a government can fall or be removed"),
    CountryFieldSpec("legislature", "Legislature / chambers"),
    CountryFieldSpec("electoral_system", "Electoral system"),
    CountryFieldSpec("judiciary", "Judiciary"),
    CountryFieldSpec("constitutional_review", "Constitutional review"),
    CountryFieldSpec("territorial_structure", "Territorial structure / autonomy"),
    CountryFieldSpec("checks_and_balances", "Checks and balances"),
)

CURRENT_GOVERNMENT_FIELDS: tuple[CountryFieldSpec, ...] = (
    CountryFieldSpec("head_of_state", "Head of state", ("president", "monarch")),
    CountryFieldSpec(
        "head_of_government",
        "Head of government",
        ("prime_minister", "premier", "chancellor"),
    ),
    CountryFieldSpec("government", "Government"),
    CountryFieldSpec("governing_coalition", "Governing coalition", ("coalition",)),
    CountryFieldSpec("cabinet", "Cabinet"),
    CountryFieldSpec("opposition", "Parliamentary / principal opposition"),
    CountryFieldSpec("major_parties", "Major political parties"),
)

GOVERNANCE_DIMENSION_FIELDS: tuple[CountryFieldSpec, ...] = (
    CountryFieldSpec("electoral_competitiveness", "Electoral competitiveness"),
    CountryFieldSpec("civil_liberties", "Civil liberties"),
    CountryFieldSpec("rule_of_law", "Rule of law"),
    CountryFieldSpec("media_freedom", "Media freedom"),
    CountryFieldSpec("voice_accountability", "Voice and accountability"),
    CountryFieldSpec("institutional_constraints", "Institutional constraints"),
)

SOCIETY_FIELDS: tuple[CountryFieldSpec, ...] = (
    CountryFieldSpec("official_languages", "Official languages"),
    CountryFieldSpec("religions", "Religious composition"),
    CountryFieldSpec("age_structure", "Age structure"),
    CountryFieldSpec("education", "Education"),
    CountryFieldSpec("inequality", "Income / wealth inequality"),
    CountryFieldSpec("migration", "Migration"),
    CountryFieldSpec("employment_structure", "Employment structure"),
    CountryFieldSpec("social_mobility", "Social mobility"),
    CountryFieldSpec("family_structure", "Family structure"),
    CountryFieldSpec("public_opinion", "Public opinion / survey evidence"),
)

HISTORY_FIELDS: tuple[CountryFieldSpec, ...] = (
    CountryFieldSpec("history_origins", "Origins / pre-state background"),
    CountryFieldSpec("history_state_formation", "State formation"),
    CountryFieldSpec("history_19th_century", "Nineteenth century"),
    CountryFieldSpec("history_20th_century", "Twentieth century"),
    CountryFieldSpec("history_21st_century", "Twenty-first century"),
    CountryFieldSpec("history_recent", "Recent history"),
)


def _utc(value: datetime | None) -> datetime:
    stamp = value or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _country_identity(iso3: str) -> dict[str, Any]:
    country = pycountry.countries.get(alpha_3=iso3)
    if country is None:
        raise ValueError(f"Unknown ISO-3 country code: {iso3}")
    return {
        "iso3": iso3,
        "iso2": getattr(country, "alpha_2", None),
        "numeric": getattr(country, "numeric", None),
        "name": getattr(country, "common_name", getattr(country, "name", iso3)),
        "official_name": getattr(country, "official_name", getattr(country, "name", iso3)),
    }


def _resolve_assertion(
    state: Mapping[str, Mapping[str, Any]],
    spec: CountryFieldSpec,
) -> tuple[str | None, Mapping[str, Any] | None]:
    for key in (spec.key, *spec.aliases):
        value = state.get(key)
        if value is not None:
            return key, value
    return None, None


def _section(
    state: Mapping[str, Mapping[str, Any]],
    specs: Sequence[CountryFieldSpec],
) -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    known = 0
    for spec in specs:
        matched_key, assertion = _resolve_assertion(state, spec)
        if assertion is None:
            fields.append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "status": "unknown",
                    "value": None,
                    "description": spec.description,
                    "semantic_class": "FACT",
                    "evidence": [],
                }
            )
            continue

        known += 1
        fields.append(
            {
                "key": spec.key,
                "matched_key": matched_key,
                "label": spec.label,
                "status": "known",
                "value": assertion.get("value"),
                "valid_from": assertion.get("valid_from"),
                "valid_to": assertion.get("valid_to"),
                "known_at": assertion.get("known_at"),
                "confidence": assertion.get("confidence"),
                "assertion_id": assertion.get("assertion_id"),
                "domain": assertion.get("domain"),
                "attributes": assertion.get("attributes") or {},
                "evidence": assertion.get("evidence") or [],
                "description": spec.description,
                "semantic_class": "FACT",
            }
        )

    total = len(specs)
    return {
        "fields": fields,
        "known_fields": known,
        "total_fields": total,
        "data_completeness": (known / total) if total else 1.0,
    }


def _decode_value(row: Mapping[str, Any]) -> Any:
    raw = row.get("value_json")
    if raw in (None, ""):
        return None
    if isinstance(raw, (dict, list, int, float, bool)):
        return raw
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return raw


def _local_indicators(
    warehouse: HistoricalWarehouse | None,
    *,
    iso3: str,
    cutoff: datetime,
) -> dict[str, Any]:
    if warehouse is None:
        return {"status": "unavailable", "indicators": {}}

    variables = [f"world_bank:{spec.code}" for spec in COUNTRY_INDICATORS.values()]
    try:
        rows = warehouse.latest_observations_as_of(
            cutoff=cutoff,
            entity_id=f"iso3:{iso3}",
            variables=variables,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}"[:1000],
            "indicators": {},
        }

    by_variable = {str(row.get("variable")): row for row in rows}
    result: dict[str, Any] = {}
    for key, spec in COUNTRY_INDICATORS.items():
        row = by_variable.get(f"world_bank:{spec.code}")
        if row is None:
            result[key] = {
                "code": spec.code,
                "label": spec.label,
                "status": "unknown",
                "value": None,
            }
            continue

        raw_value = _decode_value(row)
        display_value = None
        if isinstance(raw_value, (int, float)):
            display_value = float(raw_value) / float(spec.scale)
        result[key] = {
            "code": spec.code,
            "label": spec.label,
            "status": "known",
            "value": raw_value,
            "display_value": display_value,
            "suffix": spec.suffix,
            "valid_from": row.get("valid_from"),
            "known_at": row.get("known_at"),
            "source": row.get("source"),
            "source_ref": row.get("source_ref"),
            "retrieved_at": row.get("retrieved_at"),
            "evidence_class": row.get("evidence_class"),
        }

    return {
        "status": "ready" if rows else "empty",
        "indicators": result,
    }


def _recent_news(
    news: SharedNewsStore | None,
    *,
    country_name: str,
    cutoff: datetime,
    limit: int,
) -> dict[str, Any]:
    if news is None:
        return {
            "status": "unavailable",
            "match_mode": "country_name_text",
            "entity_resolution": "not_yet_applied",
            "articles": [],
        }
    try:
        rows = news.latest(
            query=country_name,
            known_cutoff=cutoff,
            limit=limit,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}"[:1000],
            "match_mode": "country_name_text",
            "entity_resolution": "not_yet_applied",
            "articles": [],
        }
    return {
        "status": "ready",
        "match_mode": "country_name_text",
        "entity_resolution": "not_yet_applied",
        "articles": rows,
    }


def _field_value(section: Mapping[str, Any], key: str) -> tuple[Any, str | None]:
    for item in section.get("fields", []):
        if item.get("key") == key and item.get("status") == "known":
            return item.get("value"), item.get("assertion_id")
    return None, None


def _human_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned[0]
        return ", ".join(cleaned[:-1]) + " and " + cleaned[-1]
    return str(value).strip() or None


def _derived_descriptions(
    *,
    identity: Mapping[str, Any],
    political: Mapping[str, Any],
    government: Mapping[str, Any],
    society: Mapping[str, Any],
    state: Mapping[str, Mapping[str, Any]],
    generated_at: datetime,
) -> dict[str, Any]:
    """Build conservative prose only from currently promoted factual assertions."""

    descriptions: dict[str, Any] = {}
    government_parts: list[str] = []
    government_support: list[str] = []

    head_state, head_state_id = _field_value(government, "head_of_state")
    head_gov, head_gov_id = _field_value(government, "head_of_government")
    coalition, coalition_id = _field_value(government, "governing_coalition")

    if _human_value(head_state):
        government_parts.append(f"The current head of state is {_human_value(head_state)}.")
    if _human_value(head_gov):
        government_parts.append(f"The current head of government is {_human_value(head_gov)}.")
    if _human_value(coalition):
        government_parts.append(f"The recorded governing coalition is {_human_value(coalition)}.")
    for assertion_id in (head_state_id, head_gov_id, coalition_id):
        if assertion_id:
            government_support.append(str(assertion_id))

    if government_parts:
        descriptions["current_government"] = {
            "text": " ".join(government_parts),
            "semantic_class": "FACT_DERIVED",
            "supporting_assertion_ids": government_support,
            "generated_at": generated_at,
        }

    political_parts: list[str] = []
    political_support: list[str] = []
    form, form_id = _field_value(political, "government_form")
    state_form, state_form_id = _field_value(political, "state_form")
    legislature, legislature_id = _field_value(political, "legislature")
    electoral, electoral_id = _field_value(political, "electoral_system")

    if _human_value(state_form):
        political_parts.append(f"The recorded form of state is {_human_value(state_form)}.")
    if _human_value(form):
        political_parts.append(f"The recorded form of government is {_human_value(form)}.")
    if _human_value(legislature):
        political_parts.append(f"The legislature is {_human_value(legislature)}.")
    if _human_value(electoral):
        political_parts.append(f"The recorded electoral system is {_human_value(electoral)}.")
    for assertion_id in (state_form_id, form_id, legislature_id, electoral_id):
        if assertion_id:
            political_support.append(str(assertion_id))

    if political_parts:
        descriptions["political_system"] = {
            "text": " ".join(political_parts),
            "semantic_class": "FACT_DERIVED",
            "supporting_assertion_ids": political_support,
            "generated_at": generated_at,
        }

    identity_parts: list[str] = []
    identity_support: list[str] = []
    capital = identity.get("capital")
    languages, languages_id = _field_value(society, "official_languages")
    website_assertion = state.get("official_website")

    if _human_value(capital):
        identity_parts.append(f"The capital is {_human_value(capital)}.")
        capital_assertion = state.get("capital")
        if capital_assertion and capital_assertion.get("assertion_id"):
            identity_support.append(str(capital_assertion["assertion_id"]))
    if _human_value(languages):
        identity_parts.append(f"Official languages recorded by PREACT are {_human_value(languages)}.")
    if languages_id:
        identity_support.append(str(languages_id))
    if website_assertion and website_assertion.get("value"):
        identity_parts.append(
            f"The recorded official website is {_human_value(website_assertion.get('value'))}."
        )
        if website_assertion.get("assertion_id"):
            identity_support.append(str(website_assertion["assertion_id"]))

    if identity_parts:
        descriptions["country_today"] = {
            "text": " ".join(identity_parts),
            "semantic_class": "FACT_DERIVED",
            "supporting_assertion_ids": identity_support,
            "generated_at": generated_at,
        }

    return descriptions


def assemble_country_intelligence_profile(
    iso3: str,
    *,
    world: WorldKnowledgeStore,
    as_of: datetime | None = None,
    known_cutoff: datetime | None = None,
    warehouse: HistoricalWarehouse | None = None,
    news: SharedNewsStore | None = None,
    event_limit: int = 100,
    news_limit: int = 30,
) -> dict[str, Any]:
    """Assemble a point-in-time country dossier from local PREACT evidence."""

    country_code = str(iso3).strip().upper()
    identity = _country_identity(country_code)
    world_time = _utc(as_of)
    cutoff = _utc(known_cutoff or world_time)
    entity_id = f"country:{country_code}"

    state = world.state_as_of(entity_id, world_time, known_cutoff=cutoff)
    events = world.event_timeline(
        entity_id,
        as_of=world_time,
        known_cutoff=cutoff,
        limit=max(1, min(int(event_limit), 500)),
    )
    gdelt_context = world.gkg_context_for_country(
        entity_id,
        as_of=world_time,
        known_cutoff=cutoff,
        limit=max(1, min(int(news_limit), 200)),
    )

    political = _section(state, POLITICAL_SYSTEM_FIELDS)
    government = _section(state, CURRENT_GOVERNMENT_FIELDS)
    governance = _section(state, GOVERNANCE_DIMENSION_FIELDS)
    society = _section(state, SOCIETY_FIELDS)
    history = _section(state, HISTORY_FIELDS)

    capital_assertion = state.get("capital")
    identity["capital"] = (
        capital_assertion.get("value") if capital_assertion is not None else None
    )
    identity["capital_status"] = "known" if capital_assertion is not None else "unknown"

    descriptions = _derived_descriptions(
        identity=identity,
        political=political,
        government=government,
        society=society,
        state=state,
        generated_at=cutoff,
    )

    data_gaps = {
        "political_system": [
            item["key"] for item in political["fields"] if item["status"] == "unknown"
        ],
        "current_government": [
            item["key"] for item in government["fields"] if item["status"] == "unknown"
        ],
        "governance_dimensions": [
            item["key"] for item in governance["fields"] if item["status"] == "unknown"
        ],
        "society": [
            item["key"] for item in society["fields"] if item["status"] == "unknown"
        ],
        "history": [
            item["key"] for item in history["fields"] if item["status"] == "unknown"
        ],
    }

    return {
        "entity_id": entity_id,
        "as_of": world_time,
        "known_cutoff": cutoff,
        "identity": identity,
        "descriptions": descriptions,
        "political_system": political,
        "current_government": government,
        "governance_dimensions": governance,
        "society": society,
        "history": history,
        "socioeconomic_indicators": _local_indicators(
            warehouse,
            iso3=country_code,
            cutoff=cutoff,
        ),
        "recent_events": {
            "semantic_class": "PROVIDER_DERIVED_OBSERVATION",
            "events": events,
        },
        "gdelt_context": {
            "semantic_class": "PROVIDER_DERIVED_CONTEXT",
            "documents": gdelt_context,
        },
        "recent_news": _recent_news(
            news,
            country_name=identity["name"],
            cutoff=cutoff,
            limit=max(1, min(int(news_limit), 200)),
        ),
        "data_gaps": data_gaps,
        "semantic_contract": {
            "facts_are_sourced": True,
            "unknown_fields_remain_unknown": True,
            "provider_events_are_not_promoted_facts": True,
            "gdelt_context_is_not_promoted_fact": True,
            "news_name_match_is_not_entity_resolution": True,
            "interpretations_are_separate": True,
            "forecasts_are_separate": True,
        },
    }


__all__ = [
    "CURRENT_GOVERNMENT_FIELDS",
    "GOVERNANCE_DIMENSION_FIELDS",
    "HISTORY_FIELDS",
    "POLITICAL_SYSTEM_FIELDS",
    "SOCIETY_FIELDS",
    "CountryFieldSpec",
    "assemble_country_intelligence_profile",
]
