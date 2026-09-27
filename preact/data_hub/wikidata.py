"""Shared Wikidata country/political reference acquisition.

The Shared Data Hub owns the external request and immutable snapshot. PREACT uses
this as a curated reference baseline, not as an authoritative substitute for
official constitutional/government sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable
from urllib.parse import urlparse

from .gateway import ProviderResponse, SharedProviderGateway

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

_COUNTRY_POLITICS_QUERY = r"""
SELECT ?iso3 ?country ?countryLabel
       ?capital ?capitalLabel
       ?governmentForm ?governmentFormLabel
       ?headOfState ?headOfStateLabel
       ?headOfGovernment ?headOfGovernmentLabel
       ?officialLanguage ?officialLanguageLabel
       ?officialWebsite
WHERE {
  ?country wdt:P298 ?iso3 .

  OPTIONAL { ?country wdt:P36 ?capital . }
  OPTIONAL { ?country wdt:P122 ?governmentForm . }

  OPTIONAL {
    ?country p:P35 ?hosStatement .
    ?hosStatement ps:P35 ?headOfState .
    OPTIONAL { ?hosStatement pq:P582 ?hosEnd . }
    FILTER(!BOUND(?hosEnd) || ?hosEnd > NOW())
  }

  OPTIONAL {
    ?country p:P6 ?hogStatement .
    ?hogStatement ps:P6 ?headOfGovernment .
    OPTIONAL { ?hogStatement pq:P582 ?hogEnd . }
    FILTER(!BOUND(?hogEnd) || ?hogEnd > NOW())
  }

  OPTIONAL { ?country wdt:P37 ?officialLanguage . }
  OPTIONAL { ?country wdt:P856 ?officialWebsite . }

  FILTER(STRLEN(STR(?iso3)) = 3)

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
  }
}
"""


@dataclass(frozen=True)
class WikidataCountryReference:
    iso3: str
    qid: str
    country_label: str
    capital: tuple[str, ...]
    government_forms: tuple[str, ...]
    heads_of_state: tuple[str, ...]
    heads_of_government: tuple[str, ...]
    official_languages: tuple[str, ...]
    official_websites: tuple[str, ...]
    retrieved_at: datetime
    snapshot_checksum: str | None

    @property
    def source_ref(self) -> str:
        return f"https://www.wikidata.org/wiki/{self.qid}"


def _binding_value(binding: dict[str, Any], key: str) -> str | None:
    item = binding.get(key)
    if not isinstance(item, dict):
        return None
    value = str(item.get("value") or "").strip()
    return value or None


def _qid(uri: str | None) -> str | None:
    if not uri:
        return None
    value = str(uri).rstrip("/").rsplit("/", 1)[-1]
    return value if value.startswith("Q") else None


def parse_country_reference_payload(
    payload: dict[str, Any],
    *,
    retrieved_at: datetime,
    snapshot_checksum: str | None = None,
) -> list[WikidataCountryReference]:
    rows = payload.get("results", {}).get("bindings", [])
    if not isinstance(rows, list):
        return []

    buckets: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        iso3 = (_binding_value(raw, "iso3") or "").upper()
        qid = _qid(_binding_value(raw, "country"))
        if len(iso3) != 3 or not qid:
            continue

        bucket = buckets.setdefault(
            iso3,
            {
                "qid": qid,
                "country_label": _binding_value(raw, "countryLabel") or iso3,
                "capital": set(),
                "government_forms": set(),
                "heads_of_state": set(),
                "heads_of_government": set(),
                "official_languages": set(),
                "official_websites": set(),
            },
        )

        for key, target in (
            ("capitalLabel", "capital"),
            ("governmentFormLabel", "government_forms"),
            ("headOfStateLabel", "heads_of_state"),
            ("headOfGovernmentLabel", "heads_of_government"),
            ("officialLanguageLabel", "official_languages"),
            ("officialWebsite", "official_websites"),
        ):
            value = _binding_value(raw, key)
            if value:
                bucket[target].add(value)

    result: list[WikidataCountryReference] = []
    for iso3 in sorted(buckets):
        item = buckets[iso3]
        result.append(
            WikidataCountryReference(
                iso3=iso3,
                qid=str(item["qid"]),
                country_label=str(item["country_label"]),
                capital=tuple(sorted(item["capital"])),
                government_forms=tuple(sorted(item["government_forms"])),
                heads_of_state=tuple(sorted(item["heads_of_state"])),
                heads_of_government=tuple(sorted(item["heads_of_government"])),
                official_languages=tuple(sorted(item["official_languages"])),
                official_websites=tuple(sorted(item["official_websites"])),
                retrieved_at=retrieved_at,
                snapshot_checksum=snapshot_checksum,
            )
        )
    return result


def fetch_country_references(
    gateway: SharedProviderGateway,
    *,
    ttl_seconds: int = 7200,
) -> tuple[list[WikidataCountryReference], ProviderResponse]:
    """Fetch one small global country-reference slice through the shared gateway."""

    response = gateway.get_json(
        source_id="wikidata",
        operation="country_political_reference",
        url=WIKIDATA_SPARQL_URL,
        params={"query": _COUNTRY_POLITICS_QUERY, "format": "json"},
        ttl_seconds=max(0, int(ttl_seconds)),
        minimum_interval_seconds=5.0,
        timeout_seconds=90.0,
        headers={"Accept": "application/sparql-results+json"},
    )
    payload = response.payload if isinstance(response.payload, dict) else {}
    references = parse_country_reference_payload(
        payload,
        retrieved_at=response.retrieved_at,
        snapshot_checksum=response.snapshot_checksum,
    )
    return references, response


__all__ = [
    "WIKIDATA_SPARQL_URL",
    "WikidataCountryReference",
    "fetch_country_references",
    "parse_country_reference_payload",
]
