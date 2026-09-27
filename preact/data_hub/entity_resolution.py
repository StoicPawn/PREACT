"""Conservative common entity resolution for Shared Data Hub news metadata.

The shared layer resolves only broadly reusable entities. Product-specific relevance
remains outside this module. Country mentions are stored per provider observation so
historical replay never benefits from entity resolution of metadata learned later.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
import unicodedata
from typing import Mapping

import pycountry


@dataclass(frozen=True)
class CountryMention:
    entity_id: str
    iso3: str
    name: str
    matched_alias: str
    confidence: float
    resolver: str = "shared_country_resolver_v1"


_AMBIGUOUS_NAMES = {
    "chad",
    "georgia",
    "jordan",
    "guinea",
    "jersey",
    "dominica",
}

_MANUAL_ALIASES: Mapping[str, tuple[tuple[str, float], ...]] = {
    "USA": (
        ("United States", 0.99),
        ("United States of America", 0.99),
        ("USA", 0.98),
        ("U.S.A.", 0.98),
        ("U.S.", 0.96),
    ),
    "GBR": (
        ("United Kingdom", 0.99),
        ("Great Britain", 0.96),
        ("Britain", 0.94),
        ("UK", 0.96),
        ("U.K.", 0.96),
    ),
    "RUS": (
        ("Russia", 0.99),
        ("Russian Federation", 0.99),
    ),
    "KOR": (
        ("South Korea", 0.99),
        ("Republic of Korea", 0.96),
    ),
    "PRK": (
        ("North Korea", 0.99),
        ("DPRK", 0.96),
    ),
    "CZE": (
        ("Czech Republic", 0.98),
        ("Czechia", 0.99),
    ),
    "TUR": (
        ("Turkey", 0.98),
        ("Türkiye", 0.99),
        ("Turkiye", 0.98),
    ),
    "ARE": (
        ("United Arab Emirates", 0.99),
        ("UAE", 0.97),
    ),
}


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value))
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold()).strip()


@lru_cache(maxsize=1)
def _country_aliases() -> tuple[tuple[str, str, str, float, bool], ...]:
    aliases: dict[tuple[str, str], tuple[str, str, str, float, bool]] = {}

    for country in pycountry.countries:
        iso3 = str(getattr(country, "alpha_3", "")).upper()
        if len(iso3) != 3:
            continue
        canonical = str(
            getattr(country, "common_name", getattr(country, "name", iso3))
        )
        candidates = {
            str(getattr(country, "name", "")).strip(),
            str(getattr(country, "official_name", "")).strip(),
            str(getattr(country, "common_name", "")).strip(),
        }
        for alias in candidates:
            if not alias:
                continue
            folded = _fold(alias)
            if len(folded) < 4:
                continue
            confidence = 0.74 if folded in _AMBIGUOUS_NAMES else 0.96
            aliases[(iso3, folded)] = (
                iso3,
                canonical,
                alias,
                confidence,
                False,
            )

    for iso3, rows in _MANUAL_ALIASES.items():
        country = pycountry.countries.get(alpha_3=iso3)
        canonical = (
            str(getattr(country, "common_name", getattr(country, "name", iso3)))
            if country is not None
            else iso3
        )
        for alias, confidence in rows:
            folded = _fold(alias)
            short_case_sensitive = len(folded.replace(" ", "")) <= 3
            aliases[(iso3, folded)] = (
                iso3,
                canonical,
                alias,
                float(confidence),
                short_case_sensitive,
            )

    # Prefer longer/more specific aliases first.
    return tuple(
        sorted(
            aliases.values(),
            key=lambda item: (len(_fold(item[2])), item[3]),
            reverse=True,
        )
    )


def _short_alias_present(raw_text: str, alias: str) -> bool:
    # Short acronyms such as US/UK/UAE must be visibly uppercase in the original
    # text to avoid matching ordinary words ("us") after case folding.
    compact = re.escape(alias.replace(".", ""))
    cleaned = raw_text.replace(".", "")
    return re.search(rf"(?<![A-Za-z0-9]){compact}(?![A-Za-z0-9])", cleaned) is not None


def resolve_country_mentions(
    *,
    title: str | None,
    snippet: str | None = None,
    min_confidence: float = 0.0,
) -> tuple[CountryMention, ...]:
    raw_text = " ".join(part for part in (str(title or ""), str(snippet or "")) if part)
    folded_text = f" {_fold(raw_text)} "
    if not folded_text.strip():
        return ()

    best_by_iso3: dict[str, CountryMention] = {}
    for iso3, name, alias, confidence, short_case_sensitive in _country_aliases():
        if confidence < float(min_confidence):
            continue
        folded_alias = _fold(alias)
        if not folded_alias:
            continue

        if short_case_sensitive:
            matched = _short_alias_present(raw_text, alias)
        else:
            matched = f" {folded_alias} " in folded_text
        if not matched:
            continue

        mention = CountryMention(
            entity_id=f"country:{iso3}",
            iso3=iso3,
            name=name,
            matched_alias=alias,
            confidence=confidence,
        )
        previous = best_by_iso3.get(iso3)
        if previous is None or mention.confidence > previous.confidence:
            best_by_iso3[iso3] = mention

    return tuple(
        sorted(
            best_by_iso3.values(),
            key=lambda item: (-item.confidence, item.iso3),
        )
    )


__all__ = ["CountryMention", "resolve_country_mentions"]
