"""Provider-format ingestion for GDELT data already acquired by ACEPC Shared Data Hub.

PREACT deliberately owns no GDELT network collector. This module only parses the
provider-native Events, Mentions and GKG files made available by the external hub.
"""

from __future__ import annotations

import csv
import io
import zipfile
from typing import Any, Iterable

_EVENT_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat",
    "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type",
    "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code",
    "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long",
    "Actor2Geo_FeatureID", "ActionGeo_Type", "ActionGeo_FullName",
    "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_ADM2Code",
    "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED",
    "SOURCEURL",
]

_MENTION_COLUMNS = [
    "GLOBALEVENTID", "EventTimeDate", "MentionTimeDate", "MentionType",
    "MentionSourceName", "MentionIdentifier", "SentenceID",
    "Actor1CharOffset", "Actor2CharOffset", "ActionCharOffset",
    "InRawText", "Confidence", "MentionDocLen", "MentionDocTone",
    "MentionDocTranslationInfo", "Extras",
]

_GKG_COLUMNS = [
    "GKGRECORDID", "V2DATE", "V2SOURCECOLLECTIONIDENTIFIER",
    "V2SOURCECOMMONNAME", "V2DOCUMENTIDENTIFIER", "V1COUNTS",
    "V2COUNTS", "V1THEMES", "V2ENHANCEDTHEMES", "V1LOCATIONS",
    "V2ENHANCEDLOCATIONS", "V1PERSONS", "V2ENHANCEDPERSONS",
    "V1ORGANIZATIONS", "V2ENHANCEDORGANIZATIONS", "V2TONE",
    "V2ENHANCEDDATES", "V2GCAM", "V2SHARINGIMAGE", "V2RELATEDIMAGES",
    "V2SOCIALIMAGEEMBEDS", "V2SOCIALVIDEOEMBEDS", "V2QUOTATIONS",
    "V2ALLNAMES", "V2AMOUNTS", "V2TRANSLATIONINFO", "V2EXTRASXML",
]


def _parse_zip_rows(
    payload: bytes,
    columns: list[str],
) -> list[dict[str, str]]:
    """Parse one provider TSV member without duplicating the full text in memory."""

    previous_limit = csv.field_size_limit()
    csv.field_size_limit(64 * 1024 * 1024)
    rows: list[dict[str, str]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if not names:
                return []
            with archive.open(names[0], "r") as member:
                with io.TextIOWrapper(
                    member,
                    encoding="utf-8",
                    errors="replace",
                    newline="",
                ) as text:
                    for values in csv.reader(text, delimiter="\t"):
                        if not values:
                            continue
                        padded = values + [""] * max(
                            0,
                            len(columns) - len(values),
                        )
                        rows.append(
                            dict(zip(columns, padded[: len(columns)]))
                        )
    finally:
        csv.field_size_limit(previous_limit)
    return rows


def parse_event_zip(payload: bytes) -> list[dict[str, str]]:
    return _parse_zip_rows(payload, _EVENT_COLUMNS)


def parse_mentions_zip(payload: bytes) -> list[dict[str, str]]:
    return _parse_zip_rows(payload, _MENTION_COLUMNS)


def parse_gkg_zip(payload: bytes) -> list[dict[str, str]]:
    return _parse_zip_rows(payload, _GKG_COLUMNS)
def _float(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def summarize_mentions(
    rows: Iterable[dict[str, str]],
) -> list[dict[str, Any]]:
    """Aggregate provider Mentions rows by GDELT event id.

    The result is evidence metadata. It is not a factual geopolitical claim.
    """

    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        event_id = str(row.get("GLOBALEVENTID") or "").strip()
        if not event_id:
            continue
        bucket = buckets.setdefault(
            event_id,
            {
                "provider_event_id": event_id,
                "mention_count": 0,
                "sources": set(),
                "confidences": [],
                "tones": [],
                "latest_mention_time": None,
            },
        )
        bucket["mention_count"] += 1
        source = str(row.get("MentionSourceName") or "").strip()
        if source:
            bucket["sources"].add(source)
        confidence = _float(row.get("Confidence"))
        if confidence is not None:
            bucket["confidences"].append(confidence)
        tone = _float(row.get("MentionDocTone"))
        if tone is not None:
            bucket["tones"].append(tone)
        stamp = str(row.get("MentionTimeDate") or "").strip()
        if stamp and (
            bucket["latest_mention_time"] is None
            or stamp > bucket["latest_mention_time"]
        ):
            bucket["latest_mention_time"] = stamp

    result: list[dict[str, Any]] = []
    for event_id in sorted(buckets):
        bucket = buckets[event_id]
        confidences = bucket["confidences"]
        tones = bucket["tones"]
        sources = sorted(bucket["sources"])
        result.append(
            {
                "provider_event_id": event_id,
                "mention_count": int(bucket["mention_count"]),
                "distinct_source_count": len(sources),
                "mention_sources": sources,
                "mean_confidence": (
                    sum(confidences) / len(confidences)
                    if confidences
                    else None
                ),
                "max_confidence": max(confidences) if confidences else None,
                "mean_document_tone": (
                    sum(tones) / len(tones) if tones else None
                ),
                "latest_mention_time": bucket["latest_mention_time"],
            }
        )
    return result


def _split_semicolon(value: object) -> list[str]:
    return [
        item.strip()
        for item in str(value or "").split(";")
        if item.strip()
    ]
def gkg_document_context(row: dict[str, str]) -> dict[str, Any]:
    """Normalize one GKG row into entity/theme context for PREACT consumers."""

    countries: set[str] = set()
    for location in _split_semicolon(row.get("V2ENHANCEDLOCATIONS")):
        fields = location.split("#")
        if len(fields) >= 3 and fields[2].strip():
            countries.add(fields[2].strip().upper())

    tone_fields = str(row.get("V2TONE") or "").split(",")
    overall_tone = _float(tone_fields[0]) if tone_fields else None

    return {
        "gkg_record_id": str(row.get("GKGRECORDID") or "").strip(),
        "provider_time": str(row.get("V2DATE") or "").strip(),
        "source": str(row.get("V2SOURCECOMMONNAME") or "").strip(),
        "document_url": str(row.get("V2DOCUMENTIDENTIFIER") or "").strip(),
        "country_codes": sorted(countries),
        "themes": _split_semicolon(row.get("V1THEMES")),
        "persons": _split_semicolon(row.get("V1PERSONS")),
        "organizations": _split_semicolon(row.get("V1ORGANIZATIONS")),
        "overall_tone": overall_tone,
        "all_names": _split_semicolon(row.get("V2ALLNAMES")),
    }


def normalize_gkg_documents(
    rows: Iterable[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        item
        for item in (gkg_document_context(row) for row in rows)
        if item["gkg_record_id"] and item["document_url"]
    ]


__all__ = [
    "_EVENT_COLUMNS",
    "_GKG_COLUMNS",
    "_MENTION_COLUMNS",
    "gkg_document_context",
    "normalize_gkg_documents",
    "parse_event_zip",
    "parse_gkg_zip",
    "parse_mentions_zip",
    "summarize_mentions",
]
