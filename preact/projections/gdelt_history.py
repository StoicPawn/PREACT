"""PREACT projection of shared GDELT raw data.

GoldenBull consumes the same provider through its own financial projection.
This module adds historical time/provenance semantics without changing raw data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Iterable, Mapping

from preact.history.documents import HistoricalDocument, TextAvailability
from preact.history.schema import EvidenceClass, Provenance, TemporalRecord


def _parse_gdelt_datetime(value: object, *, date_only: bool = False) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    formats = ["%Y%m%d"] if date_only else [
        "%Y%m%d%H%M%S",
        "%Y%m%dT%H%M%SZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _event_id(row: Mapping[str, Any]) -> str:
    provider_id = str(row.get("GLOBALEVENTID") or "").strip()
    if provider_id:
        return f"gdelt:event:{provider_id}"
    digest = sha256(repr(sorted(row.items())).encode("utf-8")).hexdigest()
    return f"gdelt:event:sha256:{digest}"


def gdelt_event_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    acquired_at: datetime,
    snapshot_checksum: str | None = None,
    fips_to_iso3: Mapping[str, str] | None = None,
) -> list[TemporalRecord]:
    """Convert GDELT Event rows into bitemporal evidence.

    SQLDATE is event valid time. DATEADDED is used as the earliest provider
    observation/knowledge time when available.
    """

    output: list[TemporalRecord] = []
    for row in rows:
        valid_from = _parse_gdelt_datetime(row.get("SQLDATE"), date_only=True)
        if valid_from is None:
            continue
        known_at = _parse_gdelt_datetime(row.get("DATEADDED")) or acquired_at
        if known_at < valid_from:
            known_at = valid_from

        country = str(
            row.get("ActionGeo_CountryCode")
            or row.get("Actor1CountryCode")
            or row.get("Actor2CountryCode")
            or "GLOBAL"
        ).strip().upper() or "GLOBAL"
        iso3 = (fips_to_iso3 or {}).get(country)
        entity_id = f"iso3:{iso3}" if iso3 else f"gdelt_fips:{country}"
        event_id = _event_id(row)
        source_url = str(row.get("SOURCEURL") or "").strip()

        value = {
            "event_code": row.get("EventCode"),
            "event_base_code": row.get("EventBaseCode"),
            "event_root_code": row.get("EventRootCode"),
            "quad_class": row.get("QuadClass"),
            "goldstein_scale": row.get("GoldsteinScale"),
            "avg_tone": row.get("AvgTone"),
            "num_mentions": row.get("NumMentions"),
            "num_sources": row.get("NumSources"),
            "num_articles": row.get("NumArticles"),
            "actor1": row.get("Actor1Name"),
            "actor2": row.get("Actor2Name"),
            "action_geo_name": row.get("ActionGeo_FullName"),
            "latitude": row.get("ActionGeo_Lat"),
            "longitude": row.get("ActionGeo_Long"),
        }
        output.append(
            TemporalRecord(
                record_id=event_id,
                entity_id=entity_id,
                variable="gdelt_event",
                value=value,
                valid_from=valid_from,
                known_at=known_at,
                evidence_class=EvidenceClass.OBSERVATION,
                provenance=Provenance(
                    source="gdelt",
                    source_ref=source_url or event_id,
                    retrieved_at=acquired_at,
                    dataset_version="GDELT 2.0 Events",
                    transform="raw event row -> PREACT bitemporal event",
                ),
                attributes={
                    "snapshot_checksum": snapshot_checksum,
                    "source_url": source_url,
                    "provider_country_code": country,
                    "provider_country_code_scheme": "FIPS10-4",
                    "normalized_iso3": iso3,
                },
            )
        )
    return output


def gdelt_article_documents(
    articles: Iterable[Mapping[str, Any]],
    *,
    acquired_at: datetime,
    snapshot_checksum: str | None = None,
) -> list[HistoricalDocument]:
    """Convert GDELT DOC ArticleList metadata without persisting article text."""

    documents: list[HistoricalDocument] = []
    for article in articles:
        url = str(article.get("url") or "").strip()
        title = str(article.get("title") or url or "Untitled").strip()
        published_at = (
            _parse_gdelt_datetime(article.get("seendate"))
            or _parse_gdelt_datetime(article.get("date"))
            or acquired_at
        )
        known_at = published_at
        digest = sha256((url or repr(sorted(article.items()))).encode("utf-8")).hexdigest()

        source_country = str(article.get("sourcecountry") or "").strip().upper()
        entity_ids = (
            (f"gdelt_source_country:{source_country}",)
            if source_country
            else ()
        )
        documents.append(
            HistoricalDocument(
                document_id=f"gdelt:document:{digest}",
                source_id="gdelt",
                source_ref=url or digest,
                title=title,
                published_at=published_at,
                known_at=known_at,
                acquired_at=acquired_at,
                url=url or None,
                language=str(article.get("language") or "").strip() or None,
                entity_ids=entity_ids,
                text=None,
                text_availability=TextAvailability.METADATA_ONLY,
                snapshot_checksum=snapshot_checksum,
                attributes={
                    "domain": article.get("domain"),
                    "source_country": source_country or None,
                    "social_image": article.get("socialimage"),
                    "tone": article.get("tone"),
                },
            )
        )
    return documents
