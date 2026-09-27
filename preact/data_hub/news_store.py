"""Provider-neutral local news archive for the ACEPC Shared Data Hub.

The store keeps metadata and provenance only. External acquisition is performed by
scheduled Shared Data Hub jobs; product APIs query this local archive and therefore
never need to contact Google News or GDELT directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import re
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import duckdb

from .entity_resolution import resolve_country_mentions


_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
}
_SPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_time(value: object, fallback: datetime) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        return _utc(fallback)
    for candidate in (raw, raw.replace("Z", "+00:00")):
        try:
            stamp = datetime.fromisoformat(candidate)
            return _utc(stamp)
        except ValueError:
            pass
    # GDELT DOC commonly uses compact timestamps.
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return _utc(fallback)


def _normalized_url(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    if not parts.scheme or not parts.netloc:
        return raw
    query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower().removeprefix("www."),
            parts.path.rstrip("/") or "/",
            urlencode(query),
            "",
        )
    )


def _fold(value: object) -> str:
    text = _SPACE.sub(" ", str(value or "").strip().lower())
    return _NON_WORD.sub(" ", text).strip()


def _canonical_title(article: Mapping[str, Any]) -> str:
    raw = str(article.get("title") or "").strip()
    publisher = str(article.get("publisher") or "").strip()
    if publisher:
        suffix = f" - {publisher}"
        if raw.lower().endswith(suffix.lower()):
            raw = raw[: -len(suffix)].rstrip()
    return _fold(raw)


def _canonical_key(article: Mapping[str, Any], published_at: datetime) -> str:
    title = _canonical_title(article)
    publisher = _fold(article.get("domain") or article.get("publisher"))
    # Title/publisher/day is intentionally the first cross-provider key. A direct
    # canonical URL is still retained for exact duplicate detection and navigation.
    material = "|".join([title, publisher, published_at.date().isoformat()])
    if not title:
        material = _normalized_url(article.get("url")) or repr(sorted(article.items()))
    return sha256(material.encode("utf-8")).hexdigest()


class SharedNewsStore:
    def __init__(self, path: str | Path = "data/shared_hub/shared_news.duckdb") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self):
        return duckdb.connect(self.path)

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shared_news_articles (
                    article_id VARCHAR PRIMARY KEY,
                    canonical_key VARCHAR NOT NULL,
                    title VARCHAR NOT NULL,
                    published_at TIMESTAMPTZ NOT NULL,
                    first_seen_at TIMESTAMPTZ NOT NULL,
                    last_seen_at TIMESTAMPTZ NOT NULL,
                    url VARCHAR,
                    normalized_url VARCHAR,
                    publisher VARCHAR,
                    domain VARCHAR,
                    language VARCHAR,
                    snippet VARCHAR,
                    image_url VARCHAR,
                    metadata_only BOOLEAN NOT NULL DEFAULT TRUE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shared_news_observations (
                    observation_id VARCHAR PRIMARY KEY,
                    article_id VARCHAR NOT NULL,
                    provider VARCHAR NOT NULL,
                    source_ref VARCHAR NOT NULL,
                    retrieved_at TIMESTAMPTZ NOT NULL,
                    snapshot_checksum VARCHAR,
                    feed_id VARCHAR,
                    raw_url VARCHAR,
                    title VARCHAR,
                    published_at TIMESTAMPTZ,
                    publisher VARCHAR,
                    domain VARCHAR,
                    language VARCHAR,
                    snippet VARCHAR,
                    image_url VARCHAR,
                    entity_resolution_done BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            for column, sql_type in (
                ("title", "VARCHAR"),
                ("published_at", "TIMESTAMPTZ"),
                ("publisher", "VARCHAR"),
                ("domain", "VARCHAR"),
                ("language", "VARCHAR"),
                ("snippet", "VARCHAR"),
                ("image_url", "VARCHAR"),
                ("entity_resolution_done", "BOOLEAN DEFAULT FALSE"),
            ):
                conn.execute(
                    f"ALTER TABLE shared_news_observations "
                    f"ADD COLUMN IF NOT EXISTS {column} {sql_type}"
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shared_news_entity_mentions (
                    mention_id VARCHAR PRIMARY KEY,
                    observation_id VARCHAR NOT NULL,
                    article_id VARCHAR NOT NULL,
                    entity_id VARCHAR NOT NULL,
                    entity_type VARCHAR NOT NULL,
                    resolver VARCHAR NOT NULL,
                    confidence DOUBLE NOT NULL,
                    matched_alias VARCHAR,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shared_news_entity "
                "ON shared_news_entity_mentions(entity_id, observation_id, confidence)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shared_news_time "
                "ON shared_news_articles(published_at, last_seen_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shared_news_provider "
                "ON shared_news_observations(provider, retrieved_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shared_news_article_obs "
                "ON shared_news_observations(article_id, retrieved_at)"
            )

    @staticmethod
    def _article_id(canonical_key: str) -> str:
        return "sna_" + canonical_key[:24]

    @staticmethod
    def _observation_id(
        *,
        provider: str,
        article_id: str,
        source_ref: str,
        retrieved_at: datetime,
        snapshot_checksum: str | None,
        feed_id: str | None,
    ) -> str:
        material = "|".join(
            [
                provider,
                article_id,
                source_ref,
                _utc(retrieved_at).isoformat(),
                snapshot_checksum or "",
                feed_id or "",
            ]
        )
        return "sno_" + sha256(material.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _mention_id(
        observation_id: str,
        entity_id: str,
        resolver: str,
        matched_alias: str,
    ) -> str:
        material = "|".join(
            [observation_id, entity_id, resolver, matched_alias]
        )
        return "snm_" + sha256(material.encode("utf-8")).hexdigest()[:24]

    def _resolve_observation_countries(
        self,
        conn,
        *,
        observation_id: str,
        article_id: str,
        title: str,
        snippet: str | None,
    ) -> int:
        inserted = 0
        mentions = resolve_country_mentions(title=title, snippet=snippet)
        for mention in mentions:
            mention_id = self._mention_id(
                observation_id,
                mention.entity_id,
                mention.resolver,
                mention.matched_alias,
            )
            if conn.execute(
                "SELECT 1 FROM shared_news_entity_mentions WHERE mention_id=?",
                [mention_id],
            ).fetchone():
                continue
            conn.execute(
                """
                INSERT INTO shared_news_entity_mentions(
                    mention_id,observation_id,article_id,entity_id,entity_type,
                    resolver,confidence,matched_alias
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                [
                    mention_id,
                    observation_id,
                    article_id,
                    mention.entity_id,
                    "country",
                    mention.resolver,
                    float(mention.confidence),
                    mention.matched_alias,
                ],
            )
            inserted += 1
        conn.execute(
            "UPDATE shared_news_observations "
            "SET entity_resolution_done=TRUE WHERE observation_id=?",
            [observation_id],
        )
        return inserted

    def backfill_country_mentions(self, *, limit: int = 10000) -> dict[str, int]:
        """Resolve country mentions for observations created before resolver v1."""

        safe_limit = max(1, min(int(limit), 100000))
        processed = 0
        inserted = 0
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT observation_id,article_id,title,snippet
                FROM shared_news_observations
                WHERE COALESCE(entity_resolution_done,FALSE)=FALSE
                ORDER BY retrieved_at,observation_id
                LIMIT ?
                """,
                [safe_limit],
            ).fetchall()
            for observation_id, article_id, title, snippet in rows:
                inserted += self._resolve_observation_countries(
                    conn,
                    observation_id=str(observation_id),
                    article_id=str(article_id),
                    title=str(title or ""),
                    snippet=(str(snippet) if snippet is not None else None),
                )
                processed += 1
        return {
            "processed_observations": processed,
            "inserted_mentions": inserted,
        }

    def upsert_articles(
        self,
        *,
        provider: str,
        articles: Iterable[Mapping[str, Any]],
        retrieved_at: datetime,
        snapshot_checksum: str | None = None,
        feed_id: str | None = None,
    ) -> dict[str, int]:
        provider = str(provider).strip()
        if not provider:
            raise ValueError("provider is required")
        retrieved = _utc(retrieved_at)
        inserted_articles = 0
        inserted_observations = 0

        with self.connect() as conn:
            for article in articles:
                title = str(article.get("title") or "").strip()
                raw_url = str(article.get("url") or "").strip()
                if not title or not raw_url:
                    continue

                published = _parse_time(
                    article.get("seendate") or article.get("date") or article.get("published_at"),
                    retrieved,
                )
                canonical_key = _canonical_key(article, published)
                article_id = self._article_id(canonical_key)
                normalized_url = _normalized_url(raw_url)
                publisher = str(article.get("publisher") or "").strip() or None
                domain = str(article.get("domain") or "").strip().lower().removeprefix("www.") or None
                language = str(article.get("language") or "").strip() or None
                snippet = str(article.get("snippet") or article.get("description") or "").strip() or None
                image_url = str(article.get("socialimage") or article.get("image_url") or "").strip() or None

                existing = conn.execute(
                    "SELECT 1 FROM shared_news_articles WHERE article_id=?",
                    [article_id],
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO shared_news_articles(
                            article_id,canonical_key,title,published_at,first_seen_at,
                            last_seen_at,url,normalized_url,publisher,domain,language,
                            snippet,image_url,metadata_only
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,TRUE)
                        """,
                        [
                            article_id,
                            canonical_key,
                            title,
                            published,
                            retrieved,
                            retrieved,
                            raw_url,
                            normalized_url,
                            publisher,
                            domain,
                            language,
                            snippet,
                            image_url,
                        ],
                    )
                    inserted_articles += 1
                else:
                    conn.execute(
                        """
                        UPDATE shared_news_articles
                        SET last_seen_at=?,
                            snippet=COALESCE(snippet, ?),
                            image_url=COALESCE(image_url, ?),
                            publisher=COALESCE(publisher, ?),
                            domain=COALESCE(domain, ?),
                            language=COALESCE(language, ?)
                        WHERE article_id=?
                        """,
                        [
                            retrieved,
                            snippet,
                            image_url,
                            publisher,
                            domain,
                            language,
                            article_id,
                        ],
                    )

                source_ref = normalized_url or raw_url
                observation_id = self._observation_id(
                    provider=provider,
                    article_id=article_id,
                    source_ref=source_ref,
                    retrieved_at=retrieved,
                    snapshot_checksum=snapshot_checksum,
                    feed_id=feed_id,
                )
                existing_observation = conn.execute(
                    "SELECT COALESCE(entity_resolution_done,FALSE) "
                    "FROM shared_news_observations WHERE observation_id=?",
                    [observation_id],
                ).fetchone()
                if existing_observation:
                    if not bool(existing_observation[0]):
                        self._resolve_observation_countries(
                            conn,
                            observation_id=observation_id,
                            article_id=article_id,
                            title=title,
                            snippet=snippet,
                        )
                    continue
                conn.execute(
                    """
                    INSERT INTO shared_news_observations(
                        observation_id,article_id,provider,source_ref,retrieved_at,
                        snapshot_checksum,feed_id,raw_url,title,published_at,
                        publisher,domain,language,snippet,image_url,
                        entity_resolution_done
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,FALSE)
                    """,
                    [
                        observation_id,
                        article_id,
                        provider,
                        source_ref,
                        retrieved,
                        snapshot_checksum,
                        feed_id,
                        raw_url,
                        title,
                        published,
                        publisher,
                        domain,
                        language,
                        snippet,
                        image_url,
                    ],
                )
                inserted_observations += 1
                self._resolve_observation_countries(
                    conn,
                    observation_id=observation_id,
                    article_id=article_id,
                    title=title,
                    snippet=snippet,
                )

        return {
            "inserted_articles": inserted_articles,
            "inserted_observations": inserted_observations,
        }

    def latest(
        self,
        *,
        limit: int = 50,
        query: str | None = None,
        provider: str | None = None,
        language: str | None = None,
        feed_id: str | None = None,
        entity_id: str | None = None,
        min_entity_confidence: float = 0.80,
        known_cutoff: datetime | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        clauses = []
        params: list[Any] = []

        if known_cutoff is not None:
            clauses.append("o.retrieved_at <= ?")
            params.append(_utc(known_cutoff))
        if provider:
            clauses.append("o.provider = ?")
            params.append(str(provider))
        if language:
            clauses.append("lower(coalesce(o.language,'')) = ?")
            params.append(str(language).lower())
        if feed_id:
            clauses.append("o.feed_id = ?")
            params.append(str(feed_id))
        if entity_id:
            clauses.append(
                "EXISTS (SELECT 1 FROM shared_news_entity_mentions m "
                "WHERE m.observation_id=o.observation_id "
                "AND m.entity_id=? AND m.confidence>=?)"
            )
            params.extend([str(entity_id), float(min_entity_confidence)])
        if query:
            needle = f"%{str(query).strip().lower()}%"
            clauses.append(
                "(lower(coalesce(o.title,'')) LIKE ? OR lower(coalesce(o.snippet,'')) LIKE ? "
                "OR lower(coalesce(o.publisher,'')) LIKE ?)"
            )
            params.extend([needle, needle, needle])

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(safe_limit)

        with self.connect() as conn:
            cursor = conn.execute(
                f"""
                WITH filtered AS (
                    SELECT
                        a.article_id,
                        a.first_seen_at,
                        a.metadata_only,
                        o.title,
                        o.published_at,
                        o.raw_url AS url,
                        o.publisher,
                        o.domain,
                        o.language,
                        o.snippet,
                        o.image_url,
                        o.provider,
                        o.feed_id,
                        o.retrieved_at,
                        o.snapshot_checksum,
                        ROW_NUMBER() OVER (
                            PARTITION BY a.article_id
                            ORDER BY o.retrieved_at DESC, o.provider
                        ) AS rn,
                        COUNT(*) OVER (PARTITION BY a.article_id) AS observation_count,
                        MAX(o.retrieved_at) OVER (
                            PARTITION BY a.article_id
                        ) AS last_seen_at
                    FROM shared_news_articles a
                    JOIN shared_news_observations o USING(article_id)
                    {where}
                )
                SELECT article_id,title,published_at,first_seen_at,last_seen_at,url,
                       publisher,domain,language,snippet,image_url,provider,feed_id,
                       retrieved_at,snapshot_checksum,observation_count,metadata_only
                FROM filtered
                WHERE rn=1
                ORDER BY published_at DESC, retrieved_at DESC
                LIMIT ?
                """,
                params,
            )
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            articles = int(conn.execute("SELECT COUNT(*) FROM shared_news_articles").fetchone()[0])
            observations = int(conn.execute("SELECT COUNT(*) FROM shared_news_observations").fetchone()[0])
            mentions = int(conn.execute("SELECT COUNT(*) FROM shared_news_entity_mentions").fetchone()[0])
            unresolved = int(
                conn.execute(
                    "SELECT COUNT(*) FROM shared_news_observations "
                    "WHERE COALESCE(entity_resolution_done,FALSE)=FALSE"
                ).fetchone()[0]
            )
            rows = conn.execute(
                """
                SELECT provider,COUNT(*) AS n,MAX(retrieved_at) AS newest
                FROM shared_news_observations
                GROUP BY provider
                ORDER BY provider
                """
            ).fetchall()
        return {
            "articles": articles,
            "observations": observations,
            "entity_mentions": mentions,
            "unresolved_observations": unresolved,
            "providers": {
                str(provider): {"observations": int(count), "newest_retrieved_at": newest}
                for provider, count, newest in rows
            },
        }


__all__ = ["SharedNewsStore"]
