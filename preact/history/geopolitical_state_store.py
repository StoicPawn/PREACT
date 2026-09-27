"""Persistence/query layer for PREACT latent geopolitical relationship states."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb

from preact.intelligence.geopolitical_state import (
    EventImpact,
    MODEL_VERSION,
    RelationshipState,
)
from preact.intelligence.semantic_event_enrichment import SemanticEventEnrichment


class GeopoliticalStateStore:
    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = str(path)
        self.read_only = bool(read_only)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if not self.read_only:
            self._init_schema()

    def connect(self):
        return duckdb.connect(self.path, read_only=self.read_only)

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS geopolitical_event_impacts (
                    impact_id VARCHAR PRIMARY KEY,
                    provider_event_id VARCHAR NOT NULL,
                    pair_key VARCHAR NOT NULL,
                    source_iso3 VARCHAR NOT NULL,
                    target_iso3 VARCHAR NOT NULL,
                    event_time TIMESTAMPTZ NOT NULL,
                    known_at TIMESTAMPTZ NOT NULL,
                    event_code VARCHAR,
                    event_root_code VARCHAR,
                    semantic_tags_json VARCHAR NOT NULL,
                    vector_json VARCHAR NOT NULL,
                    severity DOUBLE NOT NULL,
                    confidence DOUBLE NOT NULL,
                    half_life_days DOUBLE NOT NULL,
                    persistence VARCHAR NOT NULL,
                    source_count INTEGER NOT NULL,
                    article_count INTEGER NOT NULL,
                    source_url VARCHAR,
                    evidence_event_ids_json VARCHAR NOT NULL DEFAULT '[]',
                    cluster_size INTEGER NOT NULL DEFAULT 1,
                    model_version VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_geo_impact_pair "
                "ON geopolitical_event_impacts(pair_key,event_time,known_at)"
            )
            conn.execute(
                "ALTER TABLE geopolitical_event_impacts ADD COLUMN IF NOT EXISTS "
                "evidence_event_ids_json VARCHAR DEFAULT '[]'"
            )
            conn.execute(
                "ALTER TABLE geopolitical_event_impacts ADD COLUMN IF NOT EXISTS "
                "cluster_size INTEGER DEFAULT 1"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS geopolitical_relation_states (
                    state_id VARCHAR PRIMARY KEY,
                    pair_key VARCHAR NOT NULL,
                    source_iso3 VARCHAR NOT NULL,
                    target_iso3 VARCHAR NOT NULL,
                    as_of TIMESTAMPTZ NOT NULL,
                    known_at TIMESTAMPTZ NOT NULL,
                    mode VARCHAR NOT NULL,
                    vector_json VARCHAR NOT NULL,
                    overall_score DOUBLE NOT NULL,
                    confidence DOUBLE NOT NULL,
                    coverage DOUBLE NOT NULL,
                    status VARCHAR NOT NULL,
                    trend VARCHAR NOT NULL,
                    live_delta DOUBLE,
                    event_count INTEGER NOT NULL,
                    source_count INTEGER NOT NULL,
                    last_event_at TIMESTAMPTZ,
                    structural_anchors_json VARCHAR NOT NULL,
                    evidence_event_ids_json VARCHAR NOT NULL,
                    model_version VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_geo_state_pair "
                "ON geopolitical_relation_states(pair_key,mode,as_of,known_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_geo_state_focal "
                "ON geopolitical_relation_states(source_iso3,target_iso3,mode,as_of)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS geopolitical_semantic_enrichments (
                    enrichment_id VARCHAR PRIMARY KEY,
                    provider_event_id VARCHAR NOT NULL,
                    known_at TIMESTAMPTZ NOT NULL,
                    model VARCHAR NOT NULL,
                    schema_version VARCHAR NOT NULL,
                    event_type VARCHAR NOT NULL,
                    direction VARCHAR NOT NULL,
                    severity DOUBLE NOT NULL,
                    persistence VARCHAR NOT NULL,
                    confidence DOUBLE NOT NULL,
                    dimension_modifiers_json VARCHAR NOT NULL,
                    rationale VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_geo_semantic_event "
                "ON geopolitical_semantic_enrichments(provider_event_id,known_at)"
            )

    @staticmethod
    def _json_list(raw: Any) -> list[Any]:
        if raw is None or raw == "":
            return []
        if isinstance(raw, list):
            return raw
        try:
            value = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        return value if isinstance(value, list) else []

    def load_recent_event_evidence(
        self,
        *,
        as_of: datetime,
        known_cutoff: datetime | None = None,
        lookback_days: int = 180,
        known_since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Load latest-known event observations and attached Mentions/GKG context."""

        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        cutoff = known_cutoff or as_of
        since = as_of - timedelta(days=int(lookback_days))
        if known_since is not None and known_since.tzinfo is None:
            raise ValueError("known_since must be timezone-aware")
        context_since = (
            known_since - timedelta(hours=1)
            if known_since is not None
            else None
        )

        with self.connect() as conn:
            tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
            if "world_event_observations" not in tables:
                return []

            has_mentions = "world_event_mentions" in tables
            has_gkg = "world_gkg_documents" in tables
            mention_cte = (
                """
                , mention_ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY provider_event_id
                               ORDER BY known_at DESC, created_at DESC
                           ) AS rn
                    FROM world_event_mentions
                    WHERE known_at <= ?
                )
                """
                if has_mentions
                else ""
            )
            gkg_cte = (
                """
                , gkg_ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY document_url
                               ORDER BY known_at DESC, created_at DESC
                           ) AS rn
                    FROM world_gkg_documents
                    WHERE known_at <= ?
                )
                """
                if has_gkg
                else ""
            )
            mention_select = (
                """
                , m.mention_count AS corroborating_mentions
                , m.distinct_source_count AS mention_source_count
                , m.max_confidence AS mention_max_confidence
                , m.mean_confidence AS mention_mean_confidence
                , m.mean_document_tone AS mention_document_tone
                """
                if has_mentions
                else """
                , NULL AS corroborating_mentions
                , NULL AS mention_source_count
                , NULL AS mention_max_confidence
                , NULL AS mention_mean_confidence
                , NULL AS mention_document_tone
                """
            )
            gkg_select = (
                """
                , g.themes_json
                , g.persons_json
                , g.organizations_json
                , g.overall_tone AS gkg_tone
                """
                if has_gkg
                else """
                , NULL AS themes_json
                , NULL AS persons_json
                , NULL AS organizations_json
                , NULL AS gkg_tone
                """
            )
            joins = ""
            if has_mentions:
                joins += (
                    " LEFT JOIN mention_ranked m"
                    " ON e.provider_event_id=m.provider_event_id AND m.rn=1"
                )
            if has_gkg:
                joins += (
                    " LEFT JOIN gkg_ranked g"
                    " ON e.source_url=g.document_url AND g.rn=1"
                )

            event_known_clause = " AND known_at >= ?" if known_since is not None else ""
            mention_since_clause = (
                " AND known_at >= ?" if context_since is not None else ""
            )
            gkg_since_clause = (
                " AND known_at >= ?" if context_since is not None else ""
            )

            params: list[Any] = [as_of, cutoff, since]
            if known_since is not None:
                params.append(known_since)
            if has_mentions:
                params.append(cutoff)
                if context_since is not None:
                    params.append(context_since)
            if has_gkg:
                params.append(cutoff)
                if context_since is not None:
                    params.append(context_since)

            mention_cte = mention_cte.replace(
                "WHERE known_at <= ?",
                "WHERE known_at <= ?" + mention_since_clause,
            )
            gkg_cte = gkg_cte.replace(
                "WHERE known_at <= ?",
                "WHERE known_at <= ?" + gkg_since_clause,
            )

            sql = f"""
                WITH event_ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY provider,provider_event_id
                               ORDER BY known_at DESC,created_at DESC
                           ) AS rn
                    FROM world_event_observations
                    WHERE event_time <= ?
                      AND known_at <= ?
                      AND event_time >= ?
                      {event_known_clause}
                )
                {mention_cte}
                {gkg_cte}
                SELECT
                    e.provider_event_id,e.event_time,e.known_at,
                    e.actor1_entity_id,e.actor2_entity_id,
                    e.event_code,e.event_base_code,e.event_root_code,
                    e.quad_class,e.goldstein,e.tone,e.num_mentions,
                    e.num_sources,e.num_articles,e.actor1_name,e.actor2_name,
                    e.action_location,e.source_url,e.snapshot_checksum
                    {mention_select}
                    {gkg_select}
                FROM event_ranked e
                {joins}
                WHERE e.rn=1
                  AND e.actor2_entity_id IS NOT NULL
                ORDER BY e.event_time,e.provider_event_id
            """
            cursor = conn.execute(sql, params)
            columns = [item[0] for item in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

        for row in rows:
            row["themes"] = self._json_list(row.pop("themes_json", None))
            row["persons"] = self._json_list(row.pop("persons_json", None))
            row["organizations"] = self._json_list(
                row.pop("organizations_json", None)
            )
        return rows

    @staticmethod
    def _impact_id(impact: EventImpact) -> str:
        payload = "|".join(
            [
                impact.provider_event_id,
                impact.known_at.isoformat(),
                impact.model_version,
                json.dumps(dict(impact.vector), sort_keys=True),
            ]
        )
        return "gpi_" + sha256(payload.encode("utf-8")).hexdigest()[:24]

    def record_impacts(self, impacts: Iterable[EventImpact]) -> int:
        rows = []
        for impact in impacts:
            rows.append(
                [
                    self._impact_id(impact),
                    impact.provider_event_id,
                    impact.pair_key,
                    impact.source_iso3,
                    impact.target_iso3,
                    impact.event_time,
                    impact.known_at,
                    impact.event_code,
                    impact.event_root_code,
                    json.dumps(list(impact.semantic_tags), sort_keys=True),
                    json.dumps(dict(impact.vector), sort_keys=True),
                    impact.severity,
                    impact.confidence,
                    impact.half_life_days,
                    impact.persistence,
                    impact.source_count,
                    impact.article_count,
                    impact.source_url,
                    json.dumps(list(impact.evidence_event_ids), sort_keys=True),
                    int(impact.cluster_size),
                    impact.model_version,
                ]
            )
        if not rows:
            return 0
        with self.connect() as conn:
            before = int(
                conn.execute(
                    "SELECT COUNT(*) FROM geopolitical_event_impacts"
                ).fetchone()[0]
            )
            conn.executemany(
                """
                INSERT OR IGNORE INTO geopolitical_event_impacts(
                    impact_id,provider_event_id,pair_key,source_iso3,target_iso3,
                    event_time,known_at,event_code,event_root_code,
                    semantic_tags_json,vector_json,severity,confidence,
                    half_life_days,persistence,source_count,article_count,
                    source_url,evidence_event_ids_json,cluster_size,model_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            after = int(
                conn.execute(
                    "SELECT COUNT(*) FROM geopolitical_event_impacts"
                ).fetchone()[0]
            )
        return max(0, after - before)

    @staticmethod
    def _state_id(state: RelationshipState) -> str:
        payload = "|".join(
            [
                state.pair_key,
                state.mode,
                state.as_of.isoformat(),
                state.known_at.isoformat(),
                state.model_version,
            ]
        )
        return "gps_" + sha256(payload.encode("utf-8")).hexdigest()[:24]

    def latest_impact_known_at(self) -> datetime | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT MAX(known_at) FROM geopolitical_event_impacts "
                "WHERE model_version=?",
                [MODEL_VERSION],
            ).fetchone()
        return row[0] if row and row[0] is not None else None

    def latest_impacts(
        self,
        *,
        as_of: datetime,
        lookback_days: int = 180,
    ) -> list[EventImpact]:
        """Load one latest interpretation per logical event cluster."""
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        since = as_of - timedelta(days=int(lookback_days))
        with self.connect() as conn:
            rows = conn.execute(
                """
                WITH ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY provider_event_id
                               ORDER BY known_at DESC,created_at DESC
                           ) AS rn
                    FROM geopolitical_event_impacts
                    WHERE event_time >= ?
                      AND event_time <= ?
                      AND known_at <= ?
                      AND model_version=?
                )
                SELECT provider_event_id,source_iso3,target_iso3,event_time,known_at,
                       event_code,event_root_code,semantic_tags_json,vector_json,
                       severity,confidence,half_life_days,persistence,source_count,
                       article_count,source_url,evidence_event_ids_json,cluster_size,
                       model_version
                FROM ranked WHERE rn=1
                ORDER BY event_time,provider_event_id
                """,
                [since, as_of, as_of, MODEL_VERSION],
            ).fetchall()
        return [
            EventImpact(
                provider_event_id=str(row[0]),
                source_iso3=str(row[1]),
                target_iso3=str(row[2]),
                event_time=row[3],
                known_at=row[4],
                event_code=row[5],
                event_root_code=row[6],
                semantic_tags=tuple(json.loads(row[7] or "[]")),
                vector=json.loads(row[8] or "{}"),
                severity=float(row[9]),
                confidence=float(row[10]),
                half_life_days=float(row[11]),
                persistence=str(row[12]),
                source_count=int(row[13]),
                article_count=int(row[14]),
                source_url=row[15],
                evidence_event_ids=tuple(json.loads(row[16] or "[]")),
                cluster_size=int(row[17] or 1),
                model_version=str(row[18]),
            )
            for row in rows
        ]

    def record_states(self, states: Iterable[RelationshipState]) -> int:
        rows = []
        for state in states:
            rows.append(
                [
                    self._state_id(state),
                    state.pair_key,
                    state.source_iso3,
                    state.target_iso3,
                    state.as_of,
                    state.known_at,
                    state.mode,
                    json.dumps(dict(state.vector), sort_keys=True),
                    state.overall_score,
                    state.confidence,
                    state.coverage,
                    state.status,
                    state.trend,
                    state.live_delta,
                    state.event_count,
                    state.source_count,
                    state.last_event_at,
                    json.dumps(list(state.structural_anchors), sort_keys=True),
                    json.dumps(list(state.evidence_event_ids), sort_keys=True),
                    state.model_version,
                ]
            )
        if not rows:
            return 0
        with self.connect() as conn:
            before = int(
                conn.execute(
                    "SELECT COUNT(*) FROM geopolitical_relation_states"
                ).fetchone()[0]
            )
            conn.executemany(
                """
                INSERT OR IGNORE INTO geopolitical_relation_states(
                    state_id,pair_key,source_iso3,target_iso3,as_of,known_at,
                    mode,vector_json,overall_score,confidence,coverage,status,
                    trend,live_delta,event_count,source_count,last_event_at,
                    structural_anchors_json,evidence_event_ids_json,model_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            after = int(
                conn.execute(
                    "SELECT COUNT(*) FROM geopolitical_relation_states"
                ).fetchone()[0]
            )
        return max(0, after - before)

    def latest_states(
        self,
        *,
        mode: str,
        as_of: datetime | None = None,
    ) -> list[RelationshipState]:
        """Return the complete latest state snapshot for the current model version."""
        if mode not in {"live", "canonical"}:
            raise ValueError("mode must be 'live' or 'canonical'")
        cutoff = as_of or datetime.now(timezone.utc)
        with self.connect() as conn:
            latest_run = conn.execute(
                """
                SELECT MAX(as_of)
                FROM geopolitical_relation_states
                WHERE mode=? AND as_of <= ? AND model_version=?
                """,
                [mode, cutoff, MODEL_VERSION],
            ).fetchone()[0]
            if latest_run is None:
                return []
            rows = conn.execute(
                """
                SELECT pair_key,source_iso3,target_iso3,as_of,known_at,mode,
                       vector_json,overall_score,confidence,coverage,status,trend,
                       live_delta,event_count,source_count,last_event_at,
                       structural_anchors_json,evidence_event_ids_json,model_version
                FROM geopolitical_relation_states
                WHERE mode=? AND as_of=? AND model_version=?
                ORDER BY pair_key
                """,
                [mode, latest_run, MODEL_VERSION],
            ).fetchall()

        output: list[RelationshipState] = []
        for row in rows:
            output.append(
                RelationshipState(
                    pair_key=str(row[0]),
                    source_iso3=str(row[1]),
                    target_iso3=str(row[2]),
                    as_of=row[3],
                    known_at=row[4],
                    mode=str(row[5]),
                    vector=json.loads(row[6]),
                    overall_score=float(row[7]),
                    confidence=float(row[8]),
                    coverage=float(row[9]),
                    status=str(row[10]),
                    trend=str(row[11]),
                    live_delta=(float(row[12]) if row[12] is not None else None),
                    event_count=int(row[13]),
                    source_count=int(row[14]),
                    last_event_at=row[15],
                    structural_anchors=tuple(json.loads(row[16] or "[]")),
                    evidence_event_ids=tuple(json.loads(row[17] or "[]")),
                    model_version=str(row[18]),
                )
            )
        return output

    def states_for_focal(
        self,
        focal_iso3: str,
        *,
        mode: str = "canonical",
        as_of: datetime | None = None,
    ) -> list[RelationshipState]:
        focal = str(focal_iso3).strip().upper()
        return [
            state
            for state in self.latest_states(mode=mode, as_of=as_of)
            if focal in {state.source_iso3, state.target_iso3}
        ]

    @staticmethod
    def _enrichment_id(
        enrichment: SemanticEventEnrichment,
        known_at: datetime,
    ) -> str:
        payload = "|".join(
            [
                enrichment.provider_event_id,
                enrichment.model,
                enrichment.schema_version,
                known_at.isoformat(),
            ]
        )
        return "gse_" + sha256(payload.encode("utf-8")).hexdigest()[:24]

    def record_semantic_enrichment(
        self,
        enrichment: SemanticEventEnrichment,
        *,
        known_at: datetime,
    ) -> bool:
        enrichment_id = self._enrichment_id(enrichment, known_at)
        with self.connect() as conn:
            if conn.execute(
                """
                SELECT 1 FROM geopolitical_semantic_enrichments
                WHERE enrichment_id=?
                """,
                [enrichment_id],
            ).fetchone():
                return False
            conn.execute(
                """
                INSERT INTO geopolitical_semantic_enrichments(
                    enrichment_id,provider_event_id,known_at,model,schema_version,
                    event_type,direction,severity,persistence,confidence,
                    dimension_modifiers_json,rationale
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    enrichment_id,
                    enrichment.provider_event_id,
                    known_at,
                    enrichment.model,
                    enrichment.schema_version,
                    enrichment.event_type,
                    enrichment.direction,
                    enrichment.severity,
                    enrichment.persistence,
                    enrichment.confidence,
                    json.dumps(
                        dict(enrichment.dimension_modifiers),
                        sort_keys=True,
                    ),
                    enrichment.rationale,
                ],
            )
        return True

    def latest_semantic_enrichments(
        self,
        *,
        as_of: datetime,
        model: str | None = None,
    ) -> dict[str, SemanticEventEnrichment]:
        clauses = ["known_at <= ?"]
        params: list[Any] = [as_of]
        if model:
            clauses.append("model=?")
            params.append(model)
        with self.connect() as conn:
            tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
            if "geopolitical_semantic_enrichments" not in tables:
                return {}
            cursor = conn.execute(
                f"""
                WITH ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY provider_event_id
                               ORDER BY known_at DESC,created_at DESC
                           ) AS rn
                    FROM geopolitical_semantic_enrichments
                    WHERE {' AND '.join(clauses)}
                )
                SELECT provider_event_id,model,event_type,direction,severity,
                       persistence,confidence,dimension_modifiers_json,rationale,
                       schema_version
                FROM ranked WHERE rn=1
                """,
                params,
            )
            rows = cursor.fetchall()
        return {
            str(row[0]): SemanticEventEnrichment(
                provider_event_id=str(row[0]),
                model=str(row[1]),
                event_type=str(row[2]),
                direction=str(row[3]),
                severity=float(row[4]),
                persistence=str(row[5]),
                confidence=float(row[6]),
                dimension_modifiers=json.loads(row[7] or "{}"),
                rationale=str(row[8]),
                schema_version=str(row[9]),
            )
            for row in rows
        }

    def impacts_for_pair(
        self,
        pair_key: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                SELECT provider_event_id,source_iso3,target_iso3,event_time,known_at,
                       event_code,event_root_code,semantic_tags_json,vector_json,
                       severity,confidence,half_life_days,persistence,source_count,
                       article_count,source_url,evidence_event_ids_json,cluster_size,
                       model_version
                FROM geopolitical_event_impacts
                WHERE pair_key=?
                ORDER BY event_time DESC,known_at DESC
                LIMIT ?
                """,
                [pair_key, int(limit)],
            )
            columns = [item[0] for item in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        for row in rows:
            row["semantic_tags"] = json.loads(row.pop("semantic_tags_json") or "[]")
            row["vector"] = json.loads(row.pop("vector_json") or "{}")
            row["evidence_event_ids"] = json.loads(
                row.pop("evidence_event_ids_json", None) or "[]"
            )
        return rows


__all__ = ["GeopoliticalStateStore"]
