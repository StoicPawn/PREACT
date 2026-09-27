"""Append-oriented DuckDB store for autonomous World Knowledge maintenance."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import duckdb

from preact.intelligence.world_events import WorldEventObservation
from preact.intelligence.world_knowledge import (
    KnowledgeDomain,
    KnowledgeUpdateCandidate,
    KnowledgeUpdateKind,
    PromotionAction,
    PromotionDecision,
    SourceEvidence,
    affected_narrative_sections,
)


class WorldKnowledgeStore:
    def __init__(self, path: str | Path = "data/history/world_knowledge.duckdb") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self):
        return duckdb.connect(self.path)

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS world_knowledge_candidates (
                    candidate_id VARCHAR PRIMARY KEY,
                    claim_key VARCHAR,
                    entity_id VARCHAR NOT NULL,
                    field VARCHAR NOT NULL,
                    value_json VARCHAR NOT NULL,
                    valid_from TIMESTAMPTZ NOT NULL,
                    detected_at TIMESTAMPTZ NOT NULL,
                    domain VARCHAR NOT NULL,
                    kind VARCHAR NOT NULL,
                    confidence DOUBLE NOT NULL,
                    evidence_json VARCHAR NOT NULL,
                    attributes_json VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            conn.execute(
                "ALTER TABLE world_knowledge_candidates ADD COLUMN IF NOT EXISTS claim_key VARCHAR"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS world_knowledge_evidence (
                    evidence_id VARCHAR PRIMARY KEY,
                    claim_key VARCHAR NOT NULL,
                    candidate_id VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    source_ref VARCHAR NOT NULL,
                    published_at TIMESTAMPTZ NOT NULL,
                    retrieved_at TIMESTAMPTZ NOT NULL,
                    independent_group VARCHAR NOT NULL,
                    authoritative BOOLEAN NOT NULL,
                    excerpt_hash VARCHAR,
                    observed_confidence DOUBLE NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_world_evidence_claim "
                "ON world_knowledge_evidence(claim_key, retrieved_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS world_knowledge_decisions (
                    candidate_id VARCHAR PRIMARY KEY,
                    action VARCHAR NOT NULL,
                    reason VARCHAR NOT NULL,
                    independent_source_groups INTEGER NOT NULL,
                    authoritative_sources INTEGER NOT NULL,
                    confidence DOUBLE NOT NULL,
                    decided_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS world_knowledge_assertions (
                    assertion_id VARCHAR PRIMARY KEY,
                    candidate_id VARCHAR NOT NULL,
                    entity_id VARCHAR NOT NULL,
                    field VARCHAR NOT NULL,
                    value_json VARCHAR NOT NULL,
                    valid_from TIMESTAMPTZ NOT NULL,
                    valid_to TIMESTAMPTZ,
                    known_at TIMESTAMPTZ NOT NULL,
                    domain VARCHAR NOT NULL,
                    confidence DOUBLE NOT NULL,
                    evidence_json VARCHAR NOT NULL,
                    attributes_json VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS world_event_observations (
                    observation_id VARCHAR PRIMARY KEY,
                    provider VARCHAR NOT NULL,
                    provider_event_id VARCHAR NOT NULL,
                    event_time TIMESTAMPTZ NOT NULL,
                    known_at TIMESTAMPTZ NOT NULL,
                    actor1_entity_id VARCHAR NOT NULL,
                    actor2_entity_id VARCHAR,
                    event_code VARCHAR,
                    event_base_code VARCHAR,
                    event_root_code VARCHAR,
                    quad_class INTEGER,
                    goldstein DOUBLE,
                    tone DOUBLE,
                    num_mentions DOUBLE,
                    num_sources DOUBLE,
                    num_articles DOUBLE,
                    actor1_name VARCHAR,
                    actor2_name VARCHAR,
                    action_location VARCHAR,
                    source_url VARCHAR,
                    snapshot_checksum VARCHAR,
                    evidence_class VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_world_event_actor1 "
                "ON world_event_observations(actor1_entity_id, event_time, known_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_world_event_actor2 "
                "ON world_event_observations(actor2_entity_id, event_time, known_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_world_event_provider "
                "ON world_event_observations(provider, provider_event_id, known_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS world_narrative_jobs (
                    job_id VARCHAR PRIMARY KEY,
                    candidate_id VARCHAR NOT NULL,
                    entity_id VARCHAR NOT NULL,
                    section VARCHAR NOT NULL,
                    reason VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                    completed_at TIMESTAMPTZ
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_world_assertion_state "
                "ON world_knowledge_assertions(entity_id, field, valid_from, valid_to)"
            )

    @staticmethod
    def _evidence_payload(candidate: KnowledgeUpdateCandidate) -> list[dict[str, Any]]:
        return [
            {
                "source": item.source,
                "source_ref": item.source_ref,
                "published_at": item.published_at.isoformat(),
                "retrieved_at": item.retrieved_at.isoformat(),
                "independent_group": item.independent_group,
                "authoritative": item.authoritative,
                "excerpt_hash": item.excerpt_hash,
            }
            for item in candidate.evidence
        ]

    @staticmethod
    def _evidence_id(claim_key: str, evidence: SourceEvidence) -> str:
        material = "|".join(
            [
                claim_key,
                evidence.source.strip(),
                evidence.source_ref.strip(),
                evidence.published_at.isoformat(),
                evidence.independent_group.strip(),
            ]
        )
        import hashlib
        return "wke_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    def record_candidate(self, candidate: KnowledgeUpdateCandidate) -> bool:
        """Persist one observed candidate and its immutable evidence rows."""
        claim_key = candidate.claim_key()
        inserted = False
        with self.connect() as conn:
            if not conn.execute(
                "SELECT 1 FROM world_knowledge_candidates WHERE candidate_id = ?",
                [candidate.candidate_id],
            ).fetchone():
                conn.execute(
                    """
                    INSERT INTO world_knowledge_candidates(
                        candidate_id,claim_key,entity_id,field,value_json,valid_from,detected_at,
                        domain,kind,confidence,evidence_json,attributes_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        candidate.candidate_id,
                        claim_key,
                        candidate.entity_id,
                        candidate.field,
                        json.dumps(candidate.value, ensure_ascii=False, sort_keys=True, default=str),
                        candidate.valid_from,
                        candidate.detected_at,
                        candidate.domain.value,
                        candidate.kind.value,
                        float(candidate.confidence),
                        json.dumps(self._evidence_payload(candidate), ensure_ascii=False, sort_keys=True),
                        json.dumps(dict(candidate.attributes), ensure_ascii=False, sort_keys=True, default=str),
                    ],
                )
                inserted = True

            # Backfill claim_key for databases created by the first World Knowledge version.
            conn.execute(
                "UPDATE world_knowledge_candidates SET claim_key=? "
                "WHERE candidate_id=? AND (claim_key IS NULL OR claim_key='')",
                [claim_key, candidate.candidate_id],
            )

            for item in candidate.evidence:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO world_knowledge_evidence(
                        evidence_id,claim_key,candidate_id,source,source_ref,published_at,
                        retrieved_at,independent_group,authoritative,excerpt_hash,
                        observed_confidence
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        self._evidence_id(claim_key, item),
                        claim_key,
                        candidate.candidate_id,
                        item.source,
                        item.source_ref,
                        item.published_at,
                        item.retrieved_at,
                        item.independent_group,
                        bool(item.authoritative),
                        item.excerpt_hash,
                        float(candidate.confidence),
                    ],
                )
        return inserted

    def aggregate_claim(self, claim_key: str) -> KnowledgeUpdateCandidate:
        """Materialize the corroborated view of one claim across ingestion cycles."""
        with self.connect() as conn:
            base = conn.execute(
                """
                SELECT entity_id,field,value_json,valid_from,domain,kind,attributes_json,
                       confidence
                FROM world_knowledge_candidates
                WHERE claim_key=?
                ORDER BY detected_at DESC, created_at DESC
                LIMIT 1
                """,
                [claim_key],
            ).fetchone()
            if base is None:
                raise KeyError(f"Unknown world knowledge claim: {claim_key}")
            max_confidence = float(
                conn.execute(
                    "SELECT MAX(confidence) FROM world_knowledge_candidates WHERE claim_key=?",
                    [claim_key],
                ).fetchone()[0]
            )

            rows = conn.execute(
                """
                SELECT source,source_ref,published_at,retrieved_at,independent_group,
                       authoritative,excerpt_hash
                FROM world_knowledge_evidence
                WHERE claim_key=?
                ORDER BY retrieved_at,source_ref
                """,
                [claim_key],
            ).fetchall()

        evidence = tuple(
            SourceEvidence(
                source=str(row[0]),
                source_ref=str(row[1]),
                published_at=row[2],
                retrieved_at=row[3],
                independent_group=str(row[4]),
                authoritative=bool(row[5]),
                excerpt_hash=row[6],
            )
            for row in rows
        )
        detected_at = max((item.retrieved_at for item in evidence), default=base[3])
        return KnowledgeUpdateCandidate(
            entity_id=str(base[0]),
            field=str(base[1]),
            value=json.loads(base[2]),
            valid_from=base[3],
            detected_at=detected_at,
            domain=KnowledgeDomain(str(base[4])),
            kind=KnowledgeUpdateKind(str(base[5])),
            confidence=max_confidence,
            evidence=evidence,
            attributes=json.loads(base[6]) if base[6] else {},
            candidate_id=claim_key,
        )

    def record_decision(self, decision: PromotionDecision) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO world_knowledge_decisions(
                    candidate_id,action,reason,independent_source_groups,
                    authoritative_sources,confidence,decided_at
                ) VALUES (?,?,?,?,?,?,current_timestamp)
                """,
                [
                    decision.candidate_id,
                    decision.action.value,
                    decision.reason,
                    decision.independent_source_groups,
                    decision.authoritative_sources,
                    decision.confidence,
                ],
            )

    def promote_fact(
        self,
        candidate: KnowledgeUpdateCandidate,
        decision: PromotionDecision,
    ) -> str:
        if decision.action is not PromotionAction.AUTO_PROMOTE_FACT:
            raise ValueError("Only auto-promoted factual candidates may mutate world state")

        assertion_id = "wka_" + str(candidate.candidate_id).removeprefix("wkc_")
        value_json = json.dumps(candidate.value, ensure_ascii=False, sort_keys=True, default=str)
        evidence_json = json.dumps(self._evidence_payload(candidate), ensure_ascii=False, sort_keys=True)
        attributes_json = json.dumps(dict(candidate.attributes), ensure_ascii=False, sort_keys=True, default=str)

        with self.connect() as conn:
            existing = conn.execute(
                """
                SELECT assertion_id,value_json,valid_from
                FROM world_knowledge_assertions
                WHERE entity_id=? AND field=? AND valid_to IS NULL
                ORDER BY valid_from DESC LIMIT 1
                """,
                [candidate.entity_id, candidate.field],
            ).fetchone()

            if existing and existing[1] == value_json:
                return str(existing[0])

            if existing:
                if candidate.valid_from < existing[2]:
                    raise ValueError("Cannot supersede a current assertion with an earlier valid_from")
                conn.execute(
                    "UPDATE world_knowledge_assertions SET valid_to=? WHERE assertion_id=?",
                    [candidate.valid_from, existing[0]],
                )

            if not conn.execute(
                "SELECT 1 FROM world_knowledge_assertions WHERE assertion_id=?",
                [assertion_id],
            ).fetchone():
                conn.execute(
                    """
                    INSERT INTO world_knowledge_assertions(
                        assertion_id,candidate_id,entity_id,field,value_json,
                        valid_from,valid_to,known_at,domain,confidence,
                        evidence_json,attributes_json
                    ) VALUES (?,?,?,?,?,?,NULL,?,?,?,?,?)
                    """,
                    [
                        assertion_id,
                        candidate.candidate_id,
                        candidate.entity_id,
                        candidate.field,
                        value_json,
                        candidate.valid_from,
                        candidate.detected_at,
                        candidate.domain.value,
                        float(candidate.confidence),
                        evidence_json,
                        attributes_json,
                    ],
                )

            for section in affected_narrative_sections(candidate):
                job_id = f"wnj_{candidate.candidate_id}_{section}"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO world_narrative_jobs(
                        job_id,candidate_id,entity_id,section,reason,status
                    ) VALUES (?,?,?,?,?,?)
                    """,
                    [
                        job_id,
                        candidate.candidate_id,
                        candidate.entity_id,
                        section,
                        f"Promoted material fact: {candidate.field}",
                        "queued",
                    ],
                )

        return assertion_id

    def record_world_events(self, events: Iterable[WorldEventObservation]) -> int:
        """Persist immutable provider-event observations.

        Repeated snapshots of the same provider event are preserved as separate
        observations when their knowledge time or snapshot checksum differs.
        """
        inserted = 0
        with self.connect() as conn:
            for event in events:
                if conn.execute(
                    "SELECT 1 FROM world_event_observations WHERE observation_id=?",
                    [event.observation_id],
                ).fetchone():
                    continue
                conn.execute(
                    """
                    INSERT INTO world_event_observations(
                        observation_id,provider,provider_event_id,event_time,known_at,
                        actor1_entity_id,actor2_entity_id,event_code,event_base_code,
                        event_root_code,quad_class,goldstein,tone,num_mentions,
                        num_sources,num_articles,actor1_name,actor2_name,
                        action_location,source_url,snapshot_checksum,evidence_class
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        event.observation_id,
                        event.provider,
                        event.provider_event_id,
                        event.event_time,
                        event.known_at,
                        event.actor1_entity_id,
                        event.actor2_entity_id,
                        event.event_code,
                        event.event_base_code,
                        event.event_root_code,
                        event.quad_class,
                        event.goldstein,
                        event.tone,
                        event.num_mentions,
                        event.num_sources,
                        event.num_articles,
                        event.actor1_name,
                        event.actor2_name,
                        event.action_location,
                        event.source_url,
                        event.snapshot_checksum,
                        event.evidence_class,
                    ],
                )
                inserted += 1
        return inserted

    def event_timeline(
        self,
        entity_id: str,
        *,
        as_of: datetime,
        known_cutoff: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return one latest-known observation per provider event at a cutoff."""
        if limit < 1:
            raise ValueError("limit must be >= 1")
        cutoff = known_cutoff or as_of

        with self.connect() as conn:
            cursor = conn.execute(
                """
                WITH ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY provider, provider_event_id
                               ORDER BY known_at DESC, created_at DESC
                           ) AS rn
                    FROM world_event_observations
                    WHERE event_time <= ?
                      AND known_at <= ?
                      AND (actor1_entity_id = ? OR actor2_entity_id = ?)
                )
                SELECT observation_id,provider,provider_event_id,event_time,known_at,
                       actor1_entity_id,actor2_entity_id,event_code,event_base_code,
                       event_root_code,quad_class,goldstein,tone,num_mentions,
                       num_sources,num_articles,actor1_name,actor2_name,
                       action_location,source_url,snapshot_checksum,evidence_class
                FROM ranked
                WHERE rn=1
                ORDER BY event_time DESC, known_at DESC
                LIMIT ?
                """,
                [as_of, cutoff, entity_id, entity_id, int(limit)],
            )
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def current_state(self, entity_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT field,value_json,valid_from,known_at,domain,confidence,assertion_id,
                       evidence_json,attributes_json
                FROM world_knowledge_assertions
                WHERE entity_id=? AND valid_to IS NULL
                ORDER BY field
                """,
                [entity_id],
            ).fetchall()

        result: dict[str, Any] = {}
        for (
            field,
            value_json,
            valid_from,
            known_at,
            domain,
            confidence,
            assertion_id,
            evidence_json,
            attributes_json,
        ) in rows:
            result[str(field)] = {
                "value": json.loads(value_json),
                "valid_from": valid_from,
                "known_at": known_at,
                "domain": domain,
                "confidence": confidence,
                "assertion_id": assertion_id,
                "evidence": json.loads(evidence_json) if evidence_json else [],
                "attributes": json.loads(attributes_json) if attributes_json else {},
            }
        return result

    def state_as_of(self, entity_id: str, when: datetime, known_cutoff: Optional[datetime] = None) -> dict[str, Any]:
        cutoff = known_cutoff or when
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT field,value_json,valid_from,valid_to,known_at,domain,confidence,assertion_id,
                       evidence_json,attributes_json
                FROM world_knowledge_assertions
                WHERE entity_id=?
                  AND valid_from <= ?
                  AND (valid_to IS NULL OR ? < valid_to)
                  AND known_at <= ?
                ORDER BY field
                """,
                [entity_id, when, when, cutoff],
            ).fetchall()

        return {
            str(field): {
                "value": json.loads(value_json),
                "valid_from": valid_from,
                "valid_to": valid_to,
                "known_at": known_at,
                "domain": domain,
                "confidence": confidence,
                "assertion_id": assertion_id,
                "evidence": json.loads(evidence_json) if evidence_json else [],
                "attributes": json.loads(attributes_json) if attributes_json else {},
            }
            for (
                field,
                value_json,
                valid_from,
                valid_to,
                known_at,
                domain,
                confidence,
                assertion_id,
                evidence_json,
                attributes_json,
            ) in rows
        }

    def status(self) -> dict[str, int]:
        with self.connect() as conn:
            candidates = int(conn.execute("SELECT COUNT(*) FROM world_knowledge_candidates").fetchone()[0])
            decisions = int(conn.execute("SELECT COUNT(*) FROM world_knowledge_decisions").fetchone()[0])
            assertions = int(conn.execute("SELECT COUNT(*) FROM world_knowledge_assertions").fetchone()[0])
            current_assertions = int(
                conn.execute(
                    "SELECT COUNT(*) FROM world_knowledge_assertions WHERE valid_to IS NULL"
                ).fetchone()[0]
            )
            evidence_observations = int(
                conn.execute("SELECT COUNT(*) FROM world_knowledge_evidence").fetchone()[0]
            )
            event_observations = int(
                conn.execute("SELECT COUNT(*) FROM world_event_observations").fetchone()[0]
            )
            distinct_claims = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT claim_key) FROM world_knowledge_candidates "
                    "WHERE claim_key IS NOT NULL"
                ).fetchone()[0]
            )
            narrative_jobs = int(
                conn.execute(
                    "SELECT COUNT(*) FROM world_narrative_jobs WHERE status='queued'"
                ).fetchone()[0]
            )
        return {
            "candidates": candidates,
            "distinct_claims": distinct_claims,
            "evidence_observations": evidence_observations,
            "event_observations": event_observations,
            "decisions": decisions,
            "assertions": assertions,
            "current_assertions": current_assertions,
            "queued_narrative_jobs": narrative_jobs,
        }

    def queued_narrative_jobs(self, entity_id: Optional[str] = None) -> list[dict[str, Any]]:
        clauses = ["status='queued'"]
        params: list[Any] = []
        if entity_id:
            clauses.append("entity_id=?")
            params.append(entity_id)
        with self.connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM world_narrative_jobs WHERE " + " AND ".join(clauses) + " ORDER BY created_at",
                params,
            )
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


__all__ = ["WorldKnowledgeStore"]
