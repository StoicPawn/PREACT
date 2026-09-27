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
                    actor1_code VARCHAR,
                    actor2_code VARCHAR,
                    actor1_type1_code VARCHAR,
                    actor1_type2_code VARCHAR,
                    actor1_type3_code VARCHAR,
                    actor2_type1_code VARCHAR,
                    actor2_type2_code VARCHAR,
                    actor2_type3_code VARCHAR,
                    is_root_event BOOLEAN,
                    action_location VARCHAR,
                    source_url VARCHAR,
                    snapshot_checksum VARCHAR,
                    evidence_class VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            for column_sql in (
                "actor1_code VARCHAR",
                "actor2_code VARCHAR",
                "actor1_type1_code VARCHAR",
                "actor1_type2_code VARCHAR",
                "actor1_type3_code VARCHAR",
                "actor2_type1_code VARCHAR",
                "actor2_type2_code VARCHAR",
                "actor2_type3_code VARCHAR",
                "is_root_event BOOLEAN",
            ):
                conn.execute(
                    "ALTER TABLE world_event_observations ADD COLUMN IF NOT EXISTS "
                    + column_sql
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
                CREATE TABLE IF NOT EXISTS world_event_mentions (
                    observation_id VARCHAR PRIMARY KEY,
                    provider_event_id VARCHAR NOT NULL,
                    known_at TIMESTAMPTZ NOT NULL,
                    mention_count INTEGER NOT NULL,
                    distinct_source_count INTEGER NOT NULL,
                    mention_sources_json VARCHAR NOT NULL,
                    mean_confidence DOUBLE,
                    max_confidence DOUBLE,
                    mean_document_tone DOUBLE,
                    latest_mention_time VARCHAR,
                    snapshot_checksum VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_world_mentions_event "
                "ON world_event_mentions(provider_event_id, known_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS world_gkg_documents (
                    observation_id VARCHAR PRIMARY KEY,
                    gkg_record_id VARCHAR NOT NULL,
                    known_at TIMESTAMPTZ NOT NULL,
                    source VARCHAR,
                    document_url VARCHAR NOT NULL,
                    country_iso3_json VARCHAR NOT NULL,
                    provider_country_codes_json VARCHAR NOT NULL,
                    themes_json VARCHAR NOT NULL,
                    persons_json VARCHAR NOT NULL,
                    organizations_json VARCHAR NOT NULL,
                    overall_tone DOUBLE,
                    all_names_json VARCHAR NOT NULL,
                    snapshot_checksum VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
            )
            conn.execute(
                "ALTER TABLE world_gkg_documents ADD COLUMN IF NOT EXISTS "
                "country_mapping_checksum VARCHAR"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_world_gkg_known "
                "ON world_gkg_documents(known_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS world_gdelt_snapshot_projection (
                    snapshot_checksum VARCHAR PRIMARY KEY,
                    operation VARCHAR NOT NULL,
                    retrieved_at TIMESTAMPTZ NOT NULL,
                    processed_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )
                """
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

    def seed_reference_facts(
        self,
        candidates: Sequence[KnowledgeUpdateCandidate],
        *,
        allowed_sources: Sequence[str] = ("wikidata",),
    ) -> dict[str, tuple[str, bool]]:
        """Batch-seed unknown fields from reviewed structured references.

        One DuckDB connection/transaction is used for the whole batch. Existing
        current assertions are never overwritten.
        """
        allowed = {str(item).strip() for item in allowed_sources}
        results: dict[str, tuple[str, bool]] = {}

        with self.connect() as conn:
            for candidate in candidates:
                if candidate.kind is not KnowledgeUpdateKind.FACT:
                    raise ValueError("Reference seeds must be factual candidates")
                if not candidate.evidence:
                    raise ValueError("Reference seeds require evidence")
                observed_sources = {item.source for item in candidate.evidence}
                if not observed_sources.issubset(allowed):
                    raise ValueError("Reference seed contains a non-approved source")

                existing = conn.execute(
                    """
                    SELECT assertion_id
                    FROM world_knowledge_assertions
                    WHERE entity_id=? AND field=? AND valid_to IS NULL
                    ORDER BY valid_from DESC LIMIT 1
                    """,
                    [candidate.entity_id, candidate.field],
                ).fetchone()
                if existing:
                    results[str(candidate.candidate_id)] = (str(existing[0]), False)
                    continue

                claim_key = candidate.claim_key()
                evidence_payload = self._evidence_payload(candidate)
                value_json = json.dumps(
                    candidate.value,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                evidence_json = json.dumps(
                    evidence_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                base_attributes = dict(candidate.attributes)
                candidate_attributes_json = json.dumps(
                    base_attributes,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )

                conn.execute(
                    """
                    INSERT OR IGNORE INTO world_knowledge_candidates(
                        candidate_id,claim_key,entity_id,field,value_json,valid_from,
                        detected_at,domain,kind,confidence,evidence_json,attributes_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        candidate.candidate_id,
                        claim_key,
                        candidate.entity_id,
                        candidate.field,
                        value_json,
                        candidate.valid_from,
                        candidate.detected_at,
                        candidate.domain.value,
                        candidate.kind.value,
                        float(candidate.confidence),
                        evidence_json,
                        candidate_attributes_json,
                    ],
                )

                for item in candidate.evidence:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO world_knowledge_evidence(
                            evidence_id,claim_key,candidate_id,source,source_ref,
                            published_at,retrieved_at,independent_group,authoritative,
                            excerpt_hash,observed_confidence
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

                assertion_id = "wka_seed_" + claim_key.rsplit("_", 1)[-1]
                assertion_attributes = dict(base_attributes)
                assertion_attributes["promotion_mode"] = "reference_seed"
                assertion_attributes_json = json.dumps(
                    assertion_attributes,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO world_knowledge_assertions(
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
                        assertion_attributes_json,
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
                            f"Reference baseline seeded: {candidate.field}",
                            "queued",
                        ],
                    )
                results[str(candidate.candidate_id)] = (assertion_id, True)

        return results

    def seed_reference_fact(
        self,
        candidate: KnowledgeUpdateCandidate,
        *,
        allowed_sources: Sequence[str] = ("wikidata",),
    ) -> tuple[str, bool]:
        """Backward-compatible single-reference seed wrapper."""
        return self.seed_reference_facts(
            [candidate],
            allowed_sources=allowed_sources,
        )[str(candidate.candidate_id)]

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
                    conn.execute(
                        """
                        UPDATE world_event_observations
                        SET actor1_code=COALESCE(actor1_code,?),
                            actor2_code=COALESCE(actor2_code,?),
                            actor1_type1_code=COALESCE(actor1_type1_code,?),
                            actor1_type2_code=COALESCE(actor1_type2_code,?),
                            actor1_type3_code=COALESCE(actor1_type3_code,?),
                            actor2_type1_code=COALESCE(actor2_type1_code,?),
                            actor2_type2_code=COALESCE(actor2_type2_code,?),
                            actor2_type3_code=COALESCE(actor2_type3_code,?),
                            is_root_event=COALESCE(is_root_event,?)
                        WHERE observation_id=?
                        """,
                        [
                            event.actor1_code,
                            event.actor2_code,
                            event.actor1_type1_code,
                            event.actor1_type2_code,
                            event.actor1_type3_code,
                            event.actor2_type1_code,
                            event.actor2_type2_code,
                            event.actor2_type3_code,
                            event.is_root_event,
                            event.observation_id,
                        ],
                    )
                    continue
                conn.execute(
                    """
                    INSERT INTO world_event_observations(
                        observation_id,provider,provider_event_id,event_time,known_at,
                        actor1_entity_id,actor2_entity_id,event_code,event_base_code,
                        event_root_code,quad_class,goldstein,tone,num_mentions,
                        num_sources,num_articles,actor1_name,actor2_name,
                        actor1_code,actor2_code,
                        actor1_type1_code,actor1_type2_code,actor1_type3_code,
                        actor2_type1_code,actor2_type2_code,actor2_type3_code,
                        is_root_event,action_location,source_url,snapshot_checksum,
                        evidence_class
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                        event.actor1_code,
                        event.actor2_code,
                        event.actor1_type1_code,
                        event.actor1_type2_code,
                        event.actor1_type3_code,
                        event.actor2_type1_code,
                        event.actor2_type2_code,
                        event.actor2_type3_code,
                        event.is_root_event,
                        event.action_location,
                        event.source_url,
                        event.snapshot_checksum,
                        event.evidence_class,
                    ],
                )
                inserted += 1
        return inserted

    def processed_gdelt_context_snapshots(self) -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT snapshot_checksum FROM world_gdelt_snapshot_projection"
            ).fetchall()
        return {str(row[0]) for row in rows}

    def record_gdelt_context(
        self,
        *,
        mention_observations: Iterable[Mapping[str, Any]],
        gkg_documents: Iterable[Mapping[str, Any]],
        processed_snapshots: Iterable[Mapping[str, Any]],
    ) -> dict[str, int]:
        """Persist PREACT evidence derived from externally acquired GDELT snapshots."""

        import hashlib

        inserted_mentions = 0
        inserted_gkg = 0
        marked_snapshots = 0

        with self.connect() as conn:
            for item in mention_observations:
                event_id = str(item.get("provider_event_id") or "").strip()
                known_at = item.get("known_at")
                checksum = str(item.get("snapshot_checksum") or "").strip()
                if not event_id or known_at is None or not checksum:
                    continue
                material = f"{event_id}|{known_at}|{checksum}"
                observation_id = (
                    "wem_"
                    + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
                )
                if conn.execute(
                    "SELECT 1 FROM world_event_mentions WHERE observation_id=?",
                    [observation_id],
                ).fetchone():
                    continue
                conn.execute(
                    """
                    INSERT INTO world_event_mentions(
                        observation_id,provider_event_id,known_at,mention_count,
                        distinct_source_count,mention_sources_json,mean_confidence,
                        max_confidence,mean_document_tone,latest_mention_time,
                        snapshot_checksum
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        observation_id,
                        event_id,
                        known_at,
                        int(item.get("mention_count") or 0),
                        int(item.get("distinct_source_count") or 0),
                        json.dumps(
                            list(item.get("mention_sources") or []),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        item.get("mean_confidence"),
                        item.get("max_confidence"),
                        item.get("mean_document_tone"),
                        item.get("latest_mention_time"),
                        checksum,
                    ],
                )
                inserted_mentions += 1

            for item in gkg_documents:
                record_id = str(item.get("gkg_record_id") or "").strip()
                known_at = item.get("known_at")
                checksum = str(item.get("snapshot_checksum") or "").strip()
                url = str(item.get("document_url") or "").strip()
                if not record_id or known_at is None or not checksum or not url:
                    continue
                material = f"{record_id}|{known_at}|{checksum}"
                observation_id = (
                    "wgk_"
                    + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
                )
                country_json = json.dumps(
                    list(item.get("country_iso3") or []),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                provider_country_json = json.dumps(
                    list(item.get("country_codes") or []),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                mapping_checksum = str(
                    item.get("country_mapping_checksum") or ""
                ).strip() or None
                existing = conn.execute(
                    """
                    SELECT country_iso3_json,country_mapping_checksum
                    FROM world_gkg_documents WHERE observation_id=?
                    """,
                    [observation_id],
                ).fetchone()
                if existing:
                    if (
                        country_json != (existing[0] or "[]")
                        or mapping_checksum != existing[1]
                    ):
                        conn.execute(
                            """
                            UPDATE world_gkg_documents
                            SET country_iso3_json=?,
                                provider_country_codes_json=?,
                                country_mapping_checksum=?
                            WHERE observation_id=?
                            """,
                            [
                                country_json,
                                provider_country_json,
                                mapping_checksum,
                                observation_id,
                            ],
                        )
                    continue
                conn.execute(
                    """
                    INSERT INTO world_gkg_documents(
                        observation_id,gkg_record_id,known_at,source,document_url,
                        country_iso3_json,provider_country_codes_json,themes_json,
                        persons_json,organizations_json,overall_tone,all_names_json,
                        snapshot_checksum,country_mapping_checksum
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        observation_id,
                        record_id,
                        known_at,
                        str(item.get("source") or "").strip() or None,
                        url,
                        country_json,
                        provider_country_json,
                        json.dumps(
                            list(item.get("themes") or []),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        json.dumps(
                            list(item.get("persons") or []),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        json.dumps(
                            list(item.get("organizations") or []),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        item.get("overall_tone"),
                        json.dumps(
                            list(item.get("all_names") or []),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        checksum,
                        mapping_checksum,
                    ],
                )
                inserted_gkg += 1

            for snapshot in processed_snapshots:
                checksum = str(
                    snapshot.get("snapshot_checksum") or ""
                ).strip()
                operation = str(snapshot.get("operation") or "").strip()
                retrieved_at = snapshot.get("retrieved_at")
                if not checksum or not operation or retrieved_at is None:
                    continue
                if conn.execute(
                    """
                    SELECT 1 FROM world_gdelt_snapshot_projection
                    WHERE snapshot_checksum=?
                    """,
                    [checksum],
                ).fetchone():
                    continue
                conn.execute(
                    """
                    INSERT INTO world_gdelt_snapshot_projection(
                        snapshot_checksum,operation,retrieved_at
                    ) VALUES (?,?,?)
                    """,
                    [checksum, operation, retrieved_at],
                )
                marked_snapshots += 1

        return {
            "inserted_mentions": inserted_mentions,
            "inserted_gkg_documents": inserted_gkg,
            "marked_snapshots": marked_snapshots,
        }

    def gkg_context_for_country(
        self,
        entity_id: str,
        *,
        as_of: datetime,
        known_cutoff: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return recent GKG context linked to a country at a knowledge cutoff."""

        if limit < 1:
            raise ValueError("limit must be >= 1")
        prefix = "country:"
        if not entity_id.startswith(prefix):
            raise ValueError("entity_id must use country:ISO3 format")
        iso3 = entity_id[len(prefix):].strip().upper()
        cutoff = known_cutoff or as_of
        needle = f'%"{iso3}"%'

        with self.connect() as conn:
            cursor = conn.execute(
                """
                SELECT observation_id,gkg_record_id,known_at,source,document_url,
                       country_iso3_json,provider_country_codes_json,themes_json,
                       persons_json,organizations_json,overall_tone,all_names_json,
                       snapshot_checksum
                FROM world_gkg_documents
                WHERE known_at <= ?
                  AND known_at <= ?
                  AND country_iso3_json LIKE ?
                ORDER BY known_at DESC
                LIMIT ?
                """,
                [as_of, cutoff, needle, int(limit)],
            )
            columns = [item[0] for item in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

        for row in rows:
            for key in (
                "country_iso3_json",
                "provider_country_codes_json",
                "themes_json",
                "persons_json",
                "organizations_json",
                "all_names_json",
            ):
                decoded_key = key.removesuffix("_json")
                row[decoded_key] = json.loads(row.pop(key) or "[]")
        return rows

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
                ),
                mention_ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY provider_event_id
                               ORDER BY known_at DESC, created_at DESC
                           ) AS mrn
                    FROM world_event_mentions
                    WHERE known_at <= ?
                )
                SELECT r.observation_id,r.provider,r.provider_event_id,
                       r.event_time,r.known_at,r.actor1_entity_id,r.actor2_entity_id,
                       r.event_code,r.event_base_code,r.event_root_code,r.quad_class,
                       r.goldstein,r.tone,r.num_mentions,r.num_sources,r.num_articles,
                       r.actor1_name,r.actor2_name,r.action_location,r.source_url,
                       r.snapshot_checksum,r.evidence_class,
                       m.mention_count AS corroborating_mentions,
                       m.distinct_source_count AS mention_source_count,
                       m.mean_confidence AS mention_mean_confidence,
                       m.max_confidence AS mention_max_confidence,
                       m.mean_document_tone AS mention_document_tone,
                       m.mention_sources_json
                FROM ranked r
                LEFT JOIN mention_ranked m
                  ON r.provider_event_id=m.provider_event_id
                 AND m.mrn=1
                WHERE r.rn=1
                ORDER BY r.event_time DESC, r.known_at DESC
                LIMIT ?
                """,
                [as_of, cutoff, entity_id, entity_id, cutoff, int(limit)],
            )
            columns = [item[0] for item in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

        for row in rows:
            raw_sources = row.pop("mention_sources_json", None)
            row["mention_sources"] = (
                json.loads(raw_sources) if raw_sources else []
            )
        return rows

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
            mention_observations = int(
                conn.execute("SELECT COUNT(*) FROM world_event_mentions").fetchone()[0]
            )
            gkg_documents = int(
                conn.execute("SELECT COUNT(*) FROM world_gkg_documents").fetchone()[0]
            )
            gdelt_context_snapshots = int(
                conn.execute(
                    "SELECT COUNT(*) FROM world_gdelt_snapshot_projection"
                ).fetchone()[0]
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
            "mention_observations": mention_observations,
            "gkg_documents": gkg_documents,
            "gdelt_context_snapshots": gdelt_context_snapshots,
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
