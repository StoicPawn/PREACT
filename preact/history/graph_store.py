"""DuckDB store for bitemporal geopolitical relationships."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import duckdb

from .relations import HistoricalRelation
from .schema import KnowledgeMode


class HistoricalGraphStore:
    def __init__(self, path: str | Path = "data/history/preact_graph.duckdb") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self):
        return duckdb.connect(self.path)

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_relations (
                    relation_id VARCHAR PRIMARY KEY,
                    relation_type VARCHAR NOT NULL,
                    subject_entity_id VARCHAR NOT NULL,
                    object_entity_id VARCHAR NOT NULL,
                    valid_from TIMESTAMPTZ NOT NULL,
                    valid_to TIMESTAMPTZ,
                    known_at TIMESTAMPTZ NOT NULL,
                    directed BOOLEAN NOT NULL,
                    source VARCHAR NOT NULL,
                    source_ref VARCHAR NOT NULL,
                    retrieved_at TIMESTAMPTZ NOT NULL,
                    dataset_version VARCHAR,
                    attributes_json VARCHAR
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relation_subject_time "
                "ON historical_relations(subject_entity_id, known_at, valid_from)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relation_object_time "
                "ON historical_relations(object_entity_id, known_at, valid_from)"
            )

    def insert(self, relations: Iterable[HistoricalRelation]) -> int:
        inserted = 0
        with self.connect() as conn:
            for relation in relations:
                if conn.execute(
                    "SELECT 1 FROM historical_relations WHERE relation_id = ?",
                    [relation.relation_id],
                ).fetchone():
                    continue
                conn.execute(
                    "INSERT INTO historical_relations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        relation.relation_id,
                        relation.relation_type,
                        relation.subject_entity_id,
                        relation.object_entity_id,
                        relation.valid_from,
                        relation.valid_to,
                        relation.known_at,
                        relation.directed,
                        relation.source,
                        relation.source_ref,
                        relation.retrieved_at,
                        relation.dataset_version,
                        json.dumps(dict(relation.attributes), ensure_ascii=False, default=str),
                    ],
                )
                inserted += 1
        return inserted

    def as_of(
        self,
        *,
        cutoff: datetime,
        valid_at: datetime | None = None,
        entity_id: str | None = None,
        relation_type: str | None = None,
        knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
    ) -> list[dict]:
        world_time = valid_at or cutoff
        clauses = [
            "valid_from <= ?",
            "(valid_to IS NULL OR ? < valid_to)",
        ]
        params: list[object] = [world_time, world_time]
        if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN:
            clauses.insert(0, "known_at <= ?")
            params.insert(0, cutoff)
        if entity_id:
            clauses.append("(subject_entity_id = ? OR object_entity_id = ?)")
            params.extend([entity_id, entity_id])
        if relation_type:
            clauses.append("relation_type = ?")
            params.append(relation_type)
        with self.connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM historical_relations WHERE "
                + " AND ".join(clauses)
                + " ORDER BY relation_type, subject_entity_id, object_entity_id",
                params,
            )
            cols=[x[0] for x in cursor.description]
            return [dict(zip(cols,row)) for row in cursor.fetchall()]


    def list_entities(
        self,
        *,
        relation_type: str | None = None,
    ) -> list[str]:
        """List unique entities represented in the historical relation graph."""

        clauses = []
        params: list[object] = []
        if relation_type:
            clauses.append("relation_type = ?")
            params.append(relation_type)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT subject_entity_id AS entity_id FROM historical_relations"
                + where
                + " UNION SELECT object_entity_id AS entity_id FROM historical_relations"
                + where
                + " ORDER BY entity_id",
                [*params, *params],
            ).fetchall()
        return [str(row[0]) for row in rows if row[0]]
