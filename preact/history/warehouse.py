"""DuckDB historical warehouse for bitemporal evidence."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import duckdb

from .schema import TemporalRecord


class HistoricalWarehouse:
    """Persist immutable normalized evidence alongside raw source snapshots."""

    def __init__(self, path: str | Path = "data/history/preact_history.duckdb") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self):
        return duckdb.connect(self.path)

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS temporal_records (
                    record_id VARCHAR PRIMARY KEY,
                    entity_id VARCHAR NOT NULL,
                    variable VARCHAR NOT NULL,
                    value_json VARCHAR,
                    valid_from TIMESTAMPTZ NOT NULL,
                    valid_to TIMESTAMPTZ,
                    known_at TIMESTAMPTZ NOT NULL,
                    evidence_class VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    source_ref VARCHAR NOT NULL,
                    retrieved_at TIMESTAMPTZ NOT NULL,
                    dataset_version VARCHAR,
                    licence VARCHAR,
                    transform VARCHAR,
                    uncertainty DOUBLE,
                    attributes_json VARCHAR
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_temporal_entity_time "
                "ON temporal_records(entity_id, known_at, valid_from)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_temporal_variable_time "
                "ON temporal_records(variable, known_at, valid_from)"
            )

    @staticmethod
    def _encode(value) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    def insert_records(self, records: Iterable[TemporalRecord]) -> int:
        inserted = 0
        with self.connect() as conn:
            for record in records:
                before = conn.execute(
                    "SELECT COUNT(*) FROM temporal_records WHERE record_id = ?",
                    [record.record_id],
                ).fetchone()[0]
                if before:
                    continue
                conn.execute(
                    """
                    INSERT INTO temporal_records VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                    )
                    """,
                    [
                        record.record_id,
                        record.entity_id,
                        record.variable,
                        self._encode(record.value),
                        record.valid_from,
                        record.valid_to,
                        record.known_at,
                        record.evidence_class.value,
                        record.provenance.source,
                        record.provenance.source_ref,
                        record.provenance.retrieved_at,
                        record.provenance.dataset_version,
                        record.provenance.licence,
                        record.provenance.transform,
                        record.uncertainty,
                        self._encode(record.attributes),
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
        variable: str | None = None,
    ) -> list[dict]:
        world_time = valid_at or cutoff
        clauses = [
            "known_at <= ?",
            "valid_from <= ?",
            "(valid_to IS NULL OR ? < valid_to)",
        ]
        params: list[object] = [cutoff, world_time, world_time]
        if entity_id:
            clauses.append("entity_id = ?")
            params.append(entity_id)
        if variable:
            clauses.append("variable = ?")
            params.append(variable)

        with self.connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM temporal_records WHERE "
                + " AND ".join(clauses)
                + " ORDER BY entity_id, variable, valid_from, known_at",
                params,
            )
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
