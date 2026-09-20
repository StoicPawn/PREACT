"""DuckDB store for provenance-aware news and historical documents."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import duckdb

from .documents import HistoricalDocument


class HistoricalDocumentStore:
    def __init__(self, path: str | Path = "data/history/preact_documents.duckdb") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self):
        return duckdb.connect(self.path)

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_documents (
                    document_id VARCHAR PRIMARY KEY,
                    source_id VARCHAR NOT NULL,
                    source_ref VARCHAR NOT NULL,
                    title VARCHAR NOT NULL,
                    published_at TIMESTAMPTZ NOT NULL,
                    known_at TIMESTAMPTZ NOT NULL,
                    acquired_at TIMESTAMPTZ NOT NULL,
                    url VARCHAR,
                    language VARCHAR,
                    entity_ids_json VARCHAR,
                    place_ids_json VARCHAR,
                    text VARCHAR,
                    text_availability VARCHAR NOT NULL,
                    ocr_quality DOUBLE,
                    snapshot_checksum VARCHAR,
                    attributes_json VARCHAR
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_time "
                "ON historical_documents(known_at, published_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_source "
                "ON historical_documents(source_id, published_at)"
            )

    @staticmethod
    def _json(value) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    def insert(self, documents: Iterable[HistoricalDocument]) -> int:
        inserted = 0
        with self.connect() as conn:
            for document in documents:
                exists = conn.execute(
                    "SELECT 1 FROM historical_documents WHERE document_id = ?",
                    [document.document_id],
                ).fetchone()
                if exists:
                    continue
                conn.execute(
                    """
                    INSERT INTO historical_documents VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                    )
                    """,
                    [
                        document.document_id,
                        document.source_id,
                        document.source_ref,
                        document.title,
                        document.published_at,
                        document.known_at,
                        document.acquired_at,
                        document.url,
                        document.language,
                        self._json(document.entity_ids),
                        self._json(document.place_ids),
                        document.text,
                        document.text_availability.value,
                        document.ocr_quality,
                        document.snapshot_checksum,
                        self._json(document.attributes),
                    ],
                )
                inserted += 1
        return inserted

    def as_of(
        self,
        *,
        cutoff: datetime,
        entity_id: str | None = None,
        source_id: str | None = None,
        query: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        clauses = ["known_at <= ?"]
        params: list[object] = [cutoff]
        if entity_id:
            clauses.append("entity_ids_json LIKE ?")
            params.append(f'%"{entity_id}"%')
        if source_id:
            clauses.append("source_id = ?")
            params.append(source_id)
        if query:
            clauses.append("(lower(title) LIKE ? OR lower(coalesce(text, '')) LIKE ?)")
            needle = f"%{query.lower()}%"
            params.extend([needle, needle])
        params.append(max(1, min(int(limit), 5000)))

        with self.connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM historical_documents WHERE "
                + " AND ".join(clauses)
                + " ORDER BY published_at DESC LIMIT ?",
                params,
            )
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
