"""Operational state for historical-source ingestion runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Mapping


@dataclass(frozen=True)
class SourceRun:
    source_id: str
    started_at: datetime
    completed_at: datetime
    status: str
    rows_seen: int
    rows_inserted: int
    snapshots: int
    details: Mapping[str, object]


class IngestionStateStore:
    def __init__(self, path: str | Path = "data/history/ingestion_state.sqlite3") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rows_seen INTEGER NOT NULL,
                    rows_inserted INTEGER NOT NULL,
                    snapshots INTEGER NOT NULL,
                    details_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_source_runs_latest "
                "ON source_runs(source_id, completed_at DESC)"
            )

    def record(self, run: SourceRun) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO source_runs(
                    source_id, started_at, completed_at, status,
                    rows_seen, rows_inserted, snapshots, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run.source_id,
                    run.started_at.astimezone(timezone.utc).isoformat(),
                    run.completed_at.astimezone(timezone.utc).isoformat(),
                    run.status,
                    int(run.rows_seen),
                    int(run.rows_inserted),
                    int(run.snapshots),
                    json.dumps(
                        dict(run.details),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    ),
                ],
            )
            conn.commit()

    def latest(self, source_id: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM source_runs
                WHERE source_id = ?
                ORDER BY completed_at DESC, id DESC
                LIMIT 1
                """,
                [source_id],
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["details"] = json.loads(result.pop("details_json"))
        return result

    def all_latest(self) -> dict[str, dict]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT r.*
                FROM source_runs r
                JOIN (
                    SELECT source_id, MAX(id) AS max_id
                    FROM source_runs
                    GROUP BY source_id
                ) latest
                ON r.source_id = latest.source_id AND r.id = latest.max_id
                ORDER BY r.source_id
                """
            ).fetchall()
        output: dict[str, dict] = {}
        for row in rows:
            result = dict(row)
            result["details"] = json.loads(result.pop("details_json"))
            output[result["source_id"]] = result
        return output
