"""Durable per-consumer checkpointing for shared raw snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3


class ProjectionLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_snapshots (
                    consumer_id TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    PRIMARY KEY (consumer_id, snapshot_id)
                )
                """
            )

    def seen(self, consumer_id: str, snapshot_id: str) -> bool:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT 1 FROM processed_snapshots "
                "WHERE consumer_id = ? AND snapshot_id = ?",
                [consumer_id, snapshot_id],
            ).fetchone()
        return row is not None

    def mark(self, consumer_id: str, snapshot_id: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO processed_snapshots
                (consumer_id, snapshot_id, processed_at)
                VALUES (?, ?, ?)
                """,
                [
                    consumer_id,
                    snapshot_id,
                    datetime.now(timezone.utc).isoformat(),
                ],
            )
            conn.commit()
