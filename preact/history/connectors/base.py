"""Connector primitives for versioned historical source acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Mapping
from urllib.request import Request, urlopen

from preact.history.snapshot_store import SnapshotMetadata, SourceSnapshotStore


@dataclass(frozen=True)
class AcquiredDataset:
    source_id: str
    retrieved_at: datetime
    snapshot: SnapshotMetadata
    payload: bytes
    replay_eligible_before_retrieval: bool
    notes: str = ""


class BulkFileConnector:
    """Download a bulk file and preserve the exact bytes as an immutable vintage."""

    def __init__(
        self,
        source_id: str,
        snapshot_store: SourceSnapshotStore,
        *,
        user_agent: str = "PREACT-HistoricalIngestion/0.1",
    ) -> None:
        self.source_id = source_id
        self.snapshot_store = snapshot_store
        self.user_agent = user_agent

    def fetch(
        self,
        url: str,
        *,
        source_release: str | None = None,
        licence_reference: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 120.0,
        replay_eligible_before_retrieval: bool = True,
        notes: str = "",
    ) -> AcquiredDataset:
        request_headers = {"User-Agent": self.user_agent}
        request_headers.update(dict(headers or {}))
        request = Request(url, headers=request_headers)
        with urlopen(request, timeout=float(timeout_seconds)) as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type")

        retrieved_at = datetime.now(timezone.utc)
        snapshot = self.snapshot_store.put(
            source_id=self.source_id,
            payload=payload,
            retrieved_at=retrieved_at,
            source_url=url,
            source_release=source_release,
            content_type=content_type,
            licence_reference=licence_reference,
        )
        return AcquiredDataset(
            source_id=self.source_id,
            retrieved_at=retrieved_at,
            snapshot=snapshot,
            payload=payload,
            replay_eligible_before_retrieval=bool(replay_eligible_before_retrieval),
            notes=notes,
        )
