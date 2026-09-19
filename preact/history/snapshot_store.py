"""Immutable raw-source snapshots for leakage-safe historical replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class SnapshotMetadata:
    """Metadata required to reproduce and audit a source acquisition."""

    source_id: str
    retrieved_at: datetime
    source_url: str
    checksum_sha256: str
    payload_path: str
    source_release: Optional[str] = None
    content_type: Optional[str] = None
    licence_reference: Optional[str] = None
    request: Mapping[str, Any] = field(default_factory=dict)

    def as_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["retrieved_at"] = self.retrieved_at.astimezone(timezone.utc).isoformat()
        return data


class SourceSnapshotStore:
    """Content-addressed immutable storage for raw source payloads.

    The same payload may be referenced by multiple retrieval metadata records, but a payload
    is never overwritten. Historical replay can therefore bind a model run to exact source
    bytes and their retrieval time.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.payload_root = self.root / "payloads"
        self.metadata_root = self.root / "metadata"

    @staticmethod
    def _normalise_time(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    def put(
        self,
        *,
        source_id: str,
        payload: bytes,
        retrieved_at: datetime,
        source_url: str,
        source_release: str | None = None,
        content_type: str | None = None,
        licence_reference: str | None = None,
        request: Mapping[str, Any] | None = None,
    ) -> SnapshotMetadata:
        """Persist raw bytes and an immutable acquisition record."""

        retrieved_at = self._normalise_time(retrieved_at)
        digest = sha256(payload).hexdigest()

        payload_dir = self.payload_root / source_id / digest[:2]
        payload_dir.mkdir(parents=True, exist_ok=True)
        payload_path = payload_dir / digest

        if payload_path.exists():
            if payload_path.read_bytes() != payload:
                raise RuntimeError("SHA-256 collision or corrupted snapshot payload")
        else:
            payload_path.write_bytes(payload)

        stamp = retrieved_at.strftime("%Y%m%dT%H%M%S.%fZ")
        metadata_dir = self.metadata_root / source_id / retrieved_at.strftime("%Y/%m/%d")
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / f"{stamp}-{digest}.json"

        metadata = SnapshotMetadata(
            source_id=source_id,
            retrieved_at=retrieved_at,
            source_url=source_url,
            checksum_sha256=digest,
            payload_path=str(payload_path.relative_to(self.root)),
            source_release=source_release,
            content_type=content_type,
            licence_reference=licence_reference,
            request=dict(request or {}),
        )

        encoded = json.dumps(
            metadata.as_json_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")

        if metadata_path.exists():
            if metadata_path.read_bytes() != encoded:
                raise RuntimeError("snapshot metadata path already exists with different content")
        else:
            metadata_path.write_bytes(encoded)

        return metadata

    def read_payload(self, snapshot: SnapshotMetadata) -> bytes:
        """Read and verify the exact raw payload referenced by a snapshot."""

        payload = (self.root / snapshot.payload_path).read_bytes()
        digest = sha256(payload).hexdigest()
        if digest != snapshot.checksum_sha256:
            raise RuntimeError("snapshot checksum verification failed")
        return payload
