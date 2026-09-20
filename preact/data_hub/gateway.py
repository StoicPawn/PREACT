"""Centralised provider access with cache, throttling and immutable snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from preact.history.snapshot_store import SourceSnapshotStore


@dataclass(frozen=True)
class ProviderResponse:
    source_id: str
    operation: str
    payload: Any
    retrieved_at: datetime
    cached: bool
    request_fingerprint: str
    snapshot_checksum: str | None = None


class SharedProviderGateway:
    """One outbound-provider connection layer for multiple internal products.

    The gateway owns:
    - request canonicalisation and deduplication;
    - per-source throttling;
    - TTL response cache;
    - immutable raw source snapshots.

    Downstream products own only domain-specific transformations.
    """

    def __init__(
        self,
        root: str | Path = "data/shared_hub",
        *,
        user_agent: str = "PREACT-SharedDataHub/0.1",
    ) -> None:
        self.root = Path(root)
        self.cache_root = self.root / "cache"
        self.snapshot_store = SourceSnapshotStore(self.root / "snapshots")
        self.user_agent = user_agent
        self._locks: dict[str, threading.Lock] = {}
        self._request_locks: dict[str, threading.Lock] = {}
        self._last_request: dict[str, float] = {}

    @staticmethod
    def fingerprint(source_id: str, operation: str, params: Mapping[str, Any]) -> str:
        material = json.dumps(
            {
                "source_id": source_id,
                "operation": operation,
                "params": dict(sorted((str(k), v) for k, v in params.items())),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return sha256(material).hexdigest()

    def _cache_path(self, source_id: str, fingerprint: str) -> Path:
        return self.cache_root / source_id / f"{fingerprint}.json"

    def _read_cache(
        self,
        *,
        source_id: str,
        operation: str,
        fingerprint: str,
        ttl_seconds: int,
    ) -> ProviderResponse | None:
        path = self._cache_path(source_id, fingerprint)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            retrieved_at = datetime.fromisoformat(str(raw["retrieved_at"]).replace("Z", "+00:00"))
            if retrieved_at.tzinfo is None:
                retrieved_at = retrieved_at.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - retrieved_at > timedelta(seconds=max(0, ttl_seconds)):
                return None
            return ProviderResponse(
                source_id=source_id,
                operation=operation,
                payload=raw["payload"],
                retrieved_at=retrieved_at.astimezone(timezone.utc),
                cached=True,
                request_fingerprint=fingerprint,
                snapshot_checksum=raw.get("snapshot_checksum"),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _write_cache(self, response: ProviderResponse) -> None:
        path = self._cache_path(response.source_id, response.request_fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "source_id": response.source_id,
                    "operation": response.operation,
                    "payload": response.payload,
                    "retrieved_at": response.retrieved_at.astimezone(timezone.utc).isoformat(),
                    "snapshot_checksum": response.snapshot_checksum,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)

    def _throttle(self, source_id: str, minimum_interval_seconds: float) -> None:
        lock = self._locks.setdefault(source_id, threading.Lock())
        lock.acquire()
        try:
            elapsed = time.monotonic() - self._last_request.get(source_id, 0.0)
            wait = max(0.0, float(minimum_interval_seconds) - elapsed)
            if wait:
                time.sleep(wait)
            self._last_request[source_id] = time.monotonic()
        finally:
            lock.release()

    def get_json(
        self,
        *,
        source_id: str,
        operation: str,
        url: str,
        params: Mapping[str, Any],
        ttl_seconds: int,
        minimum_interval_seconds: float = 0.0,
        timeout_seconds: float = 30.0,
        headers: Mapping[str, str] | None = None,
    ) -> ProviderResponse:
        """Fetch JSON once for all internal consumers, snapshot it, then fan out."""

        fingerprint = self.fingerprint(source_id, operation, params)
        cached = self._read_cache(
            source_id=source_id,
            operation=operation,
            fingerprint=fingerprint,
            ttl_seconds=ttl_seconds,
        )
        if cached is not None:
            return cached

        # Single-flight by canonical request. With the service configured as one
        # worker, concurrent PREACT/GoldenBull calls for the same provider query
        # collapse into one external request and one immutable snapshot.
        request_lock = self._request_locks.setdefault(fingerprint, threading.Lock())
        with request_lock:
            # Another consumer may have completed the request while we waited.
            cached = self._read_cache(
                source_id=source_id,
                operation=operation,
                fingerprint=fingerprint,
                ttl_seconds=ttl_seconds,
            )
            if cached is not None:
                return cached

            self._throttle(source_id, minimum_interval_seconds)

            query = urlencode([(str(k), str(v)) for k, v in params.items()])
            request_url = f"{url}?{query}" if query else url
            request_headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
            request_headers.update(dict(headers or {}))
            request = Request(request_url, headers=request_headers)

            with urlopen(request, timeout=float(timeout_seconds)) as http_response:
                payload_bytes = http_response.read()
                content_type = http_response.headers.get("Content-Type")

            retrieved_at = datetime.now(timezone.utc)
            snapshot = self.snapshot_store.put(
                source_id=source_id,
                payload=payload_bytes,
                retrieved_at=retrieved_at,
                source_url=request_url,
                content_type=content_type,
                request=dict(params),
            )
            payload = json.loads(payload_bytes.decode("utf-8"))

            response = ProviderResponse(
                source_id=source_id,
                operation=operation,
                payload=payload,
                retrieved_at=retrieved_at,
                cached=False,
                request_fingerprint=fingerprint,
                snapshot_checksum=snapshot.checksum_sha256,
            )
            self._write_cache(response)
            return response
