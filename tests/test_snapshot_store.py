from datetime import datetime, timezone

import pytest

from preact.history.snapshot_store import SourceSnapshotStore


UTC = timezone.utc


def test_snapshot_store_preserves_exact_payload_and_metadata(tmp_path) -> None:
    store = SourceSnapshotStore(tmp_path)
    retrieved_at = datetime(2008, 9, 15, 13, 30, tzinfo=UTC)
    payload = b'{"value": 42}'

    snapshot = store.put(
        source_id="world_bank",
        payload=payload,
        retrieved_at=retrieved_at,
        source_url="https://example.test/data",
        source_release="2008-09-15",
        content_type="application/json",
        licence_reference="test-licence",
        request={"country": "USA", "indicator": "X"},
    )

    assert store.read_payload(snapshot) == payload
    assert snapshot.retrieved_at == retrieved_at
    assert snapshot.request["country"] == "USA"
    assert (tmp_path / snapshot.payload_path).exists()


def test_identical_payload_is_content_addressed_not_overwritten(tmp_path) -> None:
    store = SourceSnapshotStore(tmp_path)
    payload = b"same bytes"

    first = store.put(
        source_id="gdelt",
        payload=payload,
        retrieved_at=datetime(2020, 1, 1, tzinfo=UTC),
        source_url="https://example.test/one",
    )
    second = store.put(
        source_id="gdelt",
        payload=payload,
        retrieved_at=datetime(2020, 1, 2, tzinfo=UTC),
        source_url="https://example.test/two",
    )

    assert first.payload_path == second.payload_path
    assert store.read_payload(first) == payload
    assert store.read_payload(second) == payload


def test_naive_retrieval_time_is_rejected(tmp_path) -> None:
    store = SourceSnapshotStore(tmp_path)

    with pytest.raises(ValueError):
        store.put(
            source_id="vdem",
            payload=b"x",
            retrieved_at=datetime(2020, 1, 1),
            source_url="https://example.test",
        )


def test_checksum_detects_tampering(tmp_path) -> None:
    store = SourceSnapshotStore(tmp_path)
    snapshot = store.put(
        source_id="cow",
        payload=b"original",
        retrieved_at=datetime(2020, 1, 1, tzinfo=UTC),
        source_url="https://example.test",
    )

    (tmp_path / snapshot.payload_path).write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="checksum"):
        store.read_payload(snapshot)
