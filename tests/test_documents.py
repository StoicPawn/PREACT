from datetime import datetime, timezone

import pytest

from preact.history.document_store import HistoricalDocumentStore
from preact.history.documents import HistoricalDocument, TextAvailability


UTC = timezone.utc


def test_metadata_only_document_rejects_persisted_text() -> None:
    with pytest.raises(ValueError):
        HistoricalDocument(
            document_id="x",
            source_id="news",
            source_ref="x",
            title="Title",
            published_at=datetime(2020, 1, 1, tzinfo=UTC),
            known_at=datetime(2020, 1, 1, tzinfo=UTC),
            acquired_at=datetime(2020, 1, 2, tzinfo=UTC),
            text="copyright text",
            text_availability=TextAvailability.METADATA_ONLY,
        )


def test_document_store_as_of_excludes_later_known_document(tmp_path) -> None:
    store = HistoricalDocumentStore(tmp_path / "docs.duckdb")
    documents = [
        HistoricalDocument(
            document_id="old",
            source_id="archive",
            source_ref="old",
            title="Known",
            published_at=datetime(1962, 10, 20, tzinfo=UTC),
            known_at=datetime(1962, 10, 20, tzinfo=UTC),
            acquired_at=datetime(2026, 1, 1, tzinfo=UTC),
            text_availability=TextAvailability.METADATA_ONLY,
        ),
        HistoricalDocument(
            document_id="future",
            source_id="archive",
            source_ref="future",
            title="Later",
            published_at=datetime(1962, 10, 25, tzinfo=UTC),
            known_at=datetime(1962, 10, 25, tzinfo=UTC),
            acquired_at=datetime(2026, 1, 1, tzinfo=UTC),
            text_availability=TextAvailability.METADATA_ONLY,
        ),
    ]
    # Acquired later is allowed: known_at models historical public availability,
    # while acquired_at records when PREACT obtained the archive copy.
    assert store.insert(documents) == 2
    rows = store.as_of(cutoff=datetime(1962, 10, 22, tzinfo=UTC))
    assert [row["document_id"] for row in rows] == ["old"]
