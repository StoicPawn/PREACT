import io
from datetime import datetime, timezone
import zipfile

from preact.history.graph_store import HistoricalGraphStore
from preact.history.relations import HistoricalRelation
from preact.history.snapshot_store import SourceSnapshotStore
from preact.intelligence.structural_relationships import load_documented_allies


UTC = timezone.utc


def _cow_states_zip() -> bytes:
    csv_text = (
        "StateAbb,CCode,StateNme,StYear,StMonth,StDay,EndYear,EndMonth,EndDay,Version\n"
        "ITA,325,Italy,1861,3,17,2024,12,31,2024\n"
        "FRN,220,France,1816,1,1,2024,12,31,2024\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("States2024.csv", csv_text)
    return buffer.getvalue()


def test_documented_allies_respect_valid_time_and_cow_entity_mapping(tmp_path):
    store = SourceSnapshotStore(tmp_path / "shared" / "snapshots")
    mapping_snapshot = store.put(
        source_id="cow",
        payload=_cow_states_zip(),
        retrieved_at=datetime(2026, 9, 22, 9, tzinfo=UTC),
        source_url="https://example.test/States2024.zip",
        source_release="State System Membership v2024",
    )

    graph_path = tmp_path / "history" / "graph.duckdb"
    graph = HistoricalGraphStore(graph_path)
    graph.insert(
        [
            HistoricalRelation(
                relation_id="alliance-ita-fra-2010",
                relation_type="formal_alliance",
                subject_entity_id="cow_ccode:325",
                object_entity_id="cow_ccode:220",
                valid_from=datetime(2010, 1, 1, tzinfo=UTC),
                valid_to=datetime(2011, 1, 1, tzinfo=UTC),
                known_at=datetime(2026, 9, 22, 9, tzinfo=UTC),
                source="cow",
                source_ref="test-alliance",
                retrieved_at=datetime(2026, 9, 22, 9, tzinfo=UTC),
                dataset_version="Formal Alliances v4.1",
            )
        ]
    )

    historical = load_documented_allies(
        tmp_path / "shared",
        graph_path=graph_path,
        focal_iso3="ITA",
        valid_at=datetime(2010, 6, 1, tzinfo=UTC),
        knowledge_cutoff=datetime(2026, 9, 22, 12, tzinfo=UTC),
    )

    assert historical.status == "ok"
    assert historical.allies == ("FRA",)
    assert historical.mapping_snapshot_checksum == mapping_snapshot.checksum_sha256
    assert historical.evidence.iloc[0]["counterpart_iso3"] == "FRA"

    current = load_documented_allies(
        tmp_path / "shared",
        graph_path=graph_path,
        focal_iso3="ITA",
        valid_at=datetime(2026, 9, 22, tzinfo=UTC),
        knowledge_cutoff=datetime(2026, 9, 22, 12, tzinfo=UTC),
    )

    assert current.allies == ()
    assert current.status == "no_active_documented_alliances"


def test_documented_allies_do_not_use_future_known_relations(tmp_path):
    store = SourceSnapshotStore(tmp_path / "shared" / "snapshots")
    store.put(
        source_id="cow",
        payload=_cow_states_zip(),
        retrieved_at=datetime(2020, 1, 1, tzinfo=UTC),
        source_url="https://example.test/States2024.zip",
        source_release="State System Membership v2024",
    )

    graph_path = tmp_path / "history" / "graph.duckdb"
    graph = HistoricalGraphStore(graph_path)
    graph.insert(
        [
            HistoricalRelation(
                relation_id="future-known",
                relation_type="formal_alliance",
                subject_entity_id="cow_ccode:325",
                object_entity_id="cow_ccode:220",
                valid_from=datetime(2010, 1, 1, tzinfo=UTC),
                valid_to=datetime(2011, 1, 1, tzinfo=UTC),
                known_at=datetime(2025, 1, 1, tzinfo=UTC),
                source="cow",
                source_ref="future-known",
                retrieved_at=datetime(2025, 1, 1, tzinfo=UTC),
                dataset_version="Formal Alliances v4.1",
            )
        ]
    )

    batch = load_documented_allies(
        tmp_path / "shared",
        graph_path=graph_path,
        focal_iso3="ITA",
        valid_at=datetime(2010, 6, 1, tzinfo=UTC),
        knowledge_cutoff=datetime(2024, 1, 1, tzinfo=UTC),
    )

    assert batch.allies == ()
    assert batch.evidence.empty
