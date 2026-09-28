import io
from datetime import datetime, timezone
import zipfile

from preact.feature_store.world_context import build_world_context_snapshot
from preact.history.entity_crosswalk import COWISO3SemanticCrosswalk
from preact.history.graph_store import HistoricalGraphStore
from preact.history.relations import HistoricalRelation
from preact.history.snapshot_store import SourceSnapshotStore


UTC = timezone.utc


def _cow_states_zip() -> bytes:
    csv_text = (
        "StateAbb,CCode,StateNme,StYear,StMonth,StDay,EndYear,EndMonth,EndDay,Version\n"
        "ITA,325,Italy,1861,3,17,2024,12,31,2024\n"
        "FRN,220,France,1816,1,1,2024,12,31,2024\n"
        "GMY,255,Germany,1990,10,3,2024,12,31,2024\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("States2024.csv", csv_text)
    return buffer.getvalue()


def test_crosswalk_bridges_cow_and_iso3_graph_namespaces(tmp_path):
    store = SourceSnapshotStore(tmp_path / "hub" / "snapshots")
    snapshot = store.put(
        source_id="cow",
        payload=_cow_states_zip(),
        retrieved_at=datetime(2026, 9, 22, tzinfo=UTC),
        source_url="https://example.test/States2024.zip",
        source_release="State System Membership v2024",
    )
    crosswalk = COWISO3SemanticCrosswalk(store)
    aliases = crosswalk.aliases_at(datetime(2020, 1, 1, tzinfo=UTC))

    assert aliases.canonical("cow_ccode:325") == "iso3:ITA"
    assert aliases.canonical("cow_ccode:220") == "iso3:FRA"
    assert aliases.canonical("iso3:ITA") == "iso3:ITA"
    assert aliases.source_snapshot_checksum == snapshot.checksum_sha256
    assert len(aliases.fingerprint) == 64


def test_crosswalk_allows_gdelt_event_pressure_to_flow_over_cow_structure(tmp_path):
    store = SourceSnapshotStore(tmp_path / "hub" / "snapshots")
    store.put(
        source_id="cow",
        payload=_cow_states_zip(),
        retrieved_at=datetime(2026, 9, 22, tzinfo=UTC),
        source_url="https://example.test/States2024.zip",
        source_release="State System Membership v2024",
    )
    crosswalk = COWISO3SemanticCrosswalk(store)
    alias_snapshot = crosswalk.aliases_at(datetime(2020, 1, 10, tzinfo=UTC))

    graph = HistoricalGraphStore(tmp_path / "graph.duckdb")
    graph.insert(
        [
            HistoricalRelation(
                relation_id="cow-ita-fr",
                relation_type="formal_alliance",
                subject_entity_id="cow_ccode:325",
                object_entity_id="cow_ccode:220",
                valid_from=datetime(2010, 1, 1, tzinfo=UTC),
                known_at=datetime(2010, 1, 1, tzinfo=UTC),
                source="cow",
                source_ref="alliance",
                retrieved_at=datetime(2010, 1, 1, tzinfo=UTC),
            ),
            HistoricalRelation(
                relation_id="gdelt-fr-de",
                relation_type="gdelt_material_conflict",
                subject_entity_id="iso3:FRA",
                object_entity_id="iso3:DEU",
                valid_from=datetime(2020, 1, 9, tzinfo=UTC),
                valid_to=datetime(2020, 1, 10, tzinfo=UTC),
                known_at=datetime(2020, 1, 9, 12, tzinfo=UTC),
                directed=True,
                source="gdelt",
                source_ref="story",
                retrieved_at=datetime(2020, 1, 9, 12, tzinfo=UTC),
                attributes={
                    "goldstein_scale": "-6",
                    "quad_class": "4",
                    "num_articles": "10",
                },
            ),
        ]
    )

    snapshot = build_world_context_snapshot(
        graph,
        cutoff=datetime(2020, 1, 10, tzinfo=UTC),
        windows_days=(30,),
        entity_aliases=alias_snapshot.aliases,
        entity_alias_fingerprint=alias_snapshot.fingerprint,
    )
    features = snapshot.features_for("iso3:ITA")

    assert features["world_context:focal_active_neighbors"] == 1.0
    assert features["world_context:neighbor_recent_30d:total"] == 1.0
    assert features["world_context:hop1_recent_30d:conflict_pressure"] > 0.0
