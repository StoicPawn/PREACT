from datetime import datetime, timezone

from preact.feature_store.world_context import (
    build_world_context_snapshot,
    contextual_feature_fingerprint,
)
from preact.history.graph_store import HistoricalGraphStore
from preact.history.relations import HistoricalRelation


UTC = timezone.utc


def _relation(
    relation_id,
    relation_type,
    subject,
    object_,
    valid_from,
    *,
    known_at=None,
    valid_to=None,
    attributes=None,
):
    known = known_at or valid_from
    return HistoricalRelation(
        relation_id=relation_id,
        relation_type=relation_type,
        subject_entity_id=subject,
        object_entity_id=object_,
        valid_from=valid_from,
        valid_to=valid_to,
        known_at=known,
        directed=False,
        source="test",
        source_ref=relation_id,
        retrieved_at=known,
        dataset_version="test-v1",
        attributes=attributes or {},
    )


def test_world_context_propagates_neighbor_activity_without_pairwise_columns(tmp_path):
    graph = HistoricalGraphStore(tmp_path / "graph.duckdb")
    cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    graph.insert(
        [
            _relation(
                "ally-a-b",
                "formal_alliance",
                "A",
                "B",
                datetime(2010, 1, 1, tzinfo=UTC),
                valid_to=None,
            ),
            _relation(
                "b-c-dispute",
                "militarized_interstate_dispute",
                "B",
                "C",
                datetime(2019, 12, 15, tzinfo=UTC),
                valid_to=datetime(2020, 2, 1, tzinfo=UTC),
            ),
            _relation(
                "d-e-dispute",
                "militarized_interstate_dispute",
                "D",
                "E",
                datetime(2019, 12, 20, tzinfo=UTC),
                valid_to=datetime(2020, 2, 1, tzinfo=UTC),
            ),
        ]
    )

    snapshot = build_world_context_snapshot(
        graph,
        cutoff=cutoff,
        windows_days=(90, 365),
    )
    features = snapshot.features_for("A")

    assert features["world_context:focal_active_neighbors"] == 1.0
    assert features["world_context:neighbor_recent_90d:total"] == 1.0
    assert (
        features[
            "world_context:neighbor_recent_90d:militarized_interstate_dispute"
        ]
        == 1.0
    )
    assert features["world_context:second_hop_entities_90d"] == 1.0
    assert features["world_context:system_recent_90d:total"] == 2.0


def test_world_context_rejects_future_known_relation_from_strict_snapshot(tmp_path):
    graph = HistoricalGraphStore(tmp_path / "graph.duckdb")
    cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    graph.insert(
        [
            _relation(
                "known-now",
                "formal_alliance",
                "A",
                "B",
                datetime(2010, 1, 1, tzinfo=UTC),
                known_at=datetime(2010, 1, 1, tzinfo=UTC),
            ),
            _relation(
                "learned-later",
                "militarized_interstate_dispute",
                "B",
                "C",
                datetime(2019, 12, 20, tzinfo=UTC),
                known_at=datetime(2021, 1, 1, tzinfo=UTC),
            ),
        ]
    )

    snapshot = build_world_context_snapshot(graph, cutoff=cutoff, windows_days=(90,))
    features = snapshot.features_for("A")

    assert features["world_context:neighbor_recent_90d:total"] == 0.0
    assert features["world_context:system_recent_90d:total"] == 0.0


def test_context_fingerprint_changes_with_world_evidence(tmp_path):
    graph = HistoricalGraphStore(tmp_path / "graph.duckdb")
    first_cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    graph.insert(
        [
            _relation(
                "a-b",
                "formal_alliance",
                "A",
                "B",
                datetime(2010, 1, 1, tzinfo=UTC),
            )
        ]
    )
    first = build_world_context_snapshot(graph, cutoff=first_cutoff, windows_days=(365,))

    graph.insert(
        [
            _relation(
                "b-c",
                "militarized_interstate_dispute",
                "B",
                "C",
                datetime(2020, 1, 2, tzinfo=UTC),
            )
        ]
    )
    second = build_world_context_snapshot(
        graph,
        cutoff=datetime(2020, 1, 3, tzinfo=UTC),
        windows_days=(365,),
    )

    base = "a" * 64
    assert first.evidence_fingerprint != second.evidence_fingerprint
    assert (
        contextual_feature_fingerprint(base, first.evidence_fingerprint)
        != contextual_feature_fingerprint(base, second.evidence_fingerprint)
    )


def test_world_context_weights_gdelt_conflict_through_neighbor_path(tmp_path):
    graph = HistoricalGraphStore(tmp_path / "graph.duckdb")
    cutoff = datetime(2020, 1, 10, tzinfo=UTC)
    graph.insert(
        [
            _relation(
                "ally-a-b",
                "formal_alliance",
                "A",
                "B",
                datetime(2010, 1, 1, tzinfo=UTC),
            ),
            _relation(
                "gdelt-b-c",
                "gdelt_material_conflict",
                "B",
                "C",
                datetime(2020, 1, 9, tzinfo=UTC),
                valid_to=datetime(2020, 1, 10, tzinfo=UTC),
                attributes={
                    "goldstein_scale": "-8",
                    "num_articles": "20",
                    "quad_class": "4",
                },
            ),
        ]
    )

    snapshot = build_world_context_snapshot(
        graph,
        cutoff=cutoff,
        windows_days=(30,),
    )
    features = snapshot.features_for("A")

    assert features["world_context:neighbor_recent_30d:total"] == 1.0
    assert (
        features["world_context:neighbor_recent_30d:conflict_pressure"] > 0.0
    )
    assert (
        features["world_context:neighbor_recent_30d:cooperation_pressure"] == 0.0
    )
    assert (
        features["world_context:system_recent_30d:conflict_pressure"]
        >= features["world_context:neighbor_recent_30d:conflict_pressure"]
    )


def test_world_context_propagates_event_pressure_two_hops(tmp_path):
    graph = HistoricalGraphStore(tmp_path / "graph.duckdb")
    cutoff = datetime(2020, 2, 1, tzinfo=UTC)
    graph.insert(
        [
            _relation(
                "a-b",
                "formal_alliance",
                "A",
                "B",
                datetime(2010, 1, 1, tzinfo=UTC),
            ),
            _relation(
                "b-c",
                "direct_contiguity",
                "B",
                "C",
                datetime(2010, 1, 1, tzinfo=UTC),
            ),
            _relation(
                "c-d-news",
                "gdelt_verbal_conflict",
                "C",
                "D",
                datetime(2020, 1, 31, tzinfo=UTC),
                valid_to=datetime(2020, 2, 1, tzinfo=UTC),
                attributes={
                    "goldstein_scale": "-5",
                    "num_articles": "9",
                    "quad_class": "3",
                },
            ),
        ]
    )

    snapshot = build_world_context_snapshot(
        graph,
        cutoff=cutoff,
        windows_days=(30,),
        max_hops=3,
    )
    features = snapshot.features_for("A")

    assert features["world_context:hop1_recent_30d:entities"] == 1.0
    assert features["world_context:hop1_recent_30d:conflict_pressure"] == 0.0
    assert features["world_context:hop2_recent_30d:entities"] == 1.0
    assert features["world_context:hop2_recent_30d:conflict_pressure"] > 0.0
    assert features["world_context:hop3_recent_30d:entities"] == 0.0
