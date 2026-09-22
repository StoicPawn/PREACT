from datetime import datetime, timezone

from preact.feature_store.event_panel import build_event_risk_panel
from preact.history.graph_store import HistoricalGraphStore
from preact.history.schema import Provenance, TemporalRecord
from preact.history.warehouse import HistoricalWarehouse

UTC = timezone.utc


def test_event_panel_uses_future_event_only_as_target(tmp_path):
    w = HistoricalWarehouse(tmp_path / "h.duckdb")
    g = HistoricalGraphStore(tmp_path / "g.duckdb")
    w.insert_records([
        TemporalRecord(
            record_id="x1",
            entity_id="cow_ccode:1",
            variable="cow_nmc:cinc",
            value=0.1,
            valid_from=datetime(2000,1,1,tzinfo=UTC),
            known_at=datetime(2000,1,2,tzinfo=UTC),
            provenance=Provenance(
                source="test",source_ref="x1",
                retrieved_at=datetime(2000,1,2,tzinfo=UTC),
            ),
        ),
        TemporalRecord(
            record_id="coup",
            entity_id="cow_ccode:1",
            variable="event:coup_attempt",
            value=1,
            valid_from=datetime(2000,6,1,tzinfo=UTC),
            known_at=datetime(2001,1,1,tzinfo=UTC),
            provenance=Provenance(
                source="test",source_ref="coup",
                retrieved_at=datetime(2001,1,1,tzinfo=UTC),
            ),
        ),
    ])
    dataset=build_event_risk_panel(
        warehouse=w,graph=g,
        entity_ids=["cow_ccode:1"],
        cutoffs=[datetime(2000,3,1,tzinfo=UTC)],
        feature_variables=["cow_nmc:cinc"],
        target_variable="event:coup_attempt",
        horizon_days=120,
    )
    assert dataset.target.iloc[0]==1
    assert "event:coup_attempt" not in dataset.features.columns
    assert dataset.features.iloc[0]["cow_nmc:cinc"]==0.1
    fingerprint = dataset.feature_snapshot_fingerprints.iloc[0]
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64


def test_event_panel_fingerprint_changes_when_feature_vintage_changes(tmp_path):
    w = HistoricalWarehouse(tmp_path / "h.duckdb")
    g = HistoricalGraphStore(tmp_path / "g.duckdb")
    w.insert_records([
        TemporalRecord(
            record_id="v1", entity_id="cow_ccode:1", variable="cow_nmc:cinc", value=0.1,
            valid_from=datetime(2000,1,1,tzinfo=UTC), known_at=datetime(2000,1,2,tzinfo=UTC),
            provenance=Provenance(source="test", source_ref="v1", retrieved_at=datetime(2000,1,2,tzinfo=UTC)),
        ),
        TemporalRecord(
            record_id="v2", entity_id="cow_ccode:1", variable="cow_nmc:cinc", value=0.2,
            valid_from=datetime(2000,2,1,tzinfo=UTC), known_at=datetime(2000,2,2,tzinfo=UTC),
            provenance=Provenance(source="test", source_ref="v2", retrieved_at=datetime(2000,2,2,tzinfo=UTC)),
        ),
    ])
    dataset = build_event_risk_panel(
        warehouse=w, graph=g, entity_ids=["cow_ccode:1"],
        cutoffs=[datetime(2000,1,15,tzinfo=UTC), datetime(2000,3,1,tzinfo=UTC)],
        feature_variables=["cow_nmc:cinc"], target_variable="event:coup_attempt", horizon_days=30,
    )
    fingerprints = dataset.feature_snapshot_fingerprints.tolist()
    assert len(fingerprints) == 2
    assert fingerprints[0] != fingerprints[1]
