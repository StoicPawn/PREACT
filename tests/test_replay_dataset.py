from datetime import datetime, timezone

from preact.feature_store.replay_dataset import build_replay_dataset
from preact.history.schema import Provenance, TemporalRecord
from preact.history.warehouse import HistoricalWarehouse


UTC = timezone.utc


def test_replay_dataset_uses_past_features_and_future_event_only_as_label(tmp_path) -> None:
    warehouse = HistoricalWarehouse(tmp_path / "history.duckdb")
    warehouse.insert_records(
        [
            TemporalRecord(
                record_id="stress",
                entity_id="iso3:AAA",
                variable="stress",
                value=0.7,
                valid_from=datetime(2020, 1, 1, tzinfo=UTC),
                known_at=datetime(2020, 1, 2, tzinfo=UTC),
                provenance=Provenance(
                    source="test",
                    source_ref="stress",
                    retrieved_at=datetime(2020, 1, 2, tzinfo=UTC),
                ),
            ),
            TemporalRecord(
                record_id="future-coup",
                entity_id="iso3:AAA",
                variable="event:coup",
                value=1,
                valid_from=datetime(2020, 2, 1, tzinfo=UTC),
                known_at=datetime(2020, 2, 1, tzinfo=UTC),
                provenance=Provenance(
                    source="test",
                    source_ref="coup",
                    retrieved_at=datetime(2020, 2, 1, tzinfo=UTC),
                ),
            ),
        ]
    )
    cutoff = datetime(2020, 1, 15, tzinfo=UTC)
    dataset = build_replay_dataset(
        warehouse,
        entity_id="iso3:AAA",
        cutoffs=[cutoff],
        feature_variables=["stress"],
        target_variable="event:coup",
        horizon_days=30,
    )
    assert dataset.features.iloc[0]["stress"] == 0.7
    assert dataset.target.iloc[0] == 1
    # The future event is the label, not a feature.
    assert "event:coup" not in dataset.features.columns
