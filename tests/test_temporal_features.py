from datetime import datetime, timezone

from preact.feature_store.temporal import entity_feature_frame
from preact.history.schema import Provenance, TemporalRecord
from preact.history.warehouse import HistoricalWarehouse


UTC = timezone.utc


def test_temporal_features_use_latest_vintage_known_at_each_cutoff(tmp_path) -> None:
    warehouse = HistoricalWarehouse(tmp_path / "history.duckdb")
    records = [
        TemporalRecord(
            record_id="gdp-v1",
            entity_id="iso3:ITA",
            variable="world_bank:gdp",
            value=100.0,
            valid_from=datetime(2020, 1, 1, tzinfo=UTC),
            valid_to=datetime(2021, 1, 1, tzinfo=UTC),
            known_at=datetime(2021, 1, 10, tzinfo=UTC),
            provenance=Provenance(
                source="world_bank",
                source_ref="v1",
                retrieved_at=datetime(2021, 1, 10, tzinfo=UTC),
            ),
        ),
        TemporalRecord(
            record_id="gdp-v2",
            entity_id="iso3:ITA",
            variable="world_bank:gdp",
            value=101.5,
            valid_from=datetime(2020, 1, 1, tzinfo=UTC),
            valid_to=datetime(2021, 1, 1, tzinfo=UTC),
            known_at=datetime(2022, 1, 10, tzinfo=UTC),
            provenance=Provenance(
                source="world_bank",
                source_ref="v2",
                retrieved_at=datetime(2022, 1, 10, tzinfo=UTC),
            ),
        ),
    ]
    warehouse.insert_records(records)

    old_rows = warehouse.latest_state_as_of(
        cutoff=datetime(2021, 6, 1, tzinfo=UTC),
        valid_at=datetime(2020, 6, 1, tzinfo=UTC),
        entity_id="iso3:ITA",
    )
    new_rows = warehouse.latest_state_as_of(
        cutoff=datetime(2022, 6, 1, tzinfo=UTC),
        valid_at=datetime(2020, 6, 1, tzinfo=UTC),
        entity_id="iso3:ITA",
    )
    assert old_rows[0]["record_id"] == "gdp-v1"
    assert new_rows[0]["record_id"] == "gdp-v2"

    frame = entity_feature_frame(
        warehouse,
        entity_id="iso3:ITA",
        cutoffs=[
            datetime(2021, 6, 1, tzinfo=UTC),
            datetime(2022, 6, 1, tzinfo=UTC),
        ],
        variables=["world_bank:gdp"],
    )
    # Reported indicators carry forward until a newer period is available, but
    # the vintage used at each cutoff must still be the one known at that time.
    assert frame.iloc[0]["world_bank:gdp"] == 100.0
    assert frame.iloc[1]["world_bank:gdp"] == 101.5
