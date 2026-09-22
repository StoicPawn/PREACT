from datetime import datetime, timezone

from preact.history.connectors.world_bank import WorldBankObservation
from preact.intelligence.country_profile import IndicatorSpec, latest_snapshot


def test_latest_snapshot_prefers_latest_non_null_and_preserves_provenance():
    spec = IndicatorSpec("X.TEST", "Test", 1_000.0, " k")
    observations = [
        WorldBankObservation(
            country_iso3="ITA",
            indicator="X.TEST",
            year=2023,
            value=1000.0,
            retrieved_at=datetime(2026, 9, 22, tzinfo=timezone.utc),
            snapshot_checksum="old",
        ),
        WorldBankObservation(
            country_iso3="ITA",
            indicator="X.TEST",
            year=2025,
            value=None,
            retrieved_at=datetime(2026, 9, 22, tzinfo=timezone.utc),
            snapshot_checksum="null",
        ),
        WorldBankObservation(
            country_iso3="ITA",
            indicator="X.TEST",
            year=2024,
            value=2500.0,
            retrieved_at=datetime(2026, 9, 22, tzinfo=timezone.utc),
            snapshot_checksum="new",
        ),
    ]

    snapshot = latest_snapshot(observations, spec)

    assert snapshot.year == 2024
    assert snapshot.value == 2500.0
    assert snapshot.display_value == 2.5
    assert snapshot.snapshot_checksum == "new"


def test_latest_snapshot_returns_explicit_missing_value():
    spec = IndicatorSpec("X.TEST", "Test")
    snapshot = latest_snapshot([], spec)

    assert snapshot.year is None
    assert snapshot.value is None
    assert snapshot.display_value is None
    assert snapshot.snapshot_checksum is None
