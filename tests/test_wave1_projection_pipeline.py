from datetime import datetime, timezone

from preact.history.connectors.ucdp import UCDPPage
from preact.history.connectors.unhcr import UNHCRPage
from preact.history.connectors.world_bank import WorldBankObservation
from preact.projections.wave1 import (
    ucdp_records,
    unhcr_records,
    world_bank_records,
)


UTC = timezone.utc


def test_world_bank_projection_blocks_backdated_replay() -> None:
    retrieved = datetime(2026, 1, 1, tzinfo=UTC)
    records = world_bank_records(
        [
            WorldBankObservation(
                country_iso3="ITA",
                indicator="NY.GDP.MKTP.KD.ZG",
                year=2000,
                value=3.1,
                retrieved_at=retrieved,
                snapshot_checksum="abc",
            )
        ]
    )
    assert records[0].valid_from.year == 2000
    assert records[0].known_at == retrieved
    assert records[0].entity_id == "iso3:ITA"


def test_unhcr_projection_keeps_origin_and_asylum() -> None:
    retrieved = datetime(2026, 1, 1, tzinfo=UTC)
    page = UNHCRPage(
        dataset="population",
        page=1,
        max_pages=1,
        rows=(
            {
                "year": 2020,
                "coo_iso": "SYR",
                "coa_iso": "TUR",
                "refugees": 100.0,
            },
        ),
        retrieved_at=retrieved,
        snapshot_checksum="abc",
    )
    records = unhcr_records([page])
    assert records[0].entity_id == "iso3:TUR"
    assert records[0].attributes["origin_iso"] == "SYR"
    assert records[0].value["refugees"] == 100.0
    assert records[0].known_at == retrieved


def test_ucdp_projection_uses_release_time_only_when_explicit() -> None:
    retrieved = datetime(2026, 1, 1, tzinfo=UTC)
    release = datetime(2025, 12, 1, tzinfo=UTC)
    page = UCDPPage(
        resource="gedevents",
        version="26.1",
        page=1,
        total_pages=1,
        total_count=1,
        rows=(
            {
                "id": 7,
                "date_start": "2025-10-01",
                "country_id": 123,
                "best": 4,
            },
        ),
        retrieved_at=retrieved,
        snapshot_checksum="abc",
    )
    conservative = ucdp_records([page])[0]
    versioned = ucdp_records([page], version_release_at=release)[0]
    assert conservative.known_at == retrieved
    assert versioned.known_at == release
    assert versioned.provenance.dataset_version == "26.1"
