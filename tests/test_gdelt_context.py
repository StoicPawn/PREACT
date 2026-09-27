from datetime import datetime, timezone
import io
import zipfile

from preact.history.snapshot_store import SourceSnapshotStore
from preact.history.world_knowledge_store import WorldKnowledgeStore
from preact.intelligence.gdelt_context import load_recent_gdelt_context
from preact.intelligence.gdelt_ingest import _GKG_COLUMNS, _MENTION_COLUMNS


def utc(hour: int) -> datetime:
    return datetime(2026, 9, 27, hour, tzinfo=timezone.utc)


def zipped(values: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", "\t".join(values) + "\n")
    return buffer.getvalue()


def test_context_consumer_uses_shared_snapshots_without_network(tmp_path):
    store = SourceSnapshotStore(tmp_path / "snapshots")
    store.put(
        source_id="gdelt",
        payload=b"CODE\tLABEL\nIT\tItaly\nFR\tFrance\n",
        retrieved_at=utc(8),
        source_url="https://example.test/country",
        source_release="GDELT CAMEO country lookup",
        operation="reference_cameo_country",
    )

    mention = [""] * len(_MENTION_COLUMNS)
    mention[0] = "123"
    mention[2] = "20260927100000"
    mention[4] = "wire.example"
    mention[11] = "85"
    store.put(
        source_id="gdelt",
        payload=zipped(mention),
        retrieved_at=utc(10),
        source_url="https://example.test/mentions",
        source_release="mentions.zip",
        operation="realtime_mentions",
    )

    gkg = [""] * len(_GKG_COLUMNS)
    gkg[0] = "20260927100000-0"
    gkg[1] = "20260927100000"
    gkg[3] = "paper.example"
    gkg[4] = "https://paper.example/story"
    gkg[7] = "DIPLOMACY;SANCTIONS;"
    gkg[10] = "1#Italy#IT#IT##42#12#IT#1;1#France#FR#FR##46#2#FR#2"
    gkg[11] = "Person A"
    gkg[13] = "European Union"
    gkg[15] = "-1.2,0,0,0,0,0,100"
    store.put(
        source_id="gdelt",
        payload=zipped(gkg),
        retrieved_at=utc(10),
        source_url="https://example.test/gkg",
        source_release="gkg.zip",
        operation="realtime_gkg",
    )

    batch = load_recent_gdelt_context(
        tmp_path,
        as_of=utc(11),
        lookback_days=1,
    )
    assert batch.mention_snapshot_count == 1
    assert batch.gkg_snapshot_count == 1
    assert batch.mention_observations[0]["distinct_source_count"] == 1
    assert batch.gkg_documents[0]["country_iso3"] == ["FRA", "ITA"]
    assert "DIPLOMACY" in batch.gkg_documents[0]["themes"]
    assert len(batch.processed_snapshots) == 2

    world = WorldKnowledgeStore(tmp_path / "world.duckdb")
    persisted = world.record_gdelt_context(
        mention_observations=batch.mention_observations,
        gkg_documents=batch.gkg_documents,
        processed_snapshots=batch.processed_snapshots,
    )
    assert persisted["inserted_mentions"] == 1
    assert persisted["inserted_gkg_documents"] == 1
    assert persisted["marked_snapshots"] == 2

    country_context = world.gkg_context_for_country(
        "country:ITA",
        as_of=utc(11),
        known_cutoff=utc(11),
    )
    assert len(country_context) == 1
    assert country_context[0]["country_iso3"] == ["FRA", "ITA"]

    second = load_recent_gdelt_context(
        tmp_path,
        as_of=utc(11),
        lookback_days=1,
        skip_snapshot_checksums=world.processed_gdelt_context_snapshots(),
    )
    assert second.mention_snapshot_count == 0
    assert second.gkg_snapshot_count == 0


def test_gkg_fips_override_beats_conflicting_iso2_code():
    from preact.intelligence.gdelt_context import _resolve_gkg_country_code

    assert _resolve_gkg_country_code("GM", None) == "DEU"
    assert _resolve_gkg_country_code("UK", None) == "GBR"
