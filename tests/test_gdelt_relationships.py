import csv
import io
from datetime import datetime, timezone
import zipfile

from preact.data_hub.gdelt_realtime import _EVENT_COLUMNS
from preact.history.snapshot_store import SourceSnapshotStore
from preact.intelligence.gdelt_relationships import (
    load_recent_relationship_edges,
    relationship_evidence,
)


def _event_zip(rows):
    raw = io.StringIO()
    writer = csv.writer(raw, delimiter="\t", lineterminator="\n")
    for values in rows:
        row = [""] * len(_EVENT_COLUMNS)
        for key, value in values.items():
            row[_EVENT_COLUMNS.index(key)] = str(value)
        writer.writerow(row)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("events.export.csv", raw.getvalue())
    return payload.getvalue()


def test_snapshot_loader_is_point_in_time_and_deduplicates_events(tmp_path):
    store = SourceSnapshotStore(tmp_path / "snapshots")
    payload = _event_zip(
        [
            {
                "GLOBALEVENTID": "1",
                "SQLDATE": "20260921",
                "Actor1CountryCode": "ITA",
                "Actor2CountryCode": "FRA",
                "GoldsteinScale": "6",
                "AvgTone": "4",
                "NumArticles": "5",
                "SOURCEURL": "https://example.test/a",
            },
            {
                "GLOBALEVENTID": "2",
                "SQLDATE": "20260921",
                "Actor1CountryCode": "GMY",
                "Actor2CountryCode": "ITA",
                "GoldsteinScale": "1",
                "AvgTone": "1",
                "NumArticles": "2",
            },
            {
                "GLOBALEVENTID": "3",
                "SQLDATE": "20260923",
                "Actor1CountryCode": "ITA",
                "Actor2CountryCode": "FRA",
                "GoldsteinScale": "-8",
                "AvgTone": "-6",
                "NumArticles": "10",
            },
        ]
    )
    for minute in (0, 15):
        store.put(
            source_id="gdelt",
            payload=payload,
            retrieved_at=datetime(2026, 9, 22, 10, minute, tzinfo=timezone.utc),
            source_url=f"https://example.test/{minute}.export.csv.zip",
            operation="realtime_events",
        )

    batch = load_recent_relationship_edges(
        tmp_path,
        as_of=datetime(2026, 9, 22, 12, tzinfo=timezone.utc),
        lookback_days=2,
    )

    assert batch.snapshot_count == 2
    assert batch.raw_event_count == 6
    assert batch.resolved_interaction_count == 1
    assert batch.resolution_rate == 1 / 6
    assert len(batch.edges) == 1
    edge = batch.edges.iloc[0]
    assert edge["source"] == "ITA"
    assert edge["target"] == "FRA"
    assert edge["events"] == 1
    assert edge["avg_goldstein"] == 6.0

    evidence = relationship_evidence(batch, focal_iso3="ITA")
    assert len(evidence) == 1
    assert evidence.iloc[0]["counterpart_iso3"] == "FRA"
    assert evidence.iloc[0]["goldstein"] == 6.0
    assert evidence.iloc[0]["source_url"] == "https://example.test/a"

    france_only = relationship_evidence(
        batch,
        focal_iso3="ITA",
        counterpart_iso3="FRA",
    )
    assert len(france_only) == 1

    germany_only = relationship_evidence(
        batch,
        focal_iso3="ITA",
        counterpart_iso3="DEU",
    )
    assert germany_only.empty


def test_snapshot_loader_excludes_snapshots_retrieved_after_as_of(tmp_path):
    store = SourceSnapshotStore(tmp_path / "snapshots")
    payload = _event_zip(
        [
            {
                "GLOBALEVENTID": "1",
                "SQLDATE": "20260922",
                "Actor1CountryCode": "ITA",
                "Actor2CountryCode": "FRA",
            }
        ]
    )
    store.put(
        source_id="gdelt",
        payload=payload,
        retrieved_at=datetime(2026, 9, 22, 13, tzinfo=timezone.utc),
        source_url="https://example.test/future.export.csv.zip",
        operation="realtime_events",
    )

    batch = load_recent_relationship_edges(
        tmp_path,
        as_of=datetime(2026, 9, 22, 12, tzinfo=timezone.utc),
        lookback_days=1,
    )

    assert batch.snapshot_count == 0
    assert batch.edges.empty
