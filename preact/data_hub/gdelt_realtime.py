"""GDELT 2.0 realtime raw-file collector.

GDELT publishes Events, Mentions and GKG update files every 15 minutes. This
collector archives those provider-native files once in the shared hub so
multiple products do not independently download the same raw stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen
import zipfile

from preact.history.snapshot_store import SnapshotMetadata, SourceSnapshotStore

LASTUPDATE_URL = "https://data.gdeltproject.org/gdeltv2/lastupdate.txt"


@dataclass(frozen=True)
class GDELTFileRef:
    kind: str
    url: str
    expected_size: int | None = None
    expected_md5: str | None = None


@dataclass(frozen=True)
class CollectedGDELTFile:
    ref: GDELTFileRef
    snapshot: SnapshotMetadata
    downloaded: bool


def _kind_from_url(url: str) -> str:
    lowered = url.lower()
    if ".export.csv" in lowered:
        return "events"
    if ".mentions.csv" in lowered:
        return "mentions"
    if ".gkg.csv" in lowered:
        return "gkg"
    return "other"


def parse_lastupdate(text: str) -> list[GDELTFileRef]:
    refs: list[GDELTFileRef] = []
    for raw_line in text.splitlines():
        fields = raw_line.strip().split()
        if not fields:
            continue
        url = fields[-1]
        size = None
        checksum = None
        if len(fields) >= 3:
            try:
                size = int(fields[0])
            except ValueError:
                size = None
            checksum = fields[1]
        refs.append(
            GDELTFileRef(
                kind=_kind_from_url(url),
                url=url,
                expected_size=size,
                expected_md5=checksum,
            )
        )
    return refs


class GDELTRealtimeCollector:
    def __init__(
        self,
        root: str | Path,
        *,
        user_agent: str = "PREACT-SharedDataHub/0.1",
    ) -> None:
        self.root = Path(root)
        self.snapshot_store = SourceSnapshotStore(self.root / "snapshots")
        self.state_path = self.root / "gdelt_realtime_state.json"
        self.user_agent = user_agent

    def _read_state(self) -> dict[str, str]:
        if not self.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_state(self, state: dict[str, str]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def latest_refs(self) -> list[GDELTFileRef]:
        request = Request(LASTUPDATE_URL, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=30.0) as response:
            text = response.read().decode("utf-8", errors="replace")
        return parse_lastupdate(text)

    def collect_latest(
        self,
        *,
        kinds: Iterable[str] = ("events", "mentions", "gkg"),
    ) -> list[CollectedGDELTFile]:
        wanted = set(kinds)
        state = self._read_state()
        results: list[CollectedGDELTFile] = []

        for ref in self.latest_refs():
            if ref.kind not in wanted:
                continue
            existing_checksum = state.get(ref.url)
            if existing_checksum:
                # Exact raw bytes were already archived by the shared collector.
                continue

            request = Request(ref.url, headers={"User-Agent": self.user_agent})
            with urlopen(request, timeout=120.0) as response:
                payload = response.read()
                content_type = response.headers.get("Content-Type")

            if ref.expected_size is not None and len(payload) != ref.expected_size:
                raise RuntimeError(
                    f"GDELT size mismatch for {ref.url}: expected "
                    f"{ref.expected_size}, got {len(payload)}"
                )

            snapshot = self.snapshot_store.put(
                source_id="gdelt",
                payload=payload,
                retrieved_at=datetime.now(timezone.utc),
                source_url=ref.url,
                source_release=Path(ref.url).name,
                content_type=content_type,
                operation=f"realtime_{ref.kind}",
                notes=f"gdelt_realtime_kind={ref.kind}; md5={ref.expected_md5 or ''}",
            )
            state[ref.url] = snapshot.checksum_sha256
            results.append(CollectedGDELTFile(ref=ref, snapshot=snapshot, downloaded=True))

        self._write_state(state)
        return results


_EVENT_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName",
    "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL",
]


def parse_event_zip(payload: bytes) -> list[dict[str, str]]:
    """Parse a GDELT Events export ZIP into named records."""

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if not names:
            return []
        raw = archive.read(names[0]).decode("utf-8", errors="replace")

    rows: list[dict[str, str]] = []
    for values in csv.reader(io.StringIO(raw), delimiter="\t"):
        if not values:
            continue
        padded = values + [""] * max(0, len(_EVENT_COLUMNS) - len(values))
        rows.append(dict(zip(_EVENT_COLUMNS, padded[: len(_EVENT_COLUMNS)])))
    return rows
