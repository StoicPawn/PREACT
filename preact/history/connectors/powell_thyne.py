"""Powell-Thyne Global Instances of Coups connector with explicit vintages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO, StringIO
import csv

import pandas as pd

from .base import AcquiredDataset, BulkFileConnector


@dataclass(frozen=True)
class PowellThyneRelease:
    release_id: str
    url: str
    published_at: datetime
    format: str
    provisional: bool = False


CURRENT_2026_08_29 = PowellThyneRelease(
    release_id="2026-08-29",
    url="https://jonathanmpowell.com/wp-content/uploads/2026/08/pt_20260829.csv",
    published_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
    format="csv",
    provisional=True,
)

ARCHIVE_2026_07_08 = PowellThyneRelease(
    release_id="2026-07-08",
    url="https://jonathanmpowell.com/wp-content/uploads/2026/08/pt_20260708-1.xlsx",
    published_at=datetime(2026, 7, 8, tzinfo=timezone.utc),
    format="xlsx",
    provisional=False,
)


class PowellThyneCoupConnector:
    def __init__(self, bulk: BulkFileConnector) -> None:
        if bulk.source_id != "powell_thyne_coups":
            raise ValueError(
                "BulkFileConnector source_id must be 'powell_thyne_coups'"
            )
        self.bulk = bulk

    def acquire(
        self,
        release: PowellThyneRelease = CURRENT_2026_08_29,
    ) -> AcquiredDataset:
        return self.bulk.fetch(
            release.url,
            source_release=f"Powell-Thyne {release.release_id}",
            licence_reference="https://jonathanmpowell.com/coups/",
            replay_eligible_before_retrieval=True,
            notes=(
                f"Published {release.published_at.date().isoformat()}; "
                f"provisional={release.provisional}. Preserve exact vintage."
            ),
        )

    @staticmethod
    def parse(
        payload: bytes,
        *,
        format: str,
    ) -> list[dict[str, object]]:
        fmt = format.strip().lower()
        if fmt == "xlsx":
            frame = pd.read_excel(BytesIO(payload))
            return frame.where(pd.notna(frame), None).to_dict(orient="records")
        if fmt != "csv":
            raise ValueError(f"unsupported Powell-Thyne format: {format}")
        text = payload.decode("utf-8-sig", errors="replace")
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
        return [
            dict(row)
            for row in csv.DictReader(StringIO(text), dialect=dialect)
        ]
