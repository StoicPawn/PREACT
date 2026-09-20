"""CShapes 2.0 historical border snapshot connector."""

from __future__ import annotations

import csv
import io
from typing import Any

from .base import AcquiredDataset, BulkFileConnector

CSHAPES_20_CSV = "https://icr.ethz.ch/data/cshapes/CShapes-2.0.csv"


class CShapesConnector:
    def __init__(self, bulk: BulkFileConnector) -> None:
        if bulk.source_id != "cshapes":
            raise ValueError("BulkFileConnector source_id must be 'cshapes'")
        self.bulk = bulk

    def acquire(self) -> AcquiredDataset:
        return self.bulk.fetch(
            CSHAPES_20_CSV,
            source_release="CShapes 2.0",
            licence_reference="https://icr.ethz.ch/data/cshapes/",
            replay_eligible_before_retrieval=False,
            notes=(
                "Historical geometry for retrospective reconstruction. "
                "Strict replay before publication requires contemporaneous geometry/vintage."
            ),
        )

    @staticmethod
    def parse_rows(payload: bytes) -> list[dict[str, str]]:
        text = payload.decode("utf-8-sig", errors="replace")
        return [dict(row) for row in csv.DictReader(io.StringIO(text))]
