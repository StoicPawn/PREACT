"""Correlates of War State System Membership v2024 connector."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import zipfile

from preact.history.connectors.base import AcquiredDataset, BulkFileConnector
from preact.history.entities import EntityCode, PoliticalEntity

STATES_V2024_URL = "https://correlatesofwar.org/wp-content/uploads/States2024.zip"


class COWStateSystemConnector:
    def __init__(self, bulk: BulkFileConnector) -> None:
        if bulk.source_id != "cow":
            raise ValueError("BulkFileConnector source_id must be 'cow'")
        self.bulk = bulk

    def acquire(self) -> AcquiredDataset:
        return self.bulk.fetch(
            STATES_V2024_URL,
            source_release="State System Membership v2024",
            licence_reference="https://correlatesofwar.org/data-sets/state-system-membership/",
            replay_eligible_before_retrieval=False,
            notes=(
                "Versioned COW state-system release; extends through December 2024. "
                "Safe for retrospective Atlas use. Strict historical Replay must use "
                "a COW vintage published by the cutoff or contemporaneous evidence."
            ),
        )

    @staticmethod
    def parse_rows(payload: bytes) -> list[dict[str, str]]:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("States2024.zip contains no CSV file")
            text = archive.read(csv_names[0]).decode("utf-8-sig", errors="replace")
        return [dict(row) for row in csv.DictReader(io.StringIO(text))]

    @staticmethod
    def _date(row: dict[str, str], prefix: str) -> datetime:
        def ivalue(name: str, fallback: int) -> int:
            value = row.get(name)
            if value is None:
                value = row.get(name.lower())
            try:
                return int(float(str(value)))
            except (TypeError, ValueError):
                return fallback

        return datetime(
            ivalue(f"{prefix}Year", 1),
            ivalue(f"{prefix}Month", 1),
            ivalue(f"{prefix}Day", 1),
            tzinfo=timezone.utc,
        )

    @staticmethod
    def to_entities(rows: list[dict[str, str]]) -> list[PoliticalEntity]:
        from datetime import timedelta

        normalized_rows = [
            {str(k).lower(): v for k, v in row.items()}
            for row in rows
        ]
        end_dates = [
            COWStateSystemConnector._date(row, "end")
            for row in normalized_rows
        ]
        coverage_end = max(end_dates) if end_dates else None

        entities: list[PoliticalEntity] = []
        for normalized, end_inclusive in zip(normalized_rows, end_dates):
            ccode = str(normalized.get("ccode") or "").strip()
            abb = str(normalized.get("stateabb") or "").strip()
            name = str(normalized.get("statenme") or abb or ccode).strip()
            start = COWStateSystemConnector._date(normalized, "st")

            # The latest release closes still-active states at the dataset coverage
            # boundary (2024-12-31). Treat that boundary as right-censoring rather
            # than falsely claiming that all current states ended in 2024.
            right_censored = coverage_end is not None and end_inclusive == coverage_end
            valid_to = None if right_censored else end_inclusive + timedelta(days=1)

            entity_id = f"cow:{ccode}:{start.date().isoformat()}"
            entities.append(
                PoliticalEntity(
                    entity_id=entity_id,
                    name=name,
                    valid_from=start,
                    valid_to=valid_to,
                    codes=tuple(
                        code
                        for code in (
                            EntityCode("cow_ccode", ccode) if ccode else None,
                            EntityCode("cow_stateabb", abb) if abb else None,
                        )
                        if code is not None
                    ),
                    aliases=(abb,) if abb and abb.casefold() != name.casefold() else (),
                    attributes={
                        "cow_version": normalized.get("version"),
                        "source_end_date": end_inclusive.date().isoformat(),
                        "right_censored_at_release": right_censored,
                        "source_row": normalized,
                    },
                )
            )
        return entities
