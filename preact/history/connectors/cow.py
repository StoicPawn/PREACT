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
            replay_eligible_before_retrieval=True,
            notes="Versioned COW state-system release; extends through December 2024.",
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
        entities: list[PoliticalEntity] = []
        for index, row in enumerate(rows):
            normalized = {str(k).lower(): v for k, v in row.items()}
            ccode = str(normalized.get("ccode") or "").strip()
            abb = str(normalized.get("stateabb") or "").strip()
            name = str(normalized.get("statenme") or abb or ccode).strip()
            start = COWStateSystemConnector._date(normalized, "st")
            end_inclusive = COWStateSystemConnector._date(normalized, "end")
            # COW end dates are inclusive. PoliticalEntity.valid_to is exclusive.
            valid_to = end_inclusive.replace()  # converted below with one-day delta
            from datetime import timedelta
            valid_to = valid_to + timedelta(days=1)
            entity_id = f"cow:{ccode}:{start.date().isoformat()}:{index}"
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
                        "source_row": normalized,
                    },
                )
            )
        return entities
