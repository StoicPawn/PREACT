"""SIPRI Military Expenditure release connector."""

from __future__ import annotations

from .base import AcquiredDataset, BulkFileConnector

SIPRI_MILEX_2025_V12 = (
    "https://www.sipri.org/sites/default/files/"
    "SIPRI-Milex-data-1949-2025_v1.2.xlsx"
)


class SIPRIMilitaryExpenditureConnector:
    def __init__(self, bulk: BulkFileConnector) -> None:
        if bulk.source_id != "sipri":
            raise ValueError("BulkFileConnector source_id must be 'sipri'")
        self.bulk = bulk

    def acquire(self) -> AcquiredDataset:
        return self.bulk.fetch(
            SIPRI_MILEX_2025_V12,
            source_release="SIPRI Milex 1949-2025 v1.2 (27 Apr 2026)",
            licence_reference="https://www.sipri.org/databases/milex",
            replay_eligible_before_retrieval=False,
            notes=(
                "SIPRI explicitly revises historical values. Archive exact workbook "
                "release; do not use a later replacement for an earlier replay cutoff."
            ),
        )
