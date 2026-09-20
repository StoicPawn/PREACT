"""Correlates of War network and capabilities connectors."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from typing import Callable

from .base import AcquiredDataset, BulkFileConnector


ALLIANCES_V41_URL = "https://correlatesofwar.org/wp-content/uploads/version4.1_csv.zip"
CONTIGUITY_V32_URL = "https://correlatesofwar.org/wp-content/uploads/DirectContiguity320.zip"
DYADIC_MID_V403_URL = "https://correlatesofwar.org/wp-content/uploads/dyadic_mid_4.03_update.zip"
NMC_V7_URL = "https://correlatesofwar.org/wp-content/uploads/NMCv7.zip"


def _read_zip_csv(
    payload: bytes,
    *,
    filename_predicate: Callable[[str], bool],
) -> tuple[str, list[dict[str, str]]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv") and filename_predicate(name.lower())
        ]
        if not names:
            available = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            raise ValueError(f"matching CSV not found; available={available}")
        name = sorted(names, key=len)[0]
        text = archive.read(name).decode("utf-8-sig", errors="replace")
    return name, [dict(row) for row in csv.DictReader(io.StringIO(text))]


class COWNetworkConnector:
    def __init__(self, bulk: BulkFileConnector) -> None:
        if bulk.source_id != "cow":
            raise ValueError("BulkFileConnector source_id must be 'cow'")
        self.bulk = bulk

    def acquire_alliances(self) -> AcquiredDataset:
        return self.bulk.fetch(
            ALLIANCES_V41_URL,
            source_release="COW Formal Alliances v4.1",
            licence_reference="https://correlatesofwar.org/data-sets/formal-alliances/",
            replay_eligible_before_retrieval=False,
            notes="Non-commercial COW dataset; retrospective Atlas evidence.",
        )

    @staticmethod
    def parse_alliances(payload: bytes) -> list[dict[str, str]]:
        _, rows = _read_zip_csv(
            payload,
            filename_predicate=lambda n: "directed" in n and "yearly" in n,
        )
        return rows

    def acquire_contiguity(self) -> AcquiredDataset:
        return self.bulk.fetch(
            CONTIGUITY_V32_URL,
            source_release="COW Direct Contiguity v3.2",
            licence_reference="https://correlatesofwar.org/data-sets/direct-contiguity/",
            replay_eligible_before_retrieval=False,
            notes="Non-commercial COW dataset; directed dyad-year view preferred.",
        )

    @staticmethod
    def parse_contiguity(payload: bytes) -> list[dict[str, str]]:
        _, rows = _read_zip_csv(
            payload,
            filename_predicate=lambda n: n.endswith("contdird.csv"),
        )
        return rows

    def acquire_dyadic_mids(self) -> AcquiredDataset:
        return self.bulk.fetch(
            DYADIC_MID_V403_URL,
            source_release="COW Dyadic MID v4.03",
            licence_reference="https://correlatesofwar.org/data-sets/mids/",
            replay_eligible_before_retrieval=False,
            notes="Non-commercial COW dyadic disputes, 1816-2014.",
        )

    @staticmethod
    def parse_dyadic_mids(payload: bytes) -> list[dict[str, str]]:
        _, rows = _read_zip_csv(
            payload,
            filename_predicate=lambda n: "dyadic" in n and "mid" in n,
        )
        return rows

    def acquire_nmc(self) -> AcquiredDataset:
        return self.bulk.fetch(
            NMC_V7_URL,
            source_release="COW National Material Capabilities v7.0",
            licence_reference="https://correlatesofwar.org/data-sets/national-material-capabilities/",
            replay_eligible_before_retrieval=False,
            notes="Non-commercial COW NMC release, 1816-2022.",
        )

    @staticmethod
    def parse_nmc(payload: bytes) -> list[dict[str, str]]:
        _, rows = _read_zip_csv(
            payload,
            filename_predicate=lambda n: "nmc" in n and "supp" not in n,
        )
        return rows
