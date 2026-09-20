"""Maddison Project Database 2023 connector via Dataverse persistent DOI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from preact.data_hub.gateway import SharedProviderGateway
from .base import AcquiredDataset, BulkFileConnector


@dataclass(frozen=True)
class MaddisonReleaseFile:
    datafile_id: int
    label: str
    version: str | None


class Maddison2023Connector:
    DOI = "doi:10.34894/INZBF2"
    DATASET_API = "https://dataverse.nl/api/datasets/:persistentId/"
    FILE_API = "https://dataverse.nl/api/access/datafile"
    TARGET_FILE = "mpd2023_web.xlsx"

    def __init__(
        self,
        gateway: SharedProviderGateway,
        bulk: BulkFileConnector,
    ) -> None:
        if bulk.source_id != "maddison":
            raise ValueError("BulkFileConnector source_id must be 'maddison'")
        self.gateway = gateway
        self.bulk = bulk

    def resolve_file(self) -> MaddisonReleaseFile:
        response = self.gateway.get_json(
            source_id="maddison",
            operation="dataverse_metadata:mpd2023",
            url=self.DATASET_API,
            params={"persistentId": self.DOI},
            ttl_seconds=86400,
            minimum_interval_seconds=0.2,
            timeout_seconds=60.0,
        )
        payload = response.payload if isinstance(response.payload, dict) else {}
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        latest = data.get("latestVersion", {}) if isinstance(data, dict) else {}
        files = latest.get("files", []) if isinstance(latest, dict) else []
        for item in files:
            if not isinstance(item, dict):
                continue
            data_file = item.get("dataFile", {})
            if not isinstance(data_file, dict):
                continue
            label = str(data_file.get("filename") or item.get("label") or "")
            if label == self.TARGET_FILE:
                return MaddisonReleaseFile(
                    datafile_id=int(data_file["id"]),
                    label=label,
                    version=str(latest.get("versionNumber") or "") or None,
                )
        raise RuntimeError(f"{self.TARGET_FILE} not found in Maddison Dataverse release")

    def acquire(self) -> AcquiredDataset:
        resolved = self.resolve_file()
        return self.bulk.fetch(
            f"{self.FILE_API}/{resolved.datafile_id}",
            source_release=f"MPD 2023 Dataverse V{resolved.version or '1.0'}",
            licence_reference="https://doi.org/10.34894/INZBF2",
            replay_eligible_before_retrieval=False,
            notes=(
                "CC BY 4.0 long-run estimates. Retrospective Atlas evidence; "
                "strict replay needs the vintage actually available at the cutoff."
            ),
        )
