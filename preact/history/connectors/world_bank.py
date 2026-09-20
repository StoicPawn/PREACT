"""World Bank indicator acquisition through the shared provider gateway."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from preact.data_hub.gateway import SharedProviderGateway, ProviderResponse


@dataclass(frozen=True)
class WorldBankObservation:
    country_iso3: str
    indicator: str
    year: int
    value: float | None
    retrieved_at: datetime
    snapshot_checksum: str | None
    replay_eligible_before_retrieval: bool = False


class WorldBankIndicatorConnector:
    """Fetch World Bank observations while preserving current-vintage provenance.

    The public Indicators API exposes revised historical values but does not provide
    an observation-level publication timestamp. Therefore observations from a
    current pull are not considered safe for replay cutoffs earlier than retrieval
    unless a separate historical vintage is supplied.
    """

    BASE = "https://api.worldbank.org/v2"

    def __init__(self, gateway: SharedProviderGateway) -> None:
        self.gateway = gateway

    def fetch(
        self,
        *,
        country: str,
        indicator: str,
        start_year: int,
        end_year: int,
        ttl_seconds: int = 86400,
    ) -> list[WorldBankObservation]:
        if end_year < start_year:
            raise ValueError("end_year must be >= start_year")
        url = f"{self.BASE}/country/{country}/indicator/{indicator}"
        response = self.gateway.get_json(
            source_id="world_bank",
            operation="indicator",
            url=url,
            params={
                "date": f"{int(start_year)}:{int(end_year)}",
                "format": "json",
                "per_page": "20000",
            },
            ttl_seconds=max(0, int(ttl_seconds)),
            minimum_interval_seconds=0.05,
            timeout_seconds=45.0,
        )
        payload = response.payload
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        observations: list[WorldBankObservation] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            year_raw = row.get("date")
            iso3 = str(row.get("countryiso3code") or country).strip().upper()
            try:
                year = int(year_raw)
            except (TypeError, ValueError):
                continue
            value_raw = row.get("value")
            try:
                value = float(value_raw) if value_raw is not None else None
            except (TypeError, ValueError):
                value = None
            observations.append(
                WorldBankObservation(
                    country_iso3=iso3,
                    indicator=str(row.get("indicator", {}).get("id") or indicator),
                    year=year,
                    value=value,
                    retrieved_at=response.retrieved_at,
                    snapshot_checksum=response.snapshot_checksum,
                )
            )
        return sorted(observations, key=lambda item: item.year)
