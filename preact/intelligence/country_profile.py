"""Current-vintage country profile assembly for the World Explorer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from preact.data_hub.gateway import SharedProviderGateway
from preact.history.connectors.world_bank import (
    WorldBankIndicatorConnector,
    WorldBankObservation,
)


@dataclass(frozen=True)
class IndicatorSpec:
    code: str
    label: str
    scale: float = 1.0
    suffix: str = ""


@dataclass(frozen=True)
class IndicatorSnapshot:
    code: str
    label: str
    year: int | None
    value: float | None
    display_value: float | None
    suffix: str
    retrieved_at: datetime | None
    snapshot_checksum: str | None


COUNTRY_INDICATORS: Mapping[str, IndicatorSpec] = {
    "population": IndicatorSpec("SP.POP.TOTL", "Population", 1_000_000.0, " M"),
    "gdp": IndicatorSpec("NY.GDP.MKTP.CD", "GDP", 1_000_000_000.0, " B USD"),
    "unemployment": IndicatorSpec("SL.UEM.TOTL.ZS", "Unemployment", 1.0, "%"),
    "life_expectancy": IndicatorSpec("SP.DYN.LE00.IN", "Life expectancy", 1.0, " years"),
    "internet_use": IndicatorSpec("IT.NET.USER.ZS", "Internet use", 1.0, "%"),
    "urban_population": IndicatorSpec("SP.URB.TOTL.IN.ZS", "Urban population", 1.0, "%"),
}


def latest_snapshot(
    observations: list[WorldBankObservation],
    spec: IndicatorSpec,
) -> IndicatorSnapshot:
    usable = [item for item in observations if item.value is not None]
    if not usable:
        return IndicatorSnapshot(
            code=spec.code,
            label=spec.label,
            year=None,
            value=None,
            display_value=None,
            suffix=spec.suffix,
            retrieved_at=None,
            snapshot_checksum=None,
        )
    latest = max(usable, key=lambda item: item.year)
    value = float(latest.value) if latest.value is not None else None
    return IndicatorSnapshot(
        code=spec.code,
        label=spec.label,
        year=latest.year,
        value=value,
        display_value=(value / spec.scale) if value is not None else None,
        suffix=spec.suffix,
        retrieved_at=latest.retrieved_at,
        snapshot_checksum=latest.snapshot_checksum,
    )


def fetch_current_country_profile(
    country_iso3: str,
    *,
    gateway: SharedProviderGateway,
    current_year: int | None = None,
    years_back: int = 8,
) -> dict[str, IndicatorSnapshot]:
    """Fetch current-vintage World Bank indicators with snapshot provenance.

    These observations are suitable for today's country brief. They are intentionally
    not marked replay-safe for historical cutoffs because the World Bank API exposes
    revised historical values without observation-level publication timestamps.
    """

    iso3 = str(country_iso3).strip().upper()
    if len(iso3) != 3:
        raise ValueError("country_iso3 must be an ISO-3-like code")
    if years_back < 1:
        raise ValueError("years_back must be >= 1")

    end_year = int(current_year or datetime.now(timezone.utc).year)
    start_year = end_year - years_back
    connector = WorldBankIndicatorConnector(gateway)

    result: dict[str, IndicatorSnapshot] = {}
    for key, spec in COUNTRY_INDICATORS.items():
        observations = connector.fetch(
            country=iso3,
            indicator=spec.code,
            start_year=start_year,
            end_year=end_year,
            ttl_seconds=86400,
        )
        result[key] = latest_snapshot(observations, spec)
    return result


__all__ = [
    "COUNTRY_INDICATORS",
    "IndicatorSnapshot",
    "IndicatorSpec",
    "fetch_current_country_profile",
    "latest_snapshot",
]
