"""Orchestration utilities for multi-layer data ingestion."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd

from ..config import DataSourceConfig, PREACTConfig
from .sources import DataSource, IngestionResult, build_sources, fetch_all

LOGGER = logging.getLogger(__name__)


@dataclass
class IngestionArtifacts:
    """Container for bronze, silver and gold ingestion layers."""

    bronze: Dict[str, IngestionResult]
    silver: Dict[str, pd.DataFrame]
    gold: Dict[str, pd.DataFrame]


class DataIngestionOrchestrator:
    """Build bronze, silver and gold datasets from configured data sources."""

    def __init__(
        self,
        config: PREACTConfig | Iterable[DataSourceConfig],
        lookback_days: int = 30,
        storage_root: Path | None = None,
    ) -> None:
        if isinstance(config, PREACTConfig):
            self.source_configs = list(config.data_sources)
            self.storage_root = storage_root or config.storage.root_dir
        else:
            self.source_configs = list(config)
            self.storage_root = storage_root
        self.lookback_days = lookback_days
        self.sources = build_sources(self.source_configs)

    def run(
        self,
        lookback_days: Optional[int] = None,
        end: Optional[datetime] = None,
        persist: Optional[bool] = None,
    ) -> IngestionArtifacts:
        """Execute the ingestion pipeline and optionally persist outputs."""

        effective_lookback = lookback_days if lookback_days is not None else self.lookback_days
        bronze = fetch_all(self.sources, lookback_days=effective_lookback, end=end)
        silver = self._build_silver_layer(bronze)
        gold = self._build_gold_layer(silver)
        artifacts = IngestionArtifacts(bronze=bronze, silver=silver, gold=gold)

        should_persist = (persist if persist is not None else self.storage_root is not None)
        if should_persist and self.storage_root is not None:
            LOGGER.info("Persisting ingestion artifacts to %s", self.storage_root)
            self._persist_artifacts(artifacts)

        return artifacts

    def _build_silver_layer(self, bronze: Mapping[str, IngestionResult]) -> Dict[str, pd.DataFrame]:
        silver: Dict[str, pd.DataFrame] = {}
        for name, result in bronze.items():
            frame = result.data.copy()
            if frame.empty:
                silver[name] = frame
                continue
            tidy = self._standardise_columns(frame)
            tidy = self._parse_datetime_columns(tidy)
            tidy = tidy.drop_duplicates()
            date_column = self._find_date_column(tidy)
            if date_column:
                tidy = tidy.dropna(subset=[date_column])
                tidy = tidy.sort_values(date_column)
            tidy = tidy.reset_index(drop=True)
            tidy = tidy.apply(
                lambda col: pd.to_numeric(col, errors="ignore")
                if col.name != date_column
                else col
            )
            silver[name] = tidy
        return silver

    def _build_gold_layer(self, silver: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        gold: Dict[str, pd.DataFrame] = {}
        combined_frames: list[pd.DataFrame] = []
        for name, frame in silver.items():
            if frame.empty:
                gold[name] = frame
                continue
            tidy = frame.copy()
            date_column = self._find_date_column(tidy)
            grouping = []
            if date_column:
                tidy[date_column] = pd.to_datetime(tidy[date_column], errors="coerce")
                tidy = tidy.dropna(subset=[date_column])
                grouping.append(date_column)
            if "country" in tidy.columns:
                grouping.append("country")
            numeric_cols = tidy.select_dtypes(include="number").columns.tolist()
            if numeric_cols and grouping:
                aggregated = (
                    tidy.groupby(grouping, dropna=False)[numeric_cols]
                    .mean()
                    .reset_index()
                )
            elif numeric_cols:
                aggregated = pd.DataFrame({col: [tidy[col].mean()] for col in numeric_cols})
            elif grouping:
                aggregated = tidy[grouping].drop_duplicates()
            else:
                aggregated = tidy
            if grouping:
                aggregated = aggregated.sort_values(grouping)
            gold[name] = aggregated

            if date_column:
                combined = aggregated.copy()
                if date_column != "date":
                    combined = combined.rename(columns={date_column: "date"})
                if "date" in combined.columns:
                    combined["source"] = name
                    combined_frames.append(combined)
        if combined_frames:
            combined_df = pd.concat(combined_frames, ignore_index=True, sort=False)
            combined_df["date"] = pd.to_datetime(combined_df["date"], errors="coerce")
            combined_df = combined_df.dropna(subset=["date"])
            combined_df = combined_df.sort_values(["date", "source"])
        else:
            combined_df = pd.DataFrame()
        gold["combined"] = combined_df
        return gold

    def _persist_artifacts(self, artifacts: IngestionArtifacts) -> None:
        assert self.storage_root is not None
        self._persist_bronze(artifacts.bronze)
        self._persist_frame_layer("silver", artifacts.silver)
        self._persist_frame_layer("gold", artifacts.gold)

    def _persist_bronze(self, bronze: Mapping[str, IngestionResult]) -> None:
        root = self.storage_root / "bronze"
        root.mkdir(parents=True, exist_ok=True)
        for name, result in bronze.items():
            path = root / f"{self._slugify(name)}.parquet"
            result.to_parquet(path)

    def _persist_frame_layer(self, layer: str, frames: Mapping[str, pd.DataFrame]) -> None:
        root = self.storage_root / layer
        root.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            path = root / f"{self._slugify(name)}.parquet"
            frame.to_parquet(path, index=False)

    @staticmethod
    def _slugify(name: str) -> str:
        return name.lower().replace(" ", "_")

    @staticmethod
    def _standardise_columns(frame: pd.DataFrame) -> pd.DataFrame:
        tidy = frame.copy()
        rename_map: Dict[str, str] = {}
        seen: Dict[str, int] = {}
        for column in tidy.columns:
            slug = column.strip().lower().replace(" ", "_")
            count = seen.get(slug, 0)
            if count:
                slug = f"{slug}_{count}"
            rename_map[column] = slug
            seen[slug] = count + 1
        tidy = tidy.rename(columns=rename_map)
        tidy = tidy.loc[:, ~tidy.columns.duplicated()]
        return tidy

    @staticmethod
    def _parse_datetime_columns(frame: pd.DataFrame) -> pd.DataFrame:
        tidy = frame.copy()
        for column in tidy.columns:
            if "date" in column:
                tidy[column] = pd.to_datetime(tidy[column], errors="coerce")
        return tidy

    @staticmethod
    def _find_date_column(frame: pd.DataFrame) -> Optional[str]:
        for candidate in ("event_date", "date", "timestamp"):
            if candidate in frame.columns:
                return candidate
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                return column
        return None
