"""Configuration models for the PREACT early warning system."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""

    name: str
    endpoint: str
    update_frequency: str = "daily"
    params: Dict[str, str] | None = None
    requires_key: bool = False


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipelines."""

    name: str
    inputs: List[str]
    aggregation: str
    window_days: int


@dataclass
class ModelConfig:
    """Configuration for predictive models."""

    name: str
    target: str
    horizon_days: int
    features: List[str]
    hyperparameters: Dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class StorageConfig:
    """Configuration for file-based storage."""

    root_dir: Path
    feature_store: Path | None = None
    models_dir: Path | None = None
    logs_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.feature_store is None:
            self.feature_store = self.root_dir / "feature_store"
        if self.models_dir is None:
            self.models_dir = self.root_dir / "models"
        if self.logs_dir is None:
            self.logs_dir = self.root_dir / "logs"


@dataclass
class PREACTConfig:
    """Top-level configuration for the PREACT system."""

    data_sources: Iterable[DataSourceConfig]
    features: Iterable[FeatureConfig]
    models: Iterable[ModelConfig]
    storage: StorageConfig
    refresh_hour_utc: int = 3
    abstention_threshold: float = 0.35
    alert_threshold: float = 0.65
    backtest_years: int = 10

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the configuration."""

        return {
            "data_sources": [vars(ds) for ds in self.data_sources],
            "features": [vars(ft) for ft in self.features],
            "models": [vars(md) for md in self.models],
            "storage": {
                "root_dir": str(self.storage.root_dir),
                "feature_store": str(self.storage.feature_store),
                "models_dir": str(self.storage.models_dir),
                "logs_dir": str(self.storage.logs_dir),
            },
            "refresh_hour_utc": self.refresh_hour_utc,
            "abstention_threshold": self.abstention_threshold,
            "alert_threshold": self.alert_threshold,
            "backtest_years": self.backtest_years,
        }

