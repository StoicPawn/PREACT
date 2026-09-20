"""Reproducible fingerprints for predictive research datasets and runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class ExperimentManifest:
    created_at: str
    dataset_fingerprint: str
    rows: int
    events: int
    feature_columns: tuple[str, ...]
    first_date: str | None
    last_date: str | None
    entities: int
    target_name: str
    horizon_days: int
    knowledge_mode: str
    metadata: Mapping[str, object]

    def to_dict(self) -> dict:
        return asdict(self)


def fingerprint_panel(features: pd.DataFrame, target: pd.Series) -> str:
    aligned = target.reindex(features.index)
    feature_hash = pd.util.hash_pandas_object(
        features.sort_index(), index=True
    ).to_numpy(dtype="uint64")
    target_hash = pd.util.hash_pandas_object(
        aligned.sort_index(), index=True
    ).to_numpy(dtype="uint64")
    digest = sha256()
    digest.update("|".join(map(str, features.columns)).encode("utf-8"))
    digest.update(feature_hash.tobytes())
    digest.update(target_hash.tobytes())
    return digest.hexdigest()


def build_manifest(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    target_name: str,
    horizon_days: int,
    knowledge_mode: str,
    metadata: Mapping[str, object] | None = None,
) -> ExperimentManifest:
    if isinstance(features.index, pd.MultiIndex) and "date" in features.index.names:
        dates = pd.to_datetime(features.index.get_level_values("date"))
        entities = len(set(features.index.get_level_values("entity_id")))
    else:
        dates = pd.to_datetime(features.index) if len(features) else pd.DatetimeIndex([])
        entities = 1 if len(features) else 0
    aligned = target.reindex(features.index)
    return ExperimentManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_fingerprint=fingerprint_panel(features, aligned),
        rows=int(len(features)),
        events=int(aligned.fillna(0).astype(int).sum()),
        feature_columns=tuple(str(c) for c in features.columns),
        first_date=dates.min().isoformat() if len(dates) else None,
        last_date=dates.max().isoformat() if len(dates) else None,
        entities=int(entities),
        target_name=target_name,
        horizon_days=int(horizon_days),
        knowledge_mode=str(knowledge_mode),
        metadata=dict(metadata or {}),
    )


def manifest_json(manifest: ExperimentManifest) -> str:
    return json.dumps(manifest.to_dict(), indent=2, sort_keys=True, default=str)
