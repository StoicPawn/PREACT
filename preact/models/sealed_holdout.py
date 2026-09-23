"""Create-once sealed holdouts for PREACT predictive research.

The seal exists to prevent research iteration from repeatedly peeking at the tail of
history. Development code may use only dates before holdout_start. The hidden target
rows are committed with SHA-256 so the final evaluation can later prove it used the same
outcomes that were sealed before model/feature selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import ceil
from pathlib import Path

import pandas as pd

from .experiment_manifest import fingerprint_panel


SEAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HoldoutSeal:
    schema_version: int
    created_at: str
    holdout_start: str
    holdout_end: str
    development_end: str
    holdout_dates: int
    development_dates: int
    feature_schema_fingerprint: str
    sealed_target_commitment: str
    sealed_panel_commitment: str
    target_name: str
    horizon_days: int

    @property
    def holdout_start_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.holdout_start)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _panel_dates(features: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(features.index, pd.MultiIndex) or "date" not in features.index.names:
        raise TypeError("features must use a MultiIndex containing 'date'")
    dates = pd.to_datetime(features.index.get_level_values("date"), errors="raise")
    if dates.isna().any():
        raise ValueError("panel dates must not contain missing values")
    return pd.DatetimeIndex(dates)


def _schema_fingerprint(features: pd.DataFrame) -> str:
    material = json.dumps(
        {
            "columns": [str(column) for column in features.columns],
            "dtypes": [str(dtype) for dtype in features.dtypes],
            "index_names": list(features.index.names),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(material).hexdigest()


def _target_commitment(target: pd.Series) -> str:
    ordered = target.sort_index()
    digest = sha256()
    digest.update(
        pd.util.hash_pandas_object(ordered, index=True).to_numpy(dtype="uint64").tobytes()
    )
    return digest.hexdigest()


def create_holdout_seal(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    target_name: str,
    horizon_days: int,
    holdout_fraction: float = 0.20,
    min_holdout_dates: int = 5,
    min_development_dates: int = 20,
) -> HoldoutSeal:
    """Define the tail holdout without reporting any holdout outcome statistics."""

    if not 0.05 <= float(holdout_fraction) <= 0.50:
        raise ValueError("holdout_fraction must be between 0.05 and 0.50")
    if min_holdout_dates < 1:
        raise ValueError("min_holdout_dates must be >= 1")
    if min_development_dates < 5:
        raise ValueError("min_development_dates must be >= 5")
    if int(horizon_days) <= 0:
        raise ValueError("horizon_days must be positive")

    dates = _panel_dates(features)
    aligned = target.reindex(features.index)
    observed = aligned.notna()
    eligible_dates = pd.DatetimeIndex(dates[observed.to_numpy()].unique()).sort_values()
    if len(eligible_dates) < min_development_dates + min_holdout_dates:
        raise ValueError(
            "not enough fully observed dates to create a sealed holdout: "
            f"need at least {min_development_dates + min_holdout_dates}, "
            f"found {len(eligible_dates)}"
        )

    requested = max(min_holdout_dates, int(ceil(len(eligible_dates) * holdout_fraction)))
    max_holdout = len(eligible_dates) - min_development_dates
    holdout_dates = min(requested, max_holdout)
    if holdout_dates < min_holdout_dates:
        raise ValueError("insufficient dates after preserving the development minimum")

    holdout_start = pd.Timestamp(eligible_dates[-holdout_dates])
    development_dates = eligible_dates[eligible_dates < holdout_start]
    held_dates = eligible_dates[eligible_dates >= holdout_start]

    date_values = pd.to_datetime(features.index.get_level_values("date"))
    holdout_mask = date_values >= holdout_start
    sealed_features = features.loc[holdout_mask]
    sealed_target = aligned.loc[holdout_mask]

    if sealed_target.dropna().empty:
        raise ValueError("sealed holdout contains no observed target rows")

    return HoldoutSeal(
        schema_version=SEAL_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        holdout_start=holdout_start.isoformat(),
        holdout_end=pd.Timestamp(held_dates[-1]).isoformat(),
        development_end=pd.Timestamp(development_dates[-1]).isoformat(),
        holdout_dates=int(len(held_dates)),
        development_dates=int(len(development_dates)),
        feature_schema_fingerprint=_schema_fingerprint(features),
        sealed_target_commitment=_target_commitment(sealed_target),
        sealed_panel_commitment=fingerprint_panel(sealed_features, sealed_target),
        target_name=str(target_name),
        horizon_days=int(horizon_days),
    )


def save_holdout_seal_once(path: str | Path, seal: HoldoutSeal) -> HoldoutSeal:
    """Persist a seal exactly once; never silently move or regenerate its boundary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = load_holdout_seal(destination)
        immutable_fields = (
            "holdout_start",
            "holdout_end",
            "development_end",
            "feature_schema_fingerprint",
            "sealed_target_commitment",
            "sealed_panel_commitment",
            "target_name",
            "horizon_days",
        )
        mismatches = [
            name
            for name in immutable_fields
            if getattr(existing, name) != getattr(seal, name)
        ]
        if mismatches:
            raise RuntimeError(
                "refusing to replace an existing holdout seal; mismatched fields: "
                + ", ".join(mismatches)
            )
        return existing

    payload = json.dumps(seal.to_dict(), indent=2, sort_keys=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(payload, encoding="utf-8")
    temp.replace(destination)
    return seal


def load_holdout_seal(path: str | Path) -> HoldoutSeal:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    seal = HoldoutSeal(**raw)
    if seal.schema_version != SEAL_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported holdout seal schema {seal.schema_version}; "
            f"expected {SEAL_SCHEMA_VERSION}"
        )
    return seal


def validate_holdout_commitment(
    features: pd.DataFrame,
    target: pd.Series,
    seal: HoldoutSeal,
) -> None:
    """Fail closed if the sealed tail or feature schema changed after the seal."""

    if _schema_fingerprint(features) != seal.feature_schema_fingerprint:
        raise RuntimeError("feature schema changed after holdout sealing")

    dates = _panel_dates(features)
    mask = dates >= seal.holdout_start_timestamp
    held_features = features.loc[mask]
    held_target = target.reindex(features.index).loc[mask]

    if _target_commitment(held_target) != seal.sealed_target_commitment:
        raise RuntimeError("sealed holdout target commitment no longer matches")
    if fingerprint_panel(held_features, held_target) != seal.sealed_panel_commitment:
        raise RuntimeError("sealed holdout panel commitment no longer matches")


def development_view(
    features: pd.DataFrame,
    target: pd.Series,
    fingerprints: pd.Series,
    seal: HoldoutSeal,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return only pre-holdout rows for feature/model research."""

    dates = _panel_dates(features)
    mask = dates < seal.holdout_start_timestamp
    x = features.loc[mask].copy()
    y = target.reindex(features.index).loc[mask].copy()
    lineage = fingerprints.reindex(features.index).loc[mask].copy()

    if x.empty:
        raise ValueError("development view is empty")
    if pd.to_datetime(x.index.get_level_values("date")).max() >= seal.holdout_start_timestamp:
        raise RuntimeError("sealed holdout row leaked into development features")
    if not x.index.equals(y.index) or not x.index.equals(lineage.index):
        raise RuntimeError("development features, target and lineage are misaligned")
    return x, y, lineage


def seal_fingerprint(seal: HoldoutSeal) -> str:
    material = json.dumps(
        seal.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(material).hexdigest()


__all__ = [
    "HoldoutSeal",
    "create_holdout_seal",
    "development_view",
    "load_holdout_seal",
    "save_holdout_seal_once",
    "seal_fingerprint",
    "validate_holdout_commitment",
]
