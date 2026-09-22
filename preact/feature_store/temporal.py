"""Point-in-time feature materialization from the bitemporal warehouse."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from typing import Iterable

import pandas as pd

from preact.history.schema import KnowledgeMode
from preact.history.warehouse import HistoricalWarehouse


def _numeric_scalar(value_json: str | None) -> float | None:
    if value_json is None:
        return None
    try:
        value = json.loads(value_json)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _assert_point_in_time_rows(
    rows: Iterable[dict],
    *,
    cutoff: datetime,
    knowledge_mode: KnowledgeMode,
) -> None:
    """Fail closed if a feature query returns evidence from the future.

    The warehouse query is the primary temporal filter. This second boundary is
    intentionally kept in feature materialization so a future query/refactor bug
    cannot silently turn into optimistic OOS performance.
    """

    cutoff_ts = pd.Timestamp(cutoff)
    for row in rows:
        valid_from = pd.Timestamp(row["valid_from"])
        if valid_from > cutoff_ts:
            raise ValueError(
                "point-in-time feature leakage: valid_from is after prediction cutoff "
                f"({valid_from.isoformat()} > {cutoff_ts.isoformat()})"
            )
        if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN:
            known_at = pd.Timestamp(row["known_at"])
            if known_at > cutoff_ts:
                raise ValueError(
                    "point-in-time feature leakage: known_at is after prediction cutoff "
                    f"({known_at.isoformat()} > {cutoff_ts.isoformat()})"
                )


def _feature_lineage(row: dict) -> dict[str, object]:
    """Return a stable, auditable identity for the exact feature vintage used."""
    required = ("record_id", "source", "source_ref", "retrieved_at", "valid_from", "known_at")
    missing = [field for field in required if row.get(field) is None]
    if missing:
        raise ValueError("feature lineage is incomplete: missing " + ", ".join(missing))

    lineage: dict[str, object] = {
        "record_id": str(row["record_id"]),
        "source": str(row["source"]),
        "source_ref": str(row["source_ref"]),
        "dataset_version": None if row.get("dataset_version") is None else str(row["dataset_version"]),
        "valid_from": pd.Timestamp(row["valid_from"]).isoformat(),
        "known_at": pd.Timestamp(row["known_at"]).isoformat(),
        "retrieved_at": pd.Timestamp(row["retrieved_at"]).isoformat(),
    }
    canonical = json.dumps(lineage, sort_keys=True, separators=(",", ":"))
    lineage["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return lineage


def feature_snapshot_fingerprint(lineage: dict[str, dict[str, object]]) -> str:
    """Fingerprint the complete feature vintage used for one prediction row.

    The variable name is part of the canonical payload, so swapping two source
    vintages between features cannot preserve the snapshot identity.  Sorting
    makes the result independent of warehouse/dict iteration order.
    """
    canonical_rows: list[dict[str, str]] = []
    for variable, item in sorted(lineage.items()):
        fingerprint = item.get("fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError(f"feature lineage fingerprint missing or invalid for {variable}")
        canonical_rows.append({"variable": str(variable), "fingerprint": fingerprint})
    canonical = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def entity_feature_snapshot_with_lineage(
    warehouse: HistoricalWarehouse,
    *,
    entity_id: str,
    cutoff: datetime,
    variables: Iterable[str] | None = None,
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    """Materialize features plus the exact source vintage used for each value.

    Lineage is intentionally fail-closed: a research artifact claiming auditable
    point-in-time features must identify the immutable warehouse record and source
    vintage rather than merely recording the resulting numeric value.
    """
    rows = warehouse.latest_observations_as_of(
        cutoff=cutoff,
        entity_id=entity_id,
        variables=variables,
        knowledge_mode=knowledge_mode,
    )
    _assert_point_in_time_rows(rows, cutoff=cutoff, knowledge_mode=knowledge_mode)
    features: dict[str, float] = {}
    lineage: dict[str, dict[str, object]] = {}
    for row in rows:
        value = _numeric_scalar(row.get("value_json"))
        if value is not None:
            variable = str(row["variable"])
            features[variable] = value
            lineage[variable] = _feature_lineage(row)
    return features, lineage


def entity_feature_snapshot(
    warehouse: HistoricalWarehouse,
    *,
    entity_id: str,
    cutoff: datetime,
    valid_at: datetime | None = None,
    variables: Iterable[str] | None = None,
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
) -> dict[str, float]:
    rows = warehouse.latest_observations_as_of(
        cutoff=cutoff,
        entity_id=entity_id,
        variables=variables,
        knowledge_mode=knowledge_mode,
    )
    _assert_point_in_time_rows(rows, cutoff=cutoff, knowledge_mode=knowledge_mode)
    features: dict[str, float] = {}
    for row in rows:
        value = _numeric_scalar(row.get("value_json"))
        if value is not None:
            features[str(row["variable"])] = value
    return features


def entity_feature_frame(
    warehouse: HistoricalWarehouse,
    *,
    entity_id: str,
    cutoffs: Iterable[datetime],
    variables: Iterable[str] | None = None,
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for cutoff in sorted(cutoffs):
        row: dict[str, object] = {"date": pd.Timestamp(cutoff)}
        row.update(
            entity_feature_snapshot(
                warehouse,
                entity_id=entity_id,
                cutoff=cutoff,
                valid_at=cutoff,
                variables=variables,
                knowledge_mode=knowledge_mode,
            )
        )
        records.append(row)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).set_index("date").sort_index()
    return frame
