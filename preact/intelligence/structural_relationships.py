"""Documented structural relationship overlay for the World Explorer.

Structural relations come from explicit historical relation records, not from media tone.
The loader is point-in-time strict: a source mapping snapshot must already have been known
by the requested knowledge cutoff, and only relations active at the requested world time
are returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd
from preact.history.connectors.cow import COWStateSystemConnector
from preact.history.entity_crosswalk import country_name_to_iso3
from preact.history.graph_store import HistoricalGraphStore
from preact.history.schema import KnowledgeMode
from preact.history.snapshot_store import SnapshotMetadata, SourceSnapshotStore


@dataclass(frozen=True)
class StructuralRelationshipBatch:
    focal_iso3: str
    allies: tuple[str, ...]
    evidence: pd.DataFrame
    mapping_snapshot_checksum: str | None
    mapping_retrieved_at: datetime | None
    status: str


def _utc(value: datetime | pd.Timestamp | None) -> datetime:
    stamp = pd.Timestamp(value if value is not None else datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def _latest_cow_state_snapshot(
    store: SourceSnapshotStore,
    *,
    knowledge_cutoff: datetime,
) -> SnapshotMetadata | None:
    candidates = [
        item
        for item in store.iter_metadata(source_id="cow")
        if item.source_release == "State System Membership v2024"
        and item.retrieved_at <= knowledge_cutoff
    ]
    return candidates[-1] if candidates else None


def _cow_ccode_map(
    store: SourceSnapshotStore,
    *,
    valid_at: datetime,
    knowledge_cutoff: datetime,
) -> tuple[dict[str, str], SnapshotMetadata | None]:
    snapshot = _latest_cow_state_snapshot(store, knowledge_cutoff=knowledge_cutoff)
    if snapshot is None:
        return {}, None

    rows = COWStateSystemConnector.parse_rows(store.read_payload(snapshot))
    entities = COWStateSystemConnector.to_entities(rows)
    mapping: dict[str, str] = {}
    for entity in entities:
        if not entity.is_valid_at(valid_at):
            continue
        iso3 = country_name_to_iso3(entity.name)
        if iso3 is None:
            continue
        for code in entity.codes:
            namespace, value = code.normalized()
            if namespace == "cow_ccode":
                mapping[value] = iso3
    return mapping, snapshot


def _resolve_entity_id(entity_id: str, cow_map: Mapping[str, str]) -> str | None:
    raw = str(entity_id or "").strip()
    if raw.startswith("iso3:"):
        candidate = raw.split(":", 1)[1].upper()
        return candidate if len(candidate) == 3 else None
    if raw.startswith("cow_ccode:"):
        code = raw.split(":", 1)[1].strip().upper()
        return cow_map.get(code)
    return None


def load_documented_allies(
    root: str | Path,
    *,
    graph_path: str | Path = "data/history/preact_graph.duckdb",
    focal_iso3: str,
    valid_at: datetime | pd.Timestamp | None = None,
    knowledge_cutoff: datetime | pd.Timestamp | None = None,
) -> StructuralRelationshipBatch:
    """Load active formal-alliance relations with strict valid/knowledge time semantics."""

    focal = str(focal_iso3).strip().upper()
    if len(focal) != 3:
        raise ValueError("focal_iso3 must be an ISO-3-like code")

    world_time = _utc(valid_at)
    cutoff = _utc(knowledge_cutoff)
    graph_file = Path(graph_path)
    if not graph_file.exists():
        return StructuralRelationshipBatch(
            focal_iso3=focal,
            allies=(),
            evidence=pd.DataFrame(),
            mapping_snapshot_checksum=None,
            mapping_retrieved_at=None,
            status="graph_unavailable",
        )

    store = SourceSnapshotStore(Path(root) / "snapshots")
    cow_map, mapping_snapshot = _cow_ccode_map(
        store,
        valid_at=world_time,
        knowledge_cutoff=cutoff,
    )

    graph = HistoricalGraphStore(graph_file)
    relations = graph.as_of(
        cutoff=cutoff,
        valid_at=world_time,
        relation_type="formal_alliance",
        knowledge_mode=KnowledgeMode.STRICT_AS_KNOWN,
    )

    records: list[dict[str, object]] = []
    allies: set[str] = set()
    for relation in relations:
        subject = _resolve_entity_id(relation["subject_entity_id"], cow_map)
        object_ = _resolve_entity_id(relation["object_entity_id"], cow_map)
        if subject is None or object_ is None:
            continue
        if focal not in {subject, object_}:
            continue
        counterpart = object_ if subject == focal else subject
        if counterpart == focal:
            continue
        allies.add(counterpart)
        records.append(
            {
                "counterpart_iso3": counterpart,
                "relation_type": relation["relation_type"],
                "source": relation["source"],
                "source_ref": relation["source_ref"],
                "dataset_version": relation["dataset_version"],
                "valid_from": relation["valid_from"],
                "valid_to": relation["valid_to"],
                "known_at": relation["known_at"],
                "retrieved_at": relation["retrieved_at"],
            }
        )

    evidence = pd.DataFrame.from_records(records)
    status = "ok" if records else (
        "mapping_unavailable"
        if mapping_snapshot is None and any(
            str(item.get("subject_entity_id", "")).startswith("cow_ccode:")
            or str(item.get("object_entity_id", "")).startswith("cow_ccode:")
            for item in relations
        )
        else "no_active_documented_alliances"
    )
    return StructuralRelationshipBatch(
        focal_iso3=focal,
        allies=tuple(sorted(allies)),
        evidence=evidence,
        mapping_snapshot_checksum=(
            mapping_snapshot.checksum_sha256 if mapping_snapshot else None
        ),
        mapping_retrieved_at=(
            mapping_snapshot.retrieved_at if mapping_snapshot else None
        ),
        status=status,
    )


__all__ = ["StructuralRelationshipBatch", "load_documented_allies"]
