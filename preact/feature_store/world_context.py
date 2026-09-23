"""Sparse point-in-time world-context features for geopolitical forecasting.

The encoder is designed for the combinatorial nature of geopolitics: a focal country's
risk can depend on events involving its allies, neighbours, rivals and the wider system.
Instead of creating one feature for every country pair, PREACT compresses the admissible
relation graph into system, first-hop and second-hop pressure features at each cutoff.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from typing import Iterable

import pandas as pd

from preact.history.graph_store import HistoricalGraphStore
from preact.history.schema import KnowledgeMode


def _canonical_relation(row: dict) -> dict[str, str | None]:
    return {
        "relation_id": str(row.get("relation_id") or ""),
        "relation_type": str(row.get("relation_type") or ""),
        "subject_entity_id": str(row.get("subject_entity_id") or ""),
        "object_entity_id": str(row.get("object_entity_id") or ""),
        "valid_from": (
            pd.Timestamp(row["valid_from"]).isoformat()
            if row.get("valid_from") is not None
            else None
        ),
        "valid_to": (
            pd.Timestamp(row["valid_to"]).isoformat()
            if row.get("valid_to") is not None
            else None
        ),
        "known_at": (
            pd.Timestamp(row["known_at"]).isoformat()
            if row.get("known_at") is not None
            else None
        ),
        "source": str(row.get("source") or ""),
        "source_ref": str(row.get("source_ref") or ""),
        "retrieved_at": (
            pd.Timestamp(row["retrieved_at"]).isoformat()
            if row.get("retrieved_at") is not None
            else None
        ),
        "dataset_version": (
            None if row.get("dataset_version") is None else str(row["dataset_version"])
        ),
        "attributes_json": str(row.get("attributes_json") or ""),
    }


def _fingerprint_rows(rows: Iterable[dict]) -> str:
    canonical = sorted(
        (_canonical_relation(row) for row in rows),
        key=lambda item: (
            item["relation_id"] or "",
            item["valid_from"] or "",
            item["known_at"] or "",
        ),
    )
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _relation_signal(row: dict) -> tuple[float, float, float]:
    """Return cooperation, conflict and log-volume contributions for a relation row."""

    raw = row.get("attributes_json")
    if not raw:
        return 0.0, 0.0, 0.0
    try:
        attrs = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return 0.0, 0.0, 0.0
    if not isinstance(attrs, dict):
        return 0.0, 0.0, 0.0

    goldstein = pd.to_numeric(attrs.get("goldstein_scale"), errors="coerce")
    articles = pd.to_numeric(attrs.get("num_articles"), errors="coerce")
    mentions = pd.to_numeric(attrs.get("num_mentions"), errors="coerce")
    if pd.isna(articles):
        articles = mentions
    volume = math.log1p(max(0.0, 0.0 if pd.isna(articles) else float(articles)))
    if volume <= 0:
        volume = math.log(2.0)

    if pd.notna(goldstein):
        value = float(goldstein)
        if value > 0:
            return min(1.0, value / 10.0) * volume, 0.0, volume
        if value < 0:
            return 0.0, min(1.0, -value / 10.0) * volume, volume

    quad = str(attrs.get("quad_class") or "").strip()
    if quad in {"1", "2"}:
        return 0.25 * volume, 0.0, volume
    if quad in {"3", "4"}:
        return 0.0, 0.25 * volume, volume
    return 0.0, 0.0, volume


def _pressure(rows: Iterable[dict]) -> tuple[float, float, float]:
    cooperation = 0.0
    conflict = 0.0
    volume = 0.0
    for row in rows:
        coop, conf, row_volume = _relation_signal(row)
        cooperation += coop
        conflict += conf
        volume += row_volume
    return cooperation, conflict, volume


@dataclass(frozen=True)
class WorldContextSnapshot:
    cutoff: datetime
    windows_days: tuple[int, ...]
    active_rows: tuple[dict, ...]
    recent_rows: tuple[dict, ...]
    evidence_fingerprint: str
    max_hops: int = 3

    def features_for(self, entity_id: str) -> dict[str, float]:
        """Encode global, direct-neighbour and second-hop relation pressure."""

        entity = str(entity_id)
        active = list(self.active_rows)
        recent = list(self.recent_rows)

        active_types = Counter(str(row["relation_type"]) for row in active)
        neighbors: set[str] = set()
        focal_active = 0
        for row in active:
            subject = str(row["subject_entity_id"])
            object_ = str(row["object_entity_id"])
            if subject == entity:
                neighbors.add(object_)
                focal_active += 1
            elif object_ == entity:
                neighbors.add(subject)
                focal_active += 1
        neighbors.discard(entity)

        features: dict[str, float] = {
            "world_context:system_active_relations": float(len(active)),
            "world_context:system_active_relation_types": float(len(active_types)),
            "world_context:focal_active_relations": float(focal_active),
            "world_context:focal_active_neighbors": float(len(neighbors)),
        }

        # Stable/relation-state edges define the channels through which external
        # event pressure can propagate. One-day GDELT event edges are excluded
        # from this adjacency so the same news event cannot create its own path.
        adjacency: dict[str, set[str]] = defaultdict(set)
        for relation in active:
            relation_type = str(relation.get("relation_type") or "")
            if relation_type.startswith("gdelt_"):
                continue
            subject = str(relation.get("subject_entity_id") or "")
            object_ = str(relation.get("object_entity_id") or "")
            if not subject or not object_ or subject == object_:
                continue
            adjacency[subject].add(object_)
            adjacency[object_].add(subject)
        for relation_type, count in active_types.items():
            features[f"world_context:system_active:{relation_type}"] = float(count)

        for window in self.windows_days:
            start = pd.Timestamp(self.cutoff) - pd.Timedelta(days=int(window))
            rows = [
                row
                for row in recent
                if pd.Timestamp(row["valid_from"]) > start
                and pd.Timestamp(row["valid_from"]) <= pd.Timestamp(self.cutoff)
            ]
            system_by_type = Counter(str(row["relation_type"]) for row in rows)
            focal_rows: list[dict] = []
            neighbor_external_rows: list[dict] = []
            second_hop: set[str] = set()

            for row in rows:
                subject = str(row["subject_entity_id"])
                object_ = str(row["object_entity_id"])
                endpoints = {subject, object_}
                if entity in endpoints:
                    focal_rows.append(row)
                    continue
                touched_neighbors = endpoints.intersection(neighbors)
                if touched_neighbors:
                    neighbor_external_rows.append(row)
                    second_hop.update(endpoints.difference(neighbors).difference({entity}))

            features[f"world_context:system_recent_{window}d:total"] = float(len(rows))
            features[f"world_context:focal_recent_{window}d:total"] = float(len(focal_rows))
            features[f"world_context:neighbor_recent_{window}d:total"] = float(
                len(neighbor_external_rows)
            )
            features[f"world_context:second_hop_entities_{window}d"] = float(
                len(second_hop)
            )
            features[f"world_context:neighbor_share_{window}d"] = float(
                len(neighbor_external_rows) / len(rows) if rows else 0.0
            )

            for scope, selected in (
                ("system", rows),
                ("focal", focal_rows),
                ("neighbor", neighbor_external_rows),
            ):
                cooperation, conflict, volume = _pressure(selected)
                prefix = f"world_context:{scope}_recent_{window}d"
                features[f"{prefix}:cooperation_pressure"] = float(cooperation)
                features[f"{prefix}:conflict_pressure"] = float(conflict)
                features[f"{prefix}:net_pressure"] = float(cooperation - conflict)
                features[f"{prefix}:log_volume"] = float(volume)

            # Multi-hop message passing. Recent GDELT interaction pressure is first
            # attached to its endpoint countries, then aggregated by exact graph
            # distance from the focal entity over the structural relation graph.
            node_pressure: dict[str, list[float]] = defaultdict(
                lambda: [0.0, 0.0, 0.0]
            )
            for relation in rows:
                if not str(relation.get("relation_type") or "").startswith("gdelt_"):
                    continue
                cooperation, conflict, volume = _relation_signal(relation)
                for endpoint in (
                    str(relation.get("subject_entity_id") or ""),
                    str(relation.get("object_entity_id") or ""),
                ):
                    if not endpoint:
                        continue
                    node_pressure[endpoint][0] += cooperation
                    node_pressure[endpoint][1] += conflict
                    node_pressure[endpoint][2] += volume

            visited = {entity}
            frontier = {entity}
            for hop in range(1, max(1, int(self.max_hops)) + 1):
                next_frontier: set[str] = set()
                for node in frontier:
                    next_frontier.update(adjacency.get(node, set()))
                next_frontier.difference_update(visited)
                visited.update(next_frontier)
                prefix = f"world_context:hop{hop}_recent_{window}d"
                features[f"{prefix}:entities"] = float(len(next_frontier))
                features[f"{prefix}:cooperation_pressure"] = float(
                    sum(node_pressure[node][0] for node in next_frontier)
                )
                features[f"{prefix}:conflict_pressure"] = float(
                    sum(node_pressure[node][1] for node in next_frontier)
                )
                features[f"{prefix}:log_volume"] = float(
                    sum(node_pressure[node][2] for node in next_frontier)
                )
                frontier = next_frontier
                if not frontier:
                    # Preserve a stable feature schema up to max_hops.
                    for remaining in range(hop + 1, max(1, int(self.max_hops)) + 1):
                        remaining_prefix = (
                            f"world_context:hop{remaining}_recent_{window}d"
                        )
                        features[f"{remaining_prefix}:entities"] = 0.0
                        features[f"{remaining_prefix}:cooperation_pressure"] = 0.0
                        features[f"{remaining_prefix}:conflict_pressure"] = 0.0
                        features[f"{remaining_prefix}:log_volume"] = 0.0
                    break

            neighbor_by_type = Counter(
                str(row["relation_type"]) for row in neighbor_external_rows
            )
            focal_by_type = Counter(str(row["relation_type"]) for row in focal_rows)
            for relation_type, count in system_by_type.items():
                features[
                    f"world_context:system_recent_{window}d:{relation_type}"
                ] = float(count)
                features[
                    f"world_context:focal_recent_{window}d:{relation_type}"
                ] = float(focal_by_type.get(relation_type, 0))
                features[
                    f"world_context:neighbor_recent_{window}d:{relation_type}"
                ] = float(neighbor_by_type.get(relation_type, 0))

        for half_life in (90, 365):
            decay = math.log(2.0) / float(half_life)
            global_intensity = 0.0
            neighbor_intensity = 0.0
            global_conflict = 0.0
            neighbor_conflict = 0.0
            global_cooperation = 0.0
            neighbor_cooperation = 0.0
            for row in recent:
                age_days = max(
                    0.0,
                    (
                        pd.Timestamp(self.cutoff) - pd.Timestamp(row["valid_from"])
                    ).total_seconds()
                    / 86400.0,
                )
                weight = math.exp(-decay * age_days)
                global_intensity += weight
                cooperation, conflict, _ = _relation_signal(row)
                global_cooperation += weight * cooperation
                global_conflict += weight * conflict
                endpoints = {
                    str(row["subject_entity_id"]),
                    str(row["object_entity_id"]),
                }
                if entity not in endpoints and endpoints.intersection(neighbors):
                    neighbor_intensity += weight
                    neighbor_cooperation += weight * cooperation
                    neighbor_conflict += weight * conflict
            features[
                f"world_context:system_decay_{half_life}d"
            ] = float(global_intensity)
            features[
                f"world_context:neighbor_decay_{half_life}d"
            ] = float(neighbor_intensity)
            features[
                f"world_context:system_decay_{half_life}d:cooperation_pressure"
            ] = float(global_cooperation)
            features[
                f"world_context:system_decay_{half_life}d:conflict_pressure"
            ] = float(global_conflict)
            features[
                f"world_context:neighbor_decay_{half_life}d:cooperation_pressure"
            ] = float(neighbor_cooperation)
            features[
                f"world_context:neighbor_decay_{half_life}d:conflict_pressure"
            ] = float(neighbor_conflict)

        return features


def build_world_context_snapshot(
    graph: HistoricalGraphStore,
    *,
    cutoff: datetime,
    windows_days: Iterable[int] = (90, 365, 1825),
    knowledge_mode: KnowledgeMode = KnowledgeMode.STRICT_AS_KNOWN,
    max_hops: int = 3,
) -> WorldContextSnapshot:
    """Materialize one auditable system snapshot, reusable for every entity."""

    windows = tuple(sorted({max(1, int(days)) for days in windows_days}))
    if not windows:
        raise ValueError("windows_days must contain at least one positive window")
    if int(max_hops) < 1 or int(max_hops) > 5:
        raise ValueError("max_hops must be between 1 and 5")

    active = graph.as_of(
        cutoff=cutoff,
        valid_at=cutoff,
        knowledge_mode=knowledge_mode,
    )

    start = cutoff - timedelta(days=max(windows))
    clauses = ["valid_from > ?", "valid_from <= ?"]
    params: list[object] = [start, cutoff]
    if knowledge_mode is KnowledgeMode.STRICT_AS_KNOWN:
        clauses.insert(0, "known_at <= ?")
        params.insert(0, cutoff)

    with graph.connect() as conn:
        cursor = conn.execute(
            "SELECT * FROM historical_relations WHERE "
            + " AND ".join(clauses)
            + " ORDER BY valid_from, relation_id",
            params,
        )
        columns = [item[0] for item in cursor.description]
        recent = [dict(zip(columns, row)) for row in cursor.fetchall()]

    evidence: dict[str, dict] = {}
    for row in [*active, *recent]:
        relation_id = str(row.get("relation_id") or "")
        identity = relation_id or json.dumps(
            _canonical_relation(row), sort_keys=True, separators=(",", ":")
        )
        evidence[identity] = row

    return WorldContextSnapshot(
        cutoff=cutoff,
        windows_days=windows,
        active_rows=tuple(active),
        recent_rows=tuple(recent),
        evidence_fingerprint=_fingerprint_rows(evidence.values()),
        max_hops=int(max_hops),
    )


def contextual_feature_fingerprint(
    point_feature_fingerprint: str,
    world_context_fingerprint: str,
) -> str:
    """Bind entity-local feature vintage to the exact system relation snapshot."""

    for name, value in (
        ("point_feature_fingerprint", point_feature_fingerprint),
        ("world_context_fingerprint", world_context_fingerprint),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{name} must be a SHA-256 hex digest")
    material = (
        f"point={point_feature_fingerprint}|world={world_context_fingerprint}"
    ).encode("utf-8")
    return sha256(material).hexdigest()


__all__ = [
    "WorldContextSnapshot",
    "build_world_context_snapshot",
    "contextual_feature_fingerprint",
]
