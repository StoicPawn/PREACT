"""Evidence-aware country relationship signals for the World Explorer.

The layer deliberately separates documented structural relationships from dynamic
news/event signals. GDELT-derived scores are short-horizon interaction signals, not
claims that two states are formally allied or hostile.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, tanh
from typing import Iterable, Mapping

import pandas as pd


@dataclass(frozen=True, slots=True)
class RelationshipSignal:
    focal_iso3: str
    counterpart_iso3: str
    score: float
    confidence: float
    status: str
    event_count: int
    last_seen: pd.Timestamp | None
    structural_alliance: bool = False


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _dynamic_edge_score(row: pd.Series) -> float:
    """Map GDELT cooperative/conflict intensity and media tone to [-1, 1]."""

    goldstein = pd.to_numeric(row.get("avg_goldstein"), errors="coerce")
    tone = pd.to_numeric(row.get("avg_tone"), errors="coerce")
    components: list[tuple[float, float]] = []
    if pd.notna(goldstein):
        components.append((0.8, _clip(float(goldstein) / 10.0)))
    if pd.notna(tone):
        components.append((0.2, tanh(float(tone) / 10.0)))
    if not components:
        return 0.0
    weight = sum(w for w, _ in components)
    return _clip(sum(w * value for w, value in components) / weight)


def _status(score: float, confidence: float, structural_alliance: bool) -> str:
    if structural_alliance:
        return "documented_alliance"
    if confidence < 0.15:
        return "insufficient_evidence"
    if score <= -0.60:
        return "conflict"
    if score <= -0.20:
        return "tension"
    if score >= 0.60:
        return "strong_cooperation"
    if score >= 0.20:
        return "affinity"
    return "mixed"


def build_relationship_layer(
    edges: pd.DataFrame,
    *,
    focal_iso3: str,
    as_of: str | pd.Timestamp,
    structural_allies: Iterable[str] = (),
    half_life_days: float = 30.0,
    min_events: int = 1,
) -> dict[str, RelationshipSignal]:
    """Aggregate directional GDELT edges into focal-country relationship signals.

    This is point-in-time safe: an edge whose last_seen is later than as_of is
    rejected because a pre-aggregated edge containing future events cannot be
    repaired without the underlying event rows.
    """

    focal = str(focal_iso3).strip().upper()
    if len(focal) != 3:
        raise ValueError("focal_iso3 must be an ISO-3-like code")
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if min_events < 1:
        raise ValueError("min_events must be >= 1")

    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)

    if edges.empty:
        dynamic: dict[str, list[tuple[float, float, int, pd.Timestamp | None]]] = {}
    else:
        required = {"source", "target", "events"}
        missing = required.difference(edges.columns)
        if missing:
            raise ValueError(f"relationship edges missing required columns: {sorted(missing)}")

        frame = edges.copy()
        frame["source"] = frame["source"].astype(str).str.strip().str.upper()
        frame["target"] = frame["target"].astype(str).str.strip().str.upper()
        frame["events"] = pd.to_numeric(frame["events"], errors="coerce").fillna(0).astype(int)
        if "last_seen" in frame.columns:
            frame["last_seen"] = pd.to_datetime(
                frame["last_seen"], errors="coerce", utc=True
            ).dt.tz_localize(None)
            future = frame["last_seen"].notna() & (frame["last_seen"] > cutoff)
            if future.any():
                raise ValueError("relationship edges contain observations after as_of")
        else:
            frame["last_seen"] = pd.NaT

        frame = frame[
            ((frame["source"] == focal) | (frame["target"] == focal))
            & (frame["source"] != frame["target"])
            & (frame["events"] >= min_events)
        ]

        dynamic = {}
        for _, row in frame.iterrows():
            counterpart = row["target"] if row["source"] == focal else row["source"]
            if len(counterpart) != 3:
                continue
            last_seen = row["last_seen"] if pd.notna(row["last_seen"]) else None
            age_days = (
                0.0
                if last_seen is None
                else max(0.0, (cutoff - last_seen).total_seconds() / 86400.0)
            )
            recency = exp(-log(2.0) * age_days / half_life_days)
            event_count = int(row["events"])
            evidence_strength = 1.0 - exp(-event_count / 10.0)
            confidence = _clip(recency * evidence_strength, 0.0, 1.0)
            score = _dynamic_edge_score(row)
            dynamic.setdefault(counterpart, []).append(
                (score, confidence, event_count, last_seen)
            )

    allies = {str(code).strip().upper() for code in structural_allies if str(code).strip()}
    counterparts = set(dynamic).union(allies)
    result: dict[str, RelationshipSignal] = {}
    for counterpart in counterparts:
        observations = dynamic.get(counterpart, [])
        total_events = sum(item[2] for item in observations)
        if observations:
            total_weight = sum(max(item[1], 1e-12) for item in observations)
            score = (
                sum(item[0] * max(item[1], 1e-12) for item in observations)
                / total_weight
            )
            complement = 1.0
            for _, item_confidence, _, _ in observations:
                complement *= 1.0 - item_confidence
            confidence = 1.0 - complement
            dated = [item[3] for item in observations if item[3] is not None]
            last_seen = max(dated) if dated else None
        else:
            score = 0.0
            confidence = 0.0
            last_seen = None

        structural = counterpart in allies
        result[counterpart] = RelationshipSignal(
            focal_iso3=focal,
            counterpart_iso3=counterpart,
            score=_clip(score),
            confidence=_clip(confidence, 0.0, 1.0),
            status=_status(score, confidence, structural),
            event_count=total_events,
            last_seen=last_seen,
            structural_alliance=structural,
        )

    return result


def relationship_rows(signals: Mapping[str, RelationshipSignal]) -> list[dict[str, object]]:
    """Return stable serializable rows for dashboard/report consumers."""

    rows: list[dict[str, object]] = []
    for iso3 in sorted(signals):
        signal = signals[iso3]
        rows.append(
            {
                "iso3": iso3,
                "score": signal.score,
                "confidence": signal.confidence,
                "status": signal.status,
                "event_count": signal.event_count,
                "last_seen": signal.last_seen,
                "structural_alliance": signal.structural_alliance,
            }
        )
    return rows


__all__ = ["RelationshipSignal", "build_relationship_layer", "relationship_rows"]
