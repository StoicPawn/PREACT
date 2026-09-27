"""Conservative clustering of GDELT observations into logical events.

GDELT event identifiers are provider observations, not guaranteed one-to-one with
real-world episodes. PREACT clusters only strongly similar observations before
relationship-state estimation so repeated coverage does not multiply one shock.
"""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _pair(row: Mapping[str, Any]) -> tuple[str, str] | None:
    a = str(row.get("actor1_entity_id") or "").removeprefix("country:").upper()
    b = str(row.get("actor2_entity_id") or "").removeprefix("country:").upper()
    if len(a) != 3 or len(b) != 3 or a == b:
        return None
    return tuple(sorted((a, b)))


def _bucket_key(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    pair = _pair(row)
    if pair is None:
        return None
    root = str(row.get("event_root_code") or "").strip()
    base = str(row.get("event_base_code") or row.get("event_code") or "").strip()
    source_url = str(row.get("source_url") or "").strip()
    if source_url:
        # One article can yield several near-identical GDELT observations with
        # slightly different actors/locations. Count it as one evidence episode.
        return pair + (root, base, "url:" + source_url.casefold())
    location = _norm(row.get("action_location"))
    actor1 = _norm(row.get("actor1_name"))
    actor2 = _norm(row.get("actor2_name"))
    return pair + (root, base, location, actor1, actor2)


def _dt(value: object) -> datetime | None:
    return value if isinstance(value, datetime) else None


def _domain(url: object) -> str | None:
    text = str(url or "").strip()
    if not text:
        return None
    try:
        return (urlparse(text).hostname or "").lower() or None
    except ValueError:
        return None


def _union_list(rows: list[Mapping[str, Any]], key: str) -> list[str]:
    values: set[str] = set()
    for row in rows:
        raw = row.get(key) or []
        if isinstance(raw, str):
            raw = [raw]
        for value in raw:
            text = str(value).strip()
            if text:
                values.add(text)
    return sorted(values)


def _max_int(rows: list[Mapping[str, Any]], key: str) -> int:
    values: list[int] = []
    for row in rows:
        try:
            values.append(int(float(row.get(key) or 0)))
        except (TypeError, ValueError):
            continue
    return max(values, default=0)


def _weighted_mean(
    rows: list[Mapping[str, Any]],
    key: str,
    *,
    weight_key: str = "num_articles",
) -> float | None:
    total = 0.0
    mass = 0.0
    for row in rows:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        try:
            weight = max(1.0, float(row.get(weight_key) or 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        total += value * weight
        mass += weight
    return total / mass if mass else None


def _cluster_id(rows: list[Mapping[str, Any]]) -> str:
    """Stable episode id based on signature and first knowledge-time bucket."""
    first = rows[0]
    key = _bucket_key(first) or ()
    known_times = sorted(
        value for row in rows if (value := _dt(row.get("known_at")))
    )
    if known_times:
        first_known = known_times[0]
        bucket_hour = (first_known.hour // 6) * 6
        time_bucket = first_known.replace(
            hour=bucket_hour, minute=0, second=0, microsecond=0
        ).isoformat()
    else:
        event_time = _dt(first.get("event_time"))
        time_bucket = event_time.isoformat() if event_time else "unknown"
    material = "|".join((*key, time_bucket))
    return "gdc_" + sha256(material.encode("utf-8")).hexdigest()[:24]


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (
            _max_int([row], "mention_source_count"),
            _max_int([row], "num_sources"),
            _max_int([row], "num_articles"),
        ),
        reverse=True,
    )
    representative = dict(ranked[0])
    event_ids = sorted(
        {
            str(row.get("provider_event_id") or "").strip()
            for row in rows
            if str(row.get("provider_event_id") or "").strip()
        }
    )
    known_times = [value for row in rows if (value := _dt(row.get("known_at")))]
    event_times = [value for row in rows if (value := _dt(row.get("event_time")))]
    domains = sorted({value for row in rows if (value := _domain(row.get("source_url")))})

    representative["provider_event_id"] = _cluster_id(rows)
    representative["cluster_event_ids"] = event_ids
    representative["cluster_size"] = len(event_ids)
    if known_times:
        representative["known_at"] = max(known_times)
    if event_times:
        representative["event_time"] = min(event_times)

    for key in (
        "num_sources",
        "num_articles",
        "num_mentions",
        "corroborating_mentions",
        "mention_source_count",
        "mention_max_confidence",
    ):
        representative[key] = _max_int(rows, key)

    representative["goldstein"] = _weighted_mean(rows, "goldstein")
    representative["tone"] = _weighted_mean(rows, "tone")
    representative["mention_document_tone"] = _weighted_mean(
        rows, "mention_document_tone"
    )
    representative["themes"] = _union_list(rows, "themes")
    representative["persons"] = _union_list(rows, "persons")
    representative["organizations"] = _union_list(rows, "organizations")
    representative["source_domains"] = domains
    return representative


def cluster_event_evidence(
    rows: Iterable[Mapping[str, Any]],
    *,
    window_hours: float = 6.0,
) -> list[dict[str, Any]]:
    """Cluster only observations with the same bilateral/action signature.

    A new cluster starts when the knowledge-time gap exceeds window_hours.
    This is intentionally conservative: under-clustering is preferable to merging
    distinct geopolitical episodes.
    """
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")

    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        key = _bucket_key(row)
        if key is None or _dt(row.get("known_at")) is None:
            passthrough.append(dict(row))
            continue
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    max_gap = window_hours * 3600.0
    for key in sorted(groups):
        ordered = sorted(
            groups[key],
            key=lambda row: (
                _dt(row.get("known_at")),
                str(row.get("provider_event_id") or ""),
            ),
        )
        current: list[Mapping[str, Any]] = []
        last_known: datetime | None = None
        for row in ordered:
            known = _dt(row.get("known_at"))
            assert known is not None
            if (
                current
                and last_known is not None
                and (known - last_known).total_seconds() > max_gap
            ):
                output.append(_aggregate(current))
                current = []
            current.append(row)
            last_known = known
        if current:
            output.append(_aggregate(current))

    for row in passthrough:
        event_id = str(row.get("provider_event_id") or "").strip()
        row.setdefault("cluster_event_ids", [event_id] if event_id else [])
        row.setdefault("cluster_size", 1 if event_id else 0)
        output.append(row)

    return sorted(
        output,
        key=lambda row: (
            _dt(row.get("event_time")) or _dt(row.get("known_at")),
            str(row.get("provider_event_id") or ""),
        ),
    )


__all__ = ["cluster_event_evidence"]
