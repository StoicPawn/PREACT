"""Current NATO structural-alliance evidence from shared official snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
import re
from pathlib import Path

import pycountry

from preact.history.snapshot_store import SnapshotMetadata, SourceSnapshotStore


_MEMBER_RE = re.compile(
    r'<p\s+class="cie-country__name"\s+data-title="([^"]+)"[^>]*>.*?</p>'
    r'\s*<p\s+class="cie-country__yearEntry"[^>]*>\s*(\d{4})\s*</p>',
    re.IGNORECASE | re.DOTALL,
)

_NAME_ALIASES = {
    "THE NETHERLANDS": "Netherlands",
    "TÜRKIYE": "Türkiye",
    "UNITED STATES": "United States",
    "UNITED KINGDOM": "United Kingdom",
    "CZECHIA": "Czechia",
    "NORTH MACEDONIA": "North Macedonia",
}


@dataclass(frozen=True)
class NATOMember:
    name: str
    iso3: str
    joined_year: int


@dataclass(frozen=True)
class NATOMembershipSnapshot:
    members: tuple[NATOMember, ...]
    snapshot: SnapshotMetadata


def _iso3(name: str) -> str | None:
    candidate = _NAME_ALIASES.get(name.strip().upper(), name.strip())
    try:
        return pycountry.countries.lookup(candidate).alpha_3
    except LookupError:
        return None


def parse_nato_members(payload: bytes) -> tuple[NATOMember, ...]:
    text = payload.decode("utf-8", errors="replace")
    members: dict[str, NATOMember] = {}
    for raw_name, raw_year in _MEMBER_RE.findall(text):
        name = html.unescape(re.sub(r"<[^>]+>", "", raw_name)).strip()
        iso3 = _iso3(name)
        if iso3 is None:
            continue
        members[iso3] = NATOMember(
            name=name,
            iso3=iso3,
            joined_year=int(raw_year),
        )

    # Fail closed if NATO changes its HTML structure or the page is incomplete.
    if not 25 <= len(members) <= 40:
        raise ValueError(
            f"NATO membership parser produced implausible count: {len(members)}"
        )
    return tuple(sorted(members.values(), key=lambda item: item.iso3))


def latest_nato_membership(
    root: str | Path,
    *,
    knowledge_cutoff: datetime,
) -> NATOMembershipSnapshot | None:
    store = SourceSnapshotStore(Path(root) / "snapshots")
    candidates = [
        item
        for item in store.iter_metadata(source_id="nato")
        if item.operation == "current_members"
        and item.retrieved_at <= knowledge_cutoff
    ]
    if not candidates:
        return None
    snapshot = candidates[-1]
    return NATOMembershipSnapshot(
        members=parse_nato_members(store.read_payload(snapshot)),
        snapshot=snapshot,
    )


def current_nato_alliance_pairs(
    root: str | Path,
    *,
    valid_at: datetime,
    knowledge_cutoff: datetime,
) -> tuple[dict[str, tuple[str, ...]], NATOMembershipSnapshot | None]:
    current = latest_nato_membership(root, knowledge_cutoff=knowledge_cutoff)
    if current is None:
        return {}, None

    active = [
        member
        for member in current.members
        if member.joined_year <= valid_at.year
    ]
    pairs: dict[str, tuple[str, ...]] = {}
    for index, left in enumerate(active):
        for right in active[index + 1 :]:
            a, b = sorted((left.iso3, right.iso3))
            pairs[f"{a}|{b}"] = (
                "formal_alliance",
                "nato_collective_defence",
            )
    return pairs, current


__all__ = [
    "NATOMember",
    "NATOMembershipSnapshot",
    "current_nato_alliance_pairs",
    "latest_nato_membership",
    "parse_nato_members",
]
