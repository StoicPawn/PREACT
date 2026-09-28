"""Non-predictive entity namespace normalization for geopolitical data.

PREACT stores historical providers in their native identifiers (for example COW ccode)
and modern feeds in ISO-3 identifiers.  This module builds a versioned semantic crosswalk
so those namespaces can participate in the same graph without exposing crosswalk fields as
model features.

The crosswalk is normalization metadata, not evidence: it may use a current published code
dictionary to identify a historical row, but it never exports state end dates, future
relations, outcomes, or any other substantive fact into the feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from typing import Mapping

import pycountry

from preact.history.connectors.cow import COWStateSystemConnector
from preact.history.snapshot_store import SnapshotMetadata, SourceSnapshotStore


_NAME_ALIASES = {
    "UNITED STATES OF AMERICA": "United States",
    "RUSSIA": "Russian Federation",
    "IRAN": "Iran, Islamic Republic of",
    "SYRIA": "Syrian Arab Republic",
    "BOLIVIA": "Bolivia, Plurinational State of",
    "VENEZUELA": "Venezuela, Bolivarian Republic of",
    "TANZANIA": "Tanzania, United Republic of",
    "VIETNAM": "Viet Nam",
    "MOLDOVA": "Moldova, Republic of",
    "SOUTH KOREA": "Korea, Republic of",
    "NORTH KOREA": "Korea, Democratic People's Republic of",
    "LAOS": "Lao People's Democratic Republic",
    "BRUNEI": "Brunei Darussalam",
    "CZECH REPUBLIC": "Czechia",
    "SWAZILAND": "Eswatini",
    "MACEDONIA": "North Macedonia",
    "EAST TIMOR": "Timor-Leste",
    "IVORY COAST": "Côte d'Ivoire",
    "CAPE VERDE": "Cabo Verde",
}


def country_name_to_iso3(name: str) -> str | None:
    value = str(name or "").strip()
    if not value:
        return None
    candidate = _NAME_ALIASES.get(value.upper(), value)
    try:
        return pycountry.countries.lookup(candidate).alpha_3
    except LookupError:
        try:
            matches = pycountry.countries.search_fuzzy(candidate)
        except LookupError:
            return None
        return matches[0].alpha_3 if len(matches) == 1 else None


def alias_fingerprint(aliases: Mapping[str, str]) -> str:
    payload = json.dumps(
        sorted((str(key), str(value)) for key, value in aliases.items()),
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class EntityAliasSnapshot:
    valid_at: datetime
    aliases: Mapping[str, str]
    fingerprint: str
    source_snapshot_checksum: str | None
    source_retrieved_at: datetime | None
    unresolved_active_cow_entities: int

    def canonical(self, entity_id: str) -> str:
        value = str(entity_id)
        return str(self.aliases.get(value, value))


class COWISO3SemanticCrosswalk:
    """Bridge active COW state codes and ISO-3 identifiers at a world-time cutoff."""

    def __init__(self, store: SourceSnapshotStore) -> None:
        self.store = store
        candidates = [
            item
            for item in store.iter_metadata(source_id="cow")
            if item.source_release == "State System Membership v2024"
        ]
        self.snapshot: SnapshotMetadata | None = (
            max(candidates, key=lambda item: item.retrieved_at)
            if candidates
            else None
        )
        if self.snapshot is None:
            self.entities = ()
        else:
            rows = COWStateSystemConnector.parse_rows(
                store.read_payload(self.snapshot)
            )
            self.entities = tuple(COWStateSystemConnector.to_entities(rows))

    def aliases_at(self, valid_at: datetime) -> EntityAliasSnapshot:
        aliases: dict[str, str] = {}
        unresolved = 0
        for entity in self.entities:
            if not entity.is_valid_at(valid_at):
                continue
            iso3 = country_name_to_iso3(entity.name)
            if iso3 is None:
                unresolved += 1
                continue
            canonical = f"iso3:{iso3}"
            aliases[canonical] = canonical
            for code in entity.codes:
                namespace, value = code.normalized()
                aliases[f"{namespace}:{value}"] = canonical

        return EntityAliasSnapshot(
            valid_at=valid_at,
            aliases=aliases,
            fingerprint=alias_fingerprint(aliases),
            source_snapshot_checksum=(
                self.snapshot.checksum_sha256 if self.snapshot is not None else None
            ),
            source_retrieved_at=(
                self.snapshot.retrieved_at if self.snapshot is not None else None
            ),
            unresolved_active_cow_entities=unresolved,
        )


__all__ = [
    "COWISO3SemanticCrosswalk",
    "EntityAliasSnapshot",
    "alias_fingerprint",
    "country_name_to_iso3",
]
