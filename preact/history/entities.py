"""Temporal entity crosswalk for countries, polities and historical states."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping


@dataclass(frozen=True)
class EntityCode:
    namespace: str
    value: str

    def normalized(self) -> tuple[str, str]:
        return self.namespace.strip().lower(), self.value.strip().upper()


@dataclass(frozen=True)
class PoliticalEntity:
    entity_id: str
    name: str
    valid_from: datetime
    valid_to: datetime | None = None
    codes: tuple[EntityCode, ...] = ()
    aliases: tuple[str, ...] = ()
    predecessors: tuple[str, ...] = ()
    successors: tuple[str, ...] = ()
    attributes: Mapping[str, object] = field(default_factory=dict)

    def is_valid_at(self, when: datetime) -> bool:
        if when < self.valid_from:
            return False
        return self.valid_to is None or when < self.valid_to


class EntityRegistry:
    """Resolve provider-specific codes to a canonical historical polity."""

    def __init__(self, entities: Iterable[PoliticalEntity] = ()) -> None:
        self._entities: dict[str, PoliticalEntity] = {}
        self._codes: dict[tuple[str, str], list[str]] = {}
        self._aliases: dict[str, list[str]] = {}
        for entity in entities:
            self.add(entity)

    def add(self, entity: PoliticalEntity) -> None:
        if entity.entity_id in self._entities:
            raise ValueError(f"duplicate entity_id: {entity.entity_id}")
        if entity.valid_to is not None and entity.valid_to <= entity.valid_from:
            raise ValueError("valid_to must be later than valid_from")
        self._entities[entity.entity_id] = entity
        for code in entity.codes:
            self._codes.setdefault(code.normalized(), []).append(entity.entity_id)
        names = (entity.name, *entity.aliases)
        for name in names:
            key = name.strip().casefold()
            if key:
                self._aliases.setdefault(key, []).append(entity.entity_id)

    def get(self, entity_id: str) -> PoliticalEntity:
        return self._entities[entity_id]

    def resolve_code(
        self,
        namespace: str,
        value: str,
        *,
        at: datetime | None = None,
    ) -> PoliticalEntity | None:
        ids = self._codes.get((namespace.strip().lower(), value.strip().upper()), [])
        candidates = [self._entities[entity_id] for entity_id in ids]
        if at is not None:
            candidates = [entity for entity in candidates if entity.is_valid_at(at)]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return None
        raise ValueError(
            f"ambiguous historical code {namespace}:{value}; provide a temporal cutoff"
        )

    def resolve_name(self, name: str, *, at: datetime | None = None) -> PoliticalEntity | None:
        ids = self._aliases.get(name.strip().casefold(), [])
        candidates = [self._entities[entity_id] for entity_id in ids]
        if at is not None:
            candidates = [entity for entity in candidates if entity.is_valid_at(at)]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return None
        raise ValueError(f"ambiguous historical entity name: {name}")

    def lineage(self, entity_id: str) -> dict[str, tuple[str, ...]]:
        entity = self.get(entity_id)
        return {
            "predecessors": entity.predecessors,
            "successors": entity.successors,
        }

    def active_at(self, when: datetime) -> list[PoliticalEntity]:
        return sorted(
            (entity for entity in self._entities.values() if entity.is_valid_at(when)),
            key=lambda entity: entity.name,
        )
