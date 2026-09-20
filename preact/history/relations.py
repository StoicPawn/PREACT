"""Temporal geopolitical relationships such as alliances, disputes and contiguity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True)
class HistoricalRelation:
    relation_id: str
    relation_type: str
    subject_entity_id: str
    object_entity_id: str
    valid_from: datetime
    known_at: datetime
    source: str
    source_ref: str
    retrieved_at: datetime
    directed: bool = False
    valid_to: datetime | None = None
    dataset_version: str | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)

    def is_known_as_of(self, cutoff: datetime) -> bool:
        return self.known_at <= cutoff

    def is_valid_at(self, when: datetime) -> bool:
        if when < self.valid_from:
            return False
        return self.valid_to is None or when < self.valid_to
