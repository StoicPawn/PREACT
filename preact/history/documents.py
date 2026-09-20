"""Source-text evidence model for modern news and historical archives."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping


class TextAvailability(str, Enum):
    METADATA_ONLY = "metadata_only"
    EXCERPT_ONLY = "excerpt_only"
    FULL_TEXT_ALLOWED = "full_text_allowed"


@dataclass(frozen=True)
class HistoricalDocument:
    document_id: str
    source_id: str
    source_ref: str
    title: str
    published_at: datetime
    known_at: datetime
    acquired_at: datetime
    url: str | None = None
    language: str | None = None
    entity_ids: tuple[str, ...] = ()
    place_ids: tuple[str, ...] = ()
    text: str | None = None
    text_availability: TextAvailability = TextAvailability.METADATA_ONLY
    ocr_quality: float | None = None
    snapshot_checksum: str | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.known_at < self.published_at:
            raise ValueError("known_at cannot precede published_at")
        if self.acquired_at < self.known_at:
            raise ValueError("acquired_at cannot precede known_at")
        if self.ocr_quality is not None and not 0.0 <= self.ocr_quality <= 1.0:
            raise ValueError("ocr_quality must be in [0, 1]")
        if self.text and self.text_availability is TextAvailability.METADATA_ONLY:
            raise ValueError("metadata-only document cannot persist full text")
