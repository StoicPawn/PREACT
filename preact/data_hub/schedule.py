"""Recommended acquisition cadence for shared providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcquisitionSchedule:
    source_id: str
    cadence: str
    mode: str
    consumers: tuple[str, ...]
    notes: str = ""


SCHEDULES: tuple[AcquisitionSchedule, ...] = (
    AcquisitionSchedule(
        "gdelt", "15m", "realtime_raw_and_cached_api", ("preact", "goldenbull"),
        "Single shared collector for Events/Mentions/GKG; DOC queries deduplicated by fingerprint.",
    ),
    AcquisitionSchedule(
        "google_news_rss", "15m", "cached_rss", ("preact", "goldenbull"),
        "Query-specific feed snapshots; identical requests deduplicated by the hub.",
    ),
    AcquisitionSchedule(
        "world_bank", "daily", "snapshot_api", ("preact", "goldenbull_future"),
        "Daily is sufficient to capture revisions without excessive provider traffic.",
    ),
    AcquisitionSchedule(
        "ucdp", "monthly", "version_pinned_api", ("preact",),
        "Poll candidate releases separately from stable annual releases.",
    ),
    AcquisitionSchedule(
        "unhcr", "weekly", "snapshot_api", ("preact",),
        "Archive revisions; higher frequency can be enabled for active crises.",
    ),
    AcquisitionSchedule(
        "cow", "monthly_release_check", "versioned_bulk", ("preact",),
        "Download only when release/version changes.",
    ),
    AcquisitionSchedule(
        "cshapes", "monthly_release_check", "versioned_bulk", ("preact",),
        "Stable release plus separately flagged beta release.",
    ),
    AcquisitionSchedule(
        "vdem", "monthly_release_check", "versioned_bulk", ("preact",),
        "Keep all releases and archived versions.",
    ),
    AcquisitionSchedule(
        "un_wpp", "monthly_release_check", "versioned_bulk", ("preact",),
        "Revision-driven source; no need for daily polling.",
    ),
    AcquisitionSchedule(
        "maddison", "monthly_release_check", "versioned_bulk", ("preact",),
        "Release-driven long-run estimates.",
    ),
    AcquisitionSchedule(
        "sipri", "monthly_release_check", "publication_release", ("preact",),
        "Archive exact edition because historical values are revised.",
    ),
)


SCHEDULE_BY_SOURCE = {item.source_id: item for item in SCHEDULES}
