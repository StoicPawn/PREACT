"""Wave-1 ingestion plan for the global historical backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Automation = Literal["automatic", "registration", "manual_release"]


@dataclass(frozen=True)
class Wave1Source:
    source_id: str
    purpose: str
    automation: Automation
    replay_note: str


WAVE1: tuple[Wave1Source, ...] = (
    Wave1Source("cow", "wars, alliances, interstate disputes, capabilities and state system", "automatic",
                "Use provider release/version and immutable bulk files."),
    Wave1Source("cshapes", "historical borders and capitals", "automatic",
                "Versioned release; valid-time geometry."),
    Wave1Source("vdem", "political institutions and regime indicators", "registration",
                "Keep exact release; use archived releases where possible."),
    Wave1Source("world_bank", "macro, development, debt and demographics", "automatic",
                "Current API revisions require PREACT snapshots; old cutoffs need true vintages."),
    Wave1Source("ucdp", "organized violence and georeferenced conflict events", "automatic",
                "Prefer stable annual versions plus candidate-event release versions."),
    Wave1Source("gdelt", "news-derived events and article metadata", "automatic",
                "All access through Shared Data Hub; snapshot raw responses/files."),
    Wave1Source("unhcr", "refugees, displacement and humanitarian pressure", "automatic",
                "Archive each retrieval because historical series may be revised."),
    Wave1Source("un_wpp", "population and demographic structure", "automatic",
                "Bind every observation to the WPP revision."),
    Wave1Source("maddison", "long-run GDP and population", "automatic",
                "Release-based historical estimates; never silently replace revisions."),
    Wave1Source("sipri", "military expenditure, arms transfers and security data", "manual_release",
                "Archive exact published edition; historical values may be revised."),
)


WAVE1_BY_ID = {source.source_id: source for source in WAVE1}
