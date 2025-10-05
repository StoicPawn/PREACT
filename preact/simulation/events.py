"""Global event timeline models for PREACT simulations."""
from __future__ import annotations
"""Global event timeline models for PREACT simulations."""

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .economy import Shock
from .policy import PolicyAdjustment


@dataclass(frozen=True)
class GlobalEvent:
    """Representation of an exogenous event impacting policy and economy."""

    name: str
    category: str
    intensity: float
    start_tick: int = 0
    end_tick: int | None = None
    regions: Sequence[str] = field(default_factory=tuple)
    economic_multiplier: float = 1.0
    inflation_pressure: float = 0.0
    policy_adjustment: PolicyAdjustment | None = None

    def is_active(self, tick: int) -> bool:
        if tick < self.start_tick:
            return False
        if self.end_tick is None:
            return True
        return tick <= self.end_tick

    def economic_intensity(self) -> float:
        """Return the intensity contribution for economic shocks."""

        return float(self.intensity * self.economic_multiplier)

    def inflation_intensity(self) -> float:
        """Return the induced inflation contribution."""

        return float(self.intensity * self.inflation_pressure)


@dataclass(frozen=True)
class EventSnapshot:
    """State of the event timeline for a specific tick."""

    tick: int
    events: tuple[GlobalEvent, ...]
    shock: Shock | None
    adjustment: PolicyAdjustment | None
    inflation_delta: float

    @property
    def event_names(self) -> tuple[str, ...]:
        return tuple(event.name for event in self.events)

    @property
    def economic_intensity(self) -> float:
        return sum(event.economic_intensity() for event in self.events)


class EventTimeline:
    """Timeline of global events that can affect the simulation run."""

    def __init__(self, events: Iterable[GlobalEvent]):
        self._events = tuple(sorted(events, key=lambda event: event.start_tick))

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self._events)

    @property
    def events(self) -> tuple[GlobalEvent, ...]:
        return self._events

    def active_events(self, tick: int) -> tuple[GlobalEvent, ...]:
        """Return events active at the given tick."""

        return tuple(event for event in self._events if event.is_active(tick))

    def aggregate_adjustment(self, tick: int) -> PolicyAdjustment | None:
        """Combine policy adjustments from all active events."""

        active = self.active_events(tick)
        if not active:
            return None
        adjustments = [event.policy_adjustment for event in active if event.policy_adjustment]
        if not adjustments:
            return None
        merged = adjustments[0]
        for adjustment in adjustments[1:]:
            merged = merged.combine(adjustment)
        return merged

    def aggregate_shock(self, tick: int, base: Shock | None = None) -> Shock | None:
        """Combine active events into a composite shock."""

        active = self.active_events(tick)
        if not active:
            return base
        total_intensity = sum(event.economic_intensity() for event in active)
        if base and base.is_active(tick):
            intensity = float(base.intensity + total_intensity)
            start_tick = min(base.start_tick, tick)
            end_tick = base.end_tick
            name = base.name
        elif base:
            intensity = float(base.intensity + total_intensity)
            start_tick = min(base.start_tick, tick)
            end_tick = base.end_tick
            name = base.name
        else:
            intensity = float(total_intensity)
            start_tick = tick
            end_tick = None if any(event.end_tick is None for event in active) else max(
                event.end_tick or tick for event in active
            )
            name = "GlobalEvents"
        intensity = float(np.clip(intensity, -5.0, 5.0))
        return Shock(name=name, intensity=intensity, start_tick=start_tick, end_tick=end_tick)

    def snapshot(self, tick: int, base: Shock | None = None) -> EventSnapshot:
        """Return the combined impact of all events for a tick."""

        active = self.active_events(tick)
        adjustment = self.aggregate_adjustment(tick)
        shock = self.aggregate_shock(tick, base=base)
        inflation_delta = sum(event.inflation_intensity() for event in active)
        return EventSnapshot(
            tick=tick,
            events=active,
            shock=shock,
            adjustment=adjustment,
            inflation_delta=float(inflation_delta),
        )
