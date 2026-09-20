from datetime import datetime, timezone

import pytest

from preact.history.entities import EntityCode, EntityRegistry, PoliticalEntity


UTC = timezone.utc


def dt(year: int) -> datetime:
    return datetime(year, 1, 1, tzinfo=UTC)


def test_code_can_be_reused_across_non_overlapping_historical_entities() -> None:
    registry = EntityRegistry(
        [
            PoliticalEntity(
                entity_id="polity:old",
                name="Old State",
                valid_from=dt(1900),
                valid_to=dt(1950),
                codes=(EntityCode("provider", "ABC"),),
            ),
            PoliticalEntity(
                entity_id="polity:new",
                name="New State",
                valid_from=dt(1950),
                codes=(EntityCode("provider", "ABC"),),
            ),
        ]
    )

    assert registry.resolve_code("provider", "ABC", at=dt(1940)).entity_id == "polity:old"
    assert registry.resolve_code("provider", "ABC", at=dt(2000)).entity_id == "polity:new"

    with pytest.raises(ValueError, match="ambiguous"):
        registry.resolve_code("provider", "ABC")


def test_active_at_respects_validity_intervals() -> None:
    registry = EntityRegistry(
        [
            PoliticalEntity("a", "A", dt(1900), dt(1950)),
            PoliticalEntity("b", "B", dt(1950), None),
        ]
    )
    assert [entity.entity_id for entity in registry.active_at(dt(1940))] == ["a"]
    assert [entity.entity_id for entity in registry.active_at(dt(1960))] == ["b"]
