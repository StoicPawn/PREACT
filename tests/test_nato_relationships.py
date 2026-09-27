from datetime import datetime, timezone

from preact.history.snapshot_store import SourceSnapshotStore
from preact.intelligence.nato_relationships import (
    current_nato_alliance_pairs,
    latest_nato_membership,
    parse_nato_members,
)


NOW = datetime(2026, 9, 27, 18, tzinfo=timezone.utc)


def payload() -> bytes:
    countries = [
        ("France", 1949),
        ("Italy", 1949),
        ("United States", 1949),
        ("Sweden", 2024),
    ]
    # The parser has a sanity bound, so repeat with unique additional real names.
    countries += [
        ("Albania", 2009), ("Belgium", 1949), ("Bulgaria", 2004),
        ("Canada", 1949), ("Croatia", 2009), ("Czechia", 1999),
        ("Denmark", 1949), ("Estonia", 2004), ("Finland", 2023),
        ("Germany", 1955), ("Greece", 1952), ("Hungary", 1999),
        ("Iceland", 1949), ("Latvia", 2004), ("Lithuania", 2004),
        ("Luxembourg", 1949), ("Montenegro", 2017),
        ("North Macedonia", 2020), ("Norway", 1949),
        ("Poland", 1999), ("Portugal", 1949), ("Romania", 2004),
        ("Slovakia", 2004), ("Slovenia", 2004), ("Spain", 1982),
        ("Netherlands", 1949), ("Türkiye", 1952), ("United Kingdom", 1949),
    ]
    body = []
    for name, year in countries:
        body.append(
            f'<p class="cie-country__name" data-title="{name}">{name}</p>'
            f'<p class="cie-country__yearEntry">{year}</p>'
        )
    return "".join(body).encode()


def test_nato_parser_and_pair_anchor(tmp_path):
    members = parse_nato_members(payload())
    assert len(members) == 32
    assert any(item.iso3 == "SWE" and item.joined_year == 2024 for item in members)

    store = SourceSnapshotStore(tmp_path / "snapshots")
    snap = store.put(
        source_id="nato",
        payload=payload(),
        retrieved_at=NOW,
        source_url="https://www.nato.int/test",
        source_release="NATO current member countries",
        operation="current_members",
    )
    current = latest_nato_membership(
        tmp_path,
        knowledge_cutoff=NOW,
    )
    assert current is not None
    assert current.snapshot.checksum_sha256 == snap.checksum_sha256

    pairs, _ = current_nato_alliance_pairs(
        tmp_path,
        valid_at=NOW,
        knowledge_cutoff=NOW,
    )
    assert len(pairs) == 496
    assert "formal_alliance" in pairs["FRA|ITA"]
    assert "nato_collective_defence" in pairs["FRA|ITA"]


def test_nato_snapshot_is_not_available_before_it_was_known(tmp_path):
    store = SourceSnapshotStore(tmp_path / "snapshots")
    store.put(
        source_id="nato",
        payload=payload(),
        retrieved_at=NOW,
        source_url="https://www.nato.int/test",
        source_release="NATO current member countries",
        operation="current_members",
    )
    earlier = datetime(2026, 9, 26, 18, tzinfo=timezone.utc)
    assert latest_nato_membership(tmp_path, knowledge_cutoff=earlier) is None
