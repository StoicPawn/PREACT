from datetime import datetime, timezone
import json

from preact.data_hub.gateway import ProviderResponse, SharedProviderGateway


UTC = timezone.utc


def test_gateway_fingerprint_is_order_independent(tmp_path) -> None:
    gateway = SharedProviderGateway(tmp_path)
    first = gateway.fingerprint("gdelt", "doc", {"q": "x", "n": 10})
    second = gateway.fingerprint("gdelt", "doc", {"n": 10, "q": "x"})
    assert first == second


def test_gateway_cache_round_trip(tmp_path) -> None:
    gateway = SharedProviderGateway(tmp_path)
    fingerprint = gateway.fingerprint("gdelt", "doc", {"q": "x"})
    response = ProviderResponse(
        source_id="gdelt",
        operation="doc",
        payload={"articles": [{"title": "Example"}]},
        retrieved_at=datetime.now(UTC),
        cached=False,
        request_fingerprint=fingerprint,
        snapshot_checksum="abc",
    )
    gateway._write_cache(response)
    cached = gateway._read_cache(
        source_id="gdelt",
        operation="doc",
        fingerprint=fingerprint,
        ttl_seconds=3600,
    )
    assert cached is not None
    assert cached.cached is True
    assert cached.payload["articles"][0]["title"] == "Example"
    assert cached.snapshot_checksum == "abc"


def test_gateway_fingerprint_distinguishes_same_params_on_different_urls(tmp_path) -> None:
    gateway = SharedProviderGateway(tmp_path)
    params = {"date": "2020:2024", "format": "json"}
    italy = gateway.fingerprint(
        "world_bank",
        "indicator",
        params,
        url="https://api.worldbank.org/v2/country/ITA/indicator/X",
    )
    france = gateway.fingerprint(
        "world_bank",
        "indicator",
        params,
        url="https://api.worldbank.org/v2/country/FRA/indicator/X",
    )
    assert italy != france



def test_gateway_stats_count_cached_reuse(tmp_path) -> None:
    gateway = SharedProviderGateway(tmp_path)
    fingerprint = gateway.fingerprint(
        "gdelt",
        "doc",
        {"q": "x"},
        url="https://example.test/api",
    )
    response = ProviderResponse(
        source_id="gdelt",
        operation="doc",
        payload={"articles": []},
        retrieved_at=datetime.now(UTC),
        cached=False,
        request_fingerprint=fingerprint,
        snapshot_checksum="abc",
    )
    gateway._write_cache(response)

    cached = gateway.get_json(
        source_id="gdelt",
        operation="doc",
        url="https://example.test/api",
        params={"q": "x"},
        ttl_seconds=3600,
    )
    assert cached.cached is True
    stats = gateway.stats()["gdelt:doc"]
    assert stats["calls"] == 1
    assert stats["cache_hits"] == 1
    assert stats["external_requests"] == 0
    assert stats["deduplicated_requests"] == 1
