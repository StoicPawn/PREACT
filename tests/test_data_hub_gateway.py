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
