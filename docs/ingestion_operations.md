# Shared ingestion operations

## One external connection, multiple products

The Shared Data Hub is deployed as an independent service even though its code currently lives in PREACT. It is the only normal owner of overlapping external-provider connections.

Current overlap:
- GDELT -> PREACT + GoldenBull.

Likely future overlaps:
- World Bank/BIS/FRED-style macro series -> PREACT + GoldenBull.
- selected public company/macro news -> PREACT + GoldenBull.

Product-specific sources do not need forced centralisation. IBKR execution/market data and SEC/ESEF company fundamentals remain GoldenBull responsibilities unless PREACT later needs the same raw feed.

## Fan-out rule

Provider -> immutable snapshot/cache -> product projection

The raw provider response is not transformed into a "universal" domain model too early.

PREACT projection:
- historical polity/entity resolution;
- valid_time / known_time;
- conflict/institution/social/macro evidence;
- Atlas/Replay/Scenario features.

GoldenBull projection:
- issuer/ticker mapping;
- market session alignment;
- sentiment and price-impact labels;
- fundamentals and portfolio features.

## Resilience

GoldenBull currently retains direct GDELT access only as an emergency fallback. In normal operation, SHARED_DATA_HUB_URL is configured and the hub path is preferred.

The hub runs one worker so in-process single-flight locking can guarantee identical concurrent requests collapse into one external request. Persistent cache and immutable snapshots survive restarts.

## Recommended cadence

- GDELT raw: every 15 minutes.
- World Bank: daily snapshot.
- UNHCR: weekly, with optional crisis-specific higher cadence.
- UCDP Candidate: monthly/version-release driven.
- COW/CShapes/V-Dem/WPP/Maddison/SIPRI: release monitoring, not wasteful daily redownloads.

## Storage tiers

1. Raw immutable payloads: content-addressed, never overwritten.
2. Normalized evidence: DuckDB bitemporal records/documents.
3. Feature views: model-specific and reproducible from tiers 1-2.
4. Model artefacts: versioned separately with exact training cutoffs.

## Secrets

Provider credentials live only in the hub runtime where possible. They are never stored in snapshot metadata, request fingerprints or downstream product databases.
