# Shared source ownership

This matrix decides whether a provider belongs to the neutral Shared Data Hub or to one product.

| Source | Owner | Current consumers | Reason |
|---|---|---|---|
| GDELT DOC + raw Events/Mentions/GKG | Shared Data Hub | PREACT, GoldenBull | Same public provider, overlapping news/event acquisition, meaningful rate-limit/cache reuse. |
| Google News RSS | Shared Data Hub | PREACT, GoldenBull | Same discovery feed; raw RSS can be fetched/snapshotted once per identical query and projected differently. |
| World Bank Indicators | Shared Data Hub | PREACT; GoldenBull candidate | General macro-country evidence likely reusable by both products. |
| BIS / FRED-ALFRED / OECD / Eurostat | Shared Data Hub when implemented | PREACT; GoldenBull candidate | General macro/financial context, not product-owned semantics. |
| Yahoo chart feed | GoldenBull | GoldenBull | Market-price fallback tied to ticker/timeframe archive and trading research. |
| IBKR | GoldenBull | GoldenBull | Broker/account/execution and market-data responsibility; not a neutral public-data provider. |
| SEC CompanyFacts/Submissions | GoldenBull | GoldenBull | Issuer-level fundamentals and filing workflow. Move only if PREACT later needs the exact raw SEC feed. |
| ESEF / filings.xbrl.org | GoldenBull | GoldenBull | Issuer fundamentals and canonical accounting normalization. |
| COW, UCDP, V-Dem, CShapes, SIPRI, UNHCR, WPP, historical press | PREACT / historical ingestion | PREACT | Geopolitical/historical backbone with bitemporal and polity-resolution semantics. |
| ACLED and similar licensed conflict feeds | Shared credentials layer only if another product needs them | PREACT today | Centralize credentials/rate limiting only when there is a real second consumer. |

## Rule for new providers

A provider goes into the Shared Data Hub when at least one of these is true:

1. two products need the same upstream endpoint or raw payload;
2. credentials, quotas or provider rate limits should have a single owner;
3. one immutable raw snapshot can support multiple downstream projections;
4. provider revisions should be archived once and reused.

A provider stays product-specific when its semantics are inseparable from that product, such as broker execution, account state, issuer filing normalization or strategy-specific market archives.

## Fan-out contract

External provider -> shared raw snapshot/cache -> independent product projections.

The hub does not create a giant universal feature model. It preserves provider-native evidence and operational concerns. PREACT and GoldenBull retain separate normalized schemas, models, labels and refresh policies.

## Resilience

GoldenBull retains direct GDELT and Google News access only as emergency fallback while the shared service is being deployed. Normal operation should configure SHARED_DATA_HUB_URL so provider calls are centralized.

## Measurement

The hub exposes /v1/stats with internal call count, cache hits, single-flight reuse, external requests and deduplicated request count per provider operation. This is the acceptance evidence for the shared-connection design.
