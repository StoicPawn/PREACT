# ACEPC Shared News Architecture

## Principle

External information is acquired once by the ACEPC Shared Data Hub and then reused by
independent products.

```text
Google News RSS ─┐
                 ├─> Shared Data Hub ─> immutable snapshots ─> SharedNewsStore
GDELT DOC ───────┤                                      │
                 │                                      ├─> GoldenBull projection
GDELT raw ───────┘                                      └─> PREACT projection
```

Products must not need to call upstream providers during normal read paths.

## Source roles

### GDELT Events / Mentions / GKG — H24 backbone

Role: structured global sensing.

- Provider-native updates are published continuously in 15-minute batches.
- The Shared Data Hub archives raw provider bytes once.
- PREACT derives event timelines, country relationships and later candidate claims from
  the archived stream.
- A temporary provider outage does not invalidate already archived evidence.
- Reprocessing does not require a refetch because source snapshots are immutable.

This is the primary machine-readable world-event channel.

### GDELT DOC — targeted enrichment

Role: article discovery/search over GDELT's news corpus.

- Useful for targeted market, country, actor or topic discovery.
- More appropriate for selective enrichment than as the sole H24 backbone.
- The hub uses long TTLs for broad recurring feeds so a 15-minute scheduler does not
  become a 15-minute external request.
- Rate limiting or temporary failure degrades only this enrichment channel.

### Google News RSS — freshness and readable headline discovery

Role: recent readable headline discovery.

- Useful for market/home feeds and high-level political/news discovery.
- Treated as replaceable because RSS search is not the durable system-of-record contract.
- Failure of Google News RSS must not stop GDELT raw acquisition, World Knowledge or the
  rest of the Shared News archive.

## Default cadence

The Shared News scheduler runs every 15 minutes.

- Google News broad feeds: TTL 15 minutes.
- GDELT DOC broad enrichment: TTL 120 minutes.
- GDELT raw Events/Mentions/GKG: separate raw collector every 15 minutes.
- GDELT CAMEO country reference: refreshed by the shared collector only when stale.

The scheduler may run more often than a provider's TTL; the SharedProviderGateway cache
prevents unnecessary external requests.

## Local archive

`SharedNewsStore` keeps metadata only:

- canonical article id;
- title;
- publisher/domain;
- URL;
- published time;
- retrieval/knowledge time;
- language;
- snippet and image URL when supplied by the provider;
- provider observation;
- feed identifier;
- source snapshot checksum.

Article text is not mirrored by default.

## Deduplication

One canonical article may have multiple provider observations.

Example:

```text
canonical article
 ├─ Google News RSS observation @ 09:00
 ├─ GDELT observation @ 09:07
 └─ later GDELT observation @ 09:35
```

Cross-provider matching currently uses normalized title + publisher domain + publication
day, with provider URLs retained separately. The model is deliberately conservative:
uncertain matches remain separate rather than incorrectly merging distinct articles.

## Point-in-time semantics

Every provider observation retains its own metadata and `retrieved_at`.

A query with `known_cutoff=T` is evaluated only against observations available by T.
Later snippets, URLs or provider enrichments are not allowed to leak into historical
replay.

This is required by both PREACT historical replay and GoldenBull ex-ante research.

## API contract

Normal product reads use local endpoints such as:

```text
GET /v1/news/latest
GET /v1/news/latest?q=sanctions
GET /v1/news/latest?provider=gdelt
GET /v1/news/latest?known_cutoff=...
GET /v1/news/stats
```

These endpoints set:

```text
external_provider_call = false
```

Provider-specific endpoints remain available for operations, diagnostics and explicit
refresh workflows, but they are not the intended application read path.

## Product ownership

The Shared Data Hub owns:

- provider access;
- schedules;
- throttling;
- caching;
- immutable source snapshots;
- canonical news metadata;
- provider-level deduplication;
- shared entity/reference dictionaries where appropriate.

GoldenBull owns:

- ticker/entity relevance;
- financial sentiment;
- market impact;
- portfolio relevance;
- trading/research interpretation.

PREACT owns:

- geopolitical actor/entity resolution;
- event interpretation;
- relationship signals;
- factual-claim extraction and corroboration;
- World Knowledge mutation;
- historical importance;
- forecast/scenario interpretation.

No product-specific score belongs in SharedNewsStore.

## Failure policy

The acquisition system is fail-soft by provider and fail-closed for factual world-state
updates.

A broken RSS feed may reduce headline freshness but cannot corrupt World Knowledge.
A GDELT DOC rate limit may delay article enrichment but cannot stop the raw GDELT stream.
A raw event signal is never sufficient by itself to overwrite a factual country profile.
