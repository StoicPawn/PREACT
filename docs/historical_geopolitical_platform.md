# PREACT Historical-Geopolitical Platform

## Vision

Evolve PREACT from a narrow early-warning prototype into a research-grade platform for historical reconstruction, present-day geopolitical analysis and probabilistic scenario simulation.

The platform is intended for researchers, journalists, public institutions, policy analysts and historians. It must support evidence-based exploration without presenting model output as political truth or deterministic prediction.

## Core principles

1. **Time-causal by construction** — every observation has both an event time and a knowledge time. Historical replay may only use evidence that was available at the simulated cutoff date.
2. **Facts, inference and simulation are separate objects** — raw observations, model-derived estimates and counterfactual scenarios must never be silently mixed.
3. **Provenance first** — every datum records source, retrieval date, original reference, transformation lineage and confidence metadata where available.
4. **Uncertainty is a first-class output** — forecasts and reconstructions expose probability distributions or intervals, calibration diagnostics and competing explanations where appropriate.
5. **No single country score as ground truth** — country risk is decomposed into interpretable dimensions such as political violence, interstate conflict, institutional stress, fiscal/financial stress, economic fragility, humanitarian stress and information/media signals.
6. **Historical validation before future forecasting** — a model must demonstrate value in rolling-origin historical evaluation across regions and periods.

## Three operating modes

### Historical Atlas
Explore what was known about a country or region at a specific date, how alliances, borders, institutions, trade and conflict networks evolved, and which interpretations are directly sourced versus inferred.

### Historical Replay Lab
Choose a cutoff date, freeze the knowledge base there, train only on earlier information, forecast a fixed horizon, compare with subsequent outcomes and score calibration and failure modes. This is the main defence against hindsight bias and data leakage.

### Scenario Lab
Start from an observed historical or current state and apply explicit interventions or shocks, such as commodity prices, trade disruptions, alliance changes, institutional changes, migration, conflict or climate shocks. Outputs are labelled counterfactual simulations rather than forecasts.

## Canonical temporal model

Every record uses two clocks:
- **valid_time**: when the fact or event was true or occurred;
- **known_time**: when the platform or source could have known it.

This enables point-in-time queries without leaking later corrections, revised statistics or retrospective classifications.

## Evidence classes

- OBSERVATION: directly sourced event or measurement.
- DERIVED: deterministic transformation or aggregation.
- ESTIMATE: statistical reconstruction of an unobserved quantity.
- FORECAST: prediction about a later time.
- COUNTERFACTUAL: result conditional on an intervention.
- INTERPRETATION: sourced qualitative claim or scholarly interpretation.

The UI should render these classes differently.

## Data layers

### Contemporary / near-real-time
- GDELT for global news and event signals.
- ACLED for political violence and demonstrations where access permits.
- UCDP for organized violence and conflict event histories.
- World Bank and similar official macro series.
- UNHCR and humanitarian sources.
- V-Dem for institutional indicators.

### Long-run historical
- Correlates of War for state-system membership, wars, alliances, disputes and capabilities.
- Maddison Project Database for long-run economic development.
- Clio Infra for historical socioeconomic and institutional indicators.
- Seshat Global History Databank for long-run social and political structure where suitable.

Each connector must preserve licence and citation requirements.

## Model architecture

Use an ensemble, not a universal monolith:
1. temporal tabular models for country-period risk;
2. event-history and survival models for onset and duration;
3. graph models for interstate and actor networks;
4. sequence models for event streams and text-derived signals;
5. Bayesian or structural models for causal hypotheses and uncertainty;
6. agent-based or system-dynamics components for counterfactual simulation.

LLMs may assist with source extraction, entity resolution and narrative synthesis, but numeric risk estimates must remain auditable back to structured evidence.

## Evaluation

Minimum requirements:
- rolling-origin historical evaluation;
- Brier score and calibration curves;
- log loss;
- precision/recall at operational thresholds;
- lead time before events;
- false-alarm burden;
- performance by region and period;
- robustness to missing or revised data;
- source ablations.

Any benchmark using future information or revised post-event labels without point-in-time controls is invalid.

## Product surfaces

### Country / polity page
State at selected date, decomposed risk dimensions with uncertainty, timeline of sourced events, structural indicators, relationship graph, model evidence and historical analogues.

### Event page
Chronology, actors, locations, competing source accounts, machine-readable event representation and later observed consequences.

### Compare mode
Compare countries, periods or historical episodes along explicit dimensions without collapsing them into an opaque overall ranking.

### Research API / notebook
Reproducible queries with exact data versions, cutoff dates, features and model version.

## Milestones

### M0 — Temporal foundation
Canonical entity identifiers; valid_time + known_time schema; provenance; as-of filtering; source versioning.

### M1 — Historical replay
Rolling cutoff runner; train/test orchestration; leakage checks; calibration report.

### M2 — Country intelligence
Multidimensional country state; map/timeline/graph API; sourced evidence panel.

### M3 — Scenario engine
Explicit interventions; stochastic simulations; sensitivity analysis; counterfactual comparison.

### M4 — Research-grade validation
Reproducible benchmark suite; model cards; source/licence registry; documented failure cases.

## Architectural decision

The existing PREACT coup/atrocity pipeline remains one specialised risk module. The historical temporal layer becomes the common substrate for all future risk, replay and simulation modules.
