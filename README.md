# PREACT – Predictive Early-warning for Coups and Atrocities

PREACT (Predictive Early-warning for Coups and Atrocities) is a modular, open-source
platform that ingests multi-domain signals to forecast risks of coups and mass
atrocities against civilians over short horizons (7–30 days). The project combines
transparent machine-learning models, Bayesian reasoning, and scenario simulation to
provide actionable early warning insights for policy makers, governments, and NGOs.

## System Overview

The project is organised into five primary layers:

1. **Data ingestion** – Automatic daily retrieval from open data APIs such as GDELT,
   ACLED, UNHCR/HDX, and economic indicators. Synthetic fallbacks ensure continuity.
2. **Feature store** – Harmonises and aggregates indicators (e.g. protest intensity,
   repression, commodity and FX stress) into time-series features per country.
3. **Predictive engine** – Ensemble of calibrated gradient boosting models and
   Bayesian explainability to generate probabilistic risk scores for coups and
   atrocities on 7–30 day horizons.
4. **Scoring and validation** – Produces interpretable diagnostics (Brier score,
   precision/recall) with explicit abstention thresholds when uncertainty is high.
5. **Output & interface** – Streamlit dashboard, machine-readable exports, and
   counterfactual scenario simulations for policy interventions.

## Repository Structure

```
preact/
├── config.py                 # Shared configuration dataclasses
├── data_ingestion/           # Connectors for GDELT, ACLED, synthetic sources
├── feature_store/            # Feature engineering and aggregation logic
├── models/                   # Predictive engine and backtesting utilities
├── evaluation/               # Metrics for model diagnostics
├── scenarios/                # Counterfactual scenario simulation tools
├── dashboard/                # Streamlit dashboard entry point
└── utils/                    # Persistence helpers
scripts/
└── update_pipeline.py        # CLI for running the daily pipeline
```

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -e .[dev]
   ```

2. **Run the synthetic pipeline**

   ```bash
   python scripts/update_pipeline.py --output-dir data
   ```

   This command fetches the latest open data (or synthetic placeholders), builds the
   feature store, trains calibrated gradient boosting models, and writes predictions to
   `data/predictions/`.

3. **Launch the dashboard**

   ```bash
   streamlit run preact/dashboard/app.py -- --prediction-dir data/predictions
   ```

   The dashboard provides country rankings, model diagnostics, and configurable alert
   thresholds. Use the sidebar to adjust the alert sensitivity or load validation
   outcomes for hindcasting studies.

## Data Source Authentication

Several upstream providers require authenticated access. Configure the following
environment variables before running the pipeline:

- `ACLED_USERNAME` and `ACLED_PASSWORD` – credentials used to request an OAuth access
  token from ACLED. The ingestion connector automatically exchanges them for a bearer
  token, caches it until expiry, and refreshes it as needed. Optional
  `ACLED_CLIENT_ID` and `ACLED_CLIENT_SECRET` variables are included in the token
  request when present.
- `UNHCR_API_TOKEN` – bearer token for the UNHCR Population API.
- `HDX_API_TOKEN` – bearer token for the Humanitarian Data Exchange API.

Tokens are injected as `Authorization` headers when contacting the APIs. If required
credentials are missing, the connector raises a descriptive error so you can provide
the appropriate values.

## Evaluation and Backtesting

- Historical hindcasting on cases such as Niger 2023 (coup), Sudan 2023 (atrocities),
  and Sri Lanka 2022 (unrest) can be performed using the `rolling_backtest` utility in
  `preact.models.predictor`.
- Metrics include Brier score, precision, and recall computed via
  `preact.evaluation.metrics`.

## Counterfactual Scenarios

The `preact.scenarios.counterfactual` module enables simulation of preventative
policies by perturbing feature trajectories (e.g. ramping humanitarian aid,
introducing mediation). Resulting probability deltas help assess potential impact.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request describing
your proposed enhancements. All code should include docstrings and unit tests where
applicable. Use `black`, `isort`, and `mypy` for code quality.

## License

This project is released under the MIT License.

