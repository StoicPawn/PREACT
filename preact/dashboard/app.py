"""Streamlit dashboard for the PREACT early warning system."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from ..evaluation.metrics import summary_table

st.set_page_config(page_title="PREACT Early Warning", layout="wide")
st.title("PREACT – Global Early Warning Dashboard")


def load_predictions(path: Path) -> Dict[str, pd.Series]:
    """Load stored predictions from parquet files."""

    predictions: Dict[str, pd.Series] = {}
    for file in path.glob("*.parquet"):
        series = pd.read_parquet(file)["probability"]
        predictions[file.stem] = series
    return predictions


def main(prediction_dir: str, outcomes_path: str | None = None) -> None:
    st.sidebar.header("Data Inputs")
    prediction_path = Path(prediction_dir)
    predictions = load_predictions(prediction_path)

    if not predictions:
        st.warning("No predictions found. Run the pipeline to generate outputs.")
        return

    threshold = st.sidebar.slider("Alert threshold", min_value=0.1, max_value=0.9, value=0.65)
    latest = {name: series.iloc[-1] for name, series in predictions.items()}
    st.metric("Top Alert Probability", max(latest.values()))

    if outcomes_path:
        outcomes = pd.read_parquet(outcomes_path)["outcome"]
        table = summary_table(predictions, outcomes, threshold)
        st.subheader("Model Diagnostics")
        st.dataframe(table)

    st.subheader("Latest Alerts")
    ranking = pd.Series(latest).sort_values(ascending=False)
    st.bar_chart(ranking)


if __name__ == "__main__":  # pragma: no cover - entry point
    main(prediction_dir="data/predictions")

