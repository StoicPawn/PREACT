"""Streamlit dashboard for the PREACT early warning system."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from preact.evaluation.metrics import summary_table
from preact.dashboard.layout import render_dashboard
from preact.dashboard.sample_data import (
    generate_sample_evidence,
    generate_sample_outcomes,
    generate_sample_predictions,
)

st.set_page_config(page_title="PREACT Early Warning", layout="wide")
st.title("PREACT – Global Early Warning Dashboard")


def load_predictions(path: Path) -> Dict[str, pd.Series]:
    """Load stored predictions from parquet files."""

    predictions: Dict[str, pd.Series] = {}
    for file in path.glob("*.parquet"):
        frame = pd.read_parquet(file)
        series = frame["probability"]
        if "country" in frame.columns:
            name = str(frame["country"].iloc[0])
        else:
            name = file.stem.replace("_", " ").title()
        predictions[name] = series
    return predictions


def load_bayesian_evidence(path: Path) -> pd.DataFrame | None:
    evidence_file = path / "bayesian_evidence.json"
    if not evidence_file.exists():
        return None
    frame = pd.read_json(evidence_file)
    if frame.empty:
        return None
    return frame


def prepare_predictions(path: Path) -> Tuple[Dict[str, pd.Series], bool]:
    """Load predictions or provide a deterministic sample fallback."""

    predictions = load_predictions(path)
    if predictions:
        return predictions, False

    sample_predictions = generate_sample_predictions()
    return sample_predictions, True


def main(prediction_dir: str, outcomes_path: str | None = None) -> None:
    st.sidebar.header("Data inputs")
    prediction_path = Path(prediction_dir)
    predictions, using_sample = prepare_predictions(prediction_path)

    if using_sample:
        st.sidebar.warning(
            "Nessun file di previsione trovato. Mostriamo dati di esempio per l'MVP."
        )

    threshold = st.sidebar.slider("Alert threshold", min_value=0.1, max_value=0.9, value=0.65)

    outcomes: pd.Series | None = None
    diagnostics: pd.DataFrame | None = None
    evidence = load_bayesian_evidence(prediction_path)

    if outcomes_path and Path(outcomes_path).exists():
        outcomes = pd.read_parquet(outcomes_path)["outcome"]
    elif using_sample:
        outcomes = generate_sample_outcomes(predictions)

    if outcomes is not None:
        diagnostics = summary_table(predictions, outcomes, threshold)

    if using_sample and (evidence is None or evidence.empty):
        evidence = generate_sample_evidence(predictions)

    render_dashboard(
        predictions=predictions,
        threshold=threshold,
        outcomes=outcomes,
        summary=diagnostics,
        evidence=evidence,
    )


if __name__ == "__main__":  # pragma: no cover - entry point
    main(prediction_dir="data/predictions")

