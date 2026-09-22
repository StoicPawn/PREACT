"""Point-in-time lineage summaries for nested predictive research artifacts."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from .benchmark_suite import BenchmarkSuiteResult
from .reporting import attach_prediction_feature_provenance, validate_benchmark_suite_fold_audits


def benchmark_suite_feature_lineage(
    suite: BenchmarkSuiteResult,
    feature_snapshot_fingerprints: pd.Series,
) -> dict[str, dict[str, int]]:
    """Validate and summarize exact feature vintages used by every OOS model.

    This is intended for stress/placebo/ablation reports.  It deliberately
    reuses the same fail-closed temporal and point-in-time contracts as the
    primary benchmark so nested diagnostics cannot silently lose provenance.
    """
    validate_benchmark_suite_fold_audits(suite)
    lineage: dict[str, dict[str, int]] = {}
    for name, benchmark in suite.models.items():
        predictions = benchmark.predictions
        if predictions is None:
            raise ValueError(f"benchmark {name} has no OOS predictions for lineage")
        attached = attach_prediction_feature_provenance(
            predictions,
            feature_snapshot_fingerprints,
        )
        lineage[name] = {
            "rows": int(len(attached)),
            "unique_feature_snapshots": int(
                attached["feature_snapshot_fingerprint"].nunique()
            ),
        }
    return lineage
