"""Common-fold predictive benchmark suite for rare geopolitical events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .hazard import ComplementaryLogLogHazard
from .temporal_cv import TemporalFold, purged_panel_folds


@dataclass(frozen=True)
class BenchmarkMetrics:
    rows: int
    events: int
    brier: float | None
    hierarchical_baseline_brier: float | None
    brier_skill: float | None
    log_loss: float | None
    roc_auc: float | None
    average_precision: float | None
    calibration_gap: float | None
    worst_fold_brier_skill: float | None


@dataclass(frozen=True)
class SkillInterval:
    lower: float | None
    median: float | None
    upper: float | None
    samples: int


@dataclass(frozen=True)
class ModelBenchmark:
    name: str
    predictions: pd.DataFrame
    metrics: BenchmarkMetrics
    brier_skill_interval: SkillInterval


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    models: Mapping[str, ModelBenchmark]
    folds: tuple[TemporalFold, ...]
    feature_columns: tuple[str, ...]


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(q / (1.0 - q)).reshape(-1, 1)


def _builders(random_state: int) -> dict[str, Callable[[], object]]:
    return {
        "logistic_l2": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "cloglog_hazard": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", ComplementaryLogLogHazard(l2=1.0, max_iter=1000)),
            ]
        ),
        "hist_gradient_boosting": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=200,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _hierarchical_rates(
    y: pd.Series,
    *,
    entity_ids: pd.Index,
    shrinkage: float = 20.0,
) -> tuple[float, dict[str, float]]:
    global_rate = float((y.sum() + 0.5) / (len(y) + 1.0))
    frame = pd.DataFrame(
        {
            "entity_id": entity_ids.astype(str),
            "actual": y.to_numpy(dtype=int),
        }
    )
    grouped = frame.groupby("entity_id")["actual"].agg(["sum", "count"])
    rates = {
        str(entity): float(
            (row["sum"] + shrinkage * global_rate) / (row["count"] + shrinkage)
        )
        for entity, row in grouped.iterrows()
    }
    return global_rate, rates


def _baseline_for(
    entity_ids: pd.Index,
    global_rate: float,
    entity_rates: Mapping[str, float],
) -> np.ndarray:
    return np.array(
        [float(entity_rates.get(str(entity), global_rate)) for entity in entity_ids],
        dtype=float,
    )


def _calibrate(
    model,
    x_cal: pd.DataFrame,
    y_cal: pd.Series,
    raw_test: np.ndarray,
) -> np.ndarray:
    if len(x_cal) < 20 or y_cal.nunique() < 2:
        return np.clip(raw_test, 1e-6, 1 - 1e-6)
    raw_cal = model.predict_proba(x_cal)[:, 1]
    calibrator = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    calibrator.fit(_logit(raw_cal), y_cal)
    return np.clip(
        calibrator.predict_proba(_logit(raw_test))[:, 1],
        1e-6,
        1 - 1e-6,
    )


def _metric_frame(df: pd.DataFrame) -> BenchmarkMetrics:
    if df.empty:
        return BenchmarkMetrics(0, 0, None, None, None, None, None, None, None, None)
    y = df["actual"].astype(int)
    p = df["probability"].astype(float).clip(1e-6, 1 - 1e-6)
    b = df["baseline_probability"].astype(float).clip(1e-6, 1 - 1e-6)
    brier = float(brier_score_loss(y, p))
    base_brier = float(brier_score_loss(y, b))
    skill = float(1.0 - brier / base_brier) if base_brier > 0 else None
    fold_skills = []
    for _, group in df.groupby("fold"):
        gb = float(brier_score_loss(group["actual"], group["probability"]))
        gbase = float(
            brier_score_loss(group["actual"], group["baseline_probability"])
        )
        if gbase > 0:
            fold_skills.append(1.0 - gb / gbase)
    return BenchmarkMetrics(
        rows=int(len(df)),
        events=int(y.sum()),
        brier=brier,
        hierarchical_baseline_brier=base_brier,
        brier_skill=skill,
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        roc_auc=float(roc_auc_score(y, p)) if y.nunique() > 1 else None,
        average_precision=float(average_precision_score(y, p))
        if y.nunique() > 1
        else None,
        calibration_gap=float(p.mean() - y.mean()),
        worst_fold_brier_skill=float(min(fold_skills)) if fold_skills else None,
    )


def block_bootstrap_brier_skill(
    predictions: pd.DataFrame,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> SkillInterval:
    if predictions.empty or samples < 1:
        return SkillInterval(None, None, None, 0)
    dates = pd.Index(predictions["date"].unique())
    if len(dates) < 2:
        return SkillInterval(None, None, None, 0)
    rng = np.random.default_rng(seed)
    values = []
    by_date = {date: frame for date, frame in predictions.groupby("date")}
    for _ in range(int(samples)):
        drawn = rng.choice(dates.to_numpy(), size=len(dates), replace=True)
        sample = pd.concat([by_date[date] for date in drawn], ignore_index=True)
        model_brier = float(
            brier_score_loss(sample["actual"], sample["probability"])
        )
        base_brier = float(
            brier_score_loss(sample["actual"], sample["baseline_probability"])
        )
        if base_brier > 0:
            values.append(1.0 - model_brier / base_brier)
    if not values:
        return SkillInterval(None, None, None, 0)
    q = np.quantile(values, [0.025, 0.5, 0.975])
    return SkillInterval(float(q[0]), float(q[1]), float(q[2]), len(values))


def run_benchmark_suite(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    horizon_days: int,
    min_train_dates: int = 20,
    calibration_dates: int = 5,
    test_dates_per_fold: int = 5,
    entity_shrinkage: float = 20.0,
    bootstrap_samples: int = 1000,
    random_state: int = 42,
) -> BenchmarkSuiteResult:
    """Evaluate all candidate models on identical purged calendar folds."""

    if not isinstance(features.index, pd.MultiIndex):
        raise TypeError("features must use a panel MultiIndex")
    x = features.sort_index().copy()
    y = target.reindex(x.index)
    valid = y.notna()
    x = x.loc[valid]
    y = y.loc[valid].astype(int)
    folds = purged_panel_folds(
        x.index,
        horizon_days=horizon_days,
        min_train_dates=min_train_dates,
        calibration_dates=calibration_dates,
        test_dates_per_fold=test_dates_per_fold,
    )
    predictions: dict[str, list[dict]] = {
        name: [] for name in _builders(random_state)
    }

    dates_index = x.index.get_level_values("date")
    entity_index = x.index.get_level_values("entity_id")

    for fold in folds:
        fit_mask = dates_index.isin(fold.fit_dates)
        cal_mask = dates_index.isin(fold.calibration_dates)
        test_mask = dates_index.isin(fold.test_dates)
        x_fit, y_fit = x.loc[fit_mask], y.loc[fit_mask]
        x_cal, y_cal = x.loc[cal_mask], y.loc[cal_mask]
        x_test, y_test = x.loc[test_mask], y.loc[test_mask]
        if x_fit.empty or x_test.empty or y_fit.nunique() < 2:
            continue

        history_y = pd.concat([y_fit, y_cal])
        history_entities = pd.Index(
            list(x_fit.index.get_level_values("entity_id"))
            + list(x_cal.index.get_level_values("entity_id"))
        )
        global_rate, entity_rates = _hierarchical_rates(
            history_y,
            entity_ids=history_entities,
            shrinkage=entity_shrinkage,
        )
        test_entities = pd.Index(x_test.index.get_level_values("entity_id"))
        baseline = _baseline_for(test_entities, global_rate, entity_rates)

        for offset, (name, builder) in enumerate(_builders(random_state).items()):
            model = builder()
            model.fit(x_fit, y_fit)
            raw_test = model.predict_proba(x_test)[:, 1]
            probabilities = _calibrate(model, x_cal, y_cal, raw_test)
            for idx, actual, probability, base_probability in zip(
                x_test.index,
                y_test.to_numpy(),
                probabilities,
                baseline,
            ):
                predictions[name].append(
                    {
                        "date": pd.Timestamp(idx[x.index.names.index("date")]),
                        "entity_id": str(
                            idx[x.index.names.index("entity_id")]
                        ),
                        "fold": fold.fold,
                        "actual": int(actual),
                        "probability": float(probability),
                        "baseline_probability": float(base_probability),
                        "training_cutoff": fold.training_cutoff,
                    }
                )

    models: dict[str, ModelBenchmark] = {}
    for name, rows in predictions.items():
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values(["date", "entity_id"]).reset_index(drop=True)
        metrics = _metric_frame(frame)
        interval = block_bootstrap_brier_skill(
            frame,
            samples=bootstrap_samples,
            seed=random_state,
        )
        models[name] = ModelBenchmark(name, frame, metrics, interval)

    return BenchmarkSuiteResult(
        models=models,
        folds=tuple(folds),
        feature_columns=tuple(str(c) for c in x.columns),
    )
