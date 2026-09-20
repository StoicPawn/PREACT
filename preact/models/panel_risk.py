"""Leakage-resistant global panel model for country/polity event risk."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class PanelBacktestMetrics:
    rows:int
    events:int
    brier:float|None
    baseline_brier:float|None
    brier_skill:float|None
    log_loss:float|None
    roc_auc:float|None
    average_precision:float|None
    calibration_gap:float|None


@dataclass(frozen=True)
class PanelBacktestResult:
    predictions:pd.DataFrame
    metrics:PanelBacktestMetrics
    folds_used:int
    folds_skipped:int


def _base_rate(y:pd.Series)->float:
    return float((y.sum()+0.5)/(len(y)+1.0))


def _model(random_state:int)->Pipeline:
    return Pipeline([
        ("imputer",SimpleImputer(strategy="median",add_indicator=True)),
        ("model",HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state,
        )),
    ])


def _metrics(df:pd.DataFrame)->PanelBacktestMetrics:
    if df.empty:
        return PanelBacktestMetrics(0,0,None,None,None,None,None,None,None)
    y=df["actual"].astype(int)
    p=df["probability"].astype(float).clip(1e-6,1-1e-6)
    b=df["baseline_probability"].astype(float).clip(1e-6,1-1e-6)
    bs=float(brier_score_loss(y,p))
    bbs=float(brier_score_loss(y,b))
    return PanelBacktestMetrics(
        rows=len(df),
        events=int(y.sum()),
        brier=bs,
        baseline_brier=bbs,
        brier_skill=float(1-bs/bbs) if bbs>0 else None,
        log_loss=float(log_loss(y,p,labels=[0,1])),
        roc_auc=float(roc_auc_score(y,p)) if y.nunique()>1 else None,
        average_precision=float(average_precision_score(y,p)) if y.nunique()>1 else None,
        calibration_gap=float(p.mean()-y.mean()),
    )


def panel_walk_forward_backtest(
    features:pd.DataFrame,
    target:pd.Series,
    *,
    horizon_days:int,
    min_train_dates:int=20,
    test_dates_per_fold:int=5,
    calibration_dates:int=5,
    random_state:int=42,
)->PanelBacktestResult:
    """Pooled multi-entity walk-forward evaluation split strictly by calendar date."""

    if not isinstance(features.index,pd.MultiIndex) or "date" not in features.index.names:
        raise TypeError("features must have MultiIndex containing date")
    x=features.sort_index()
    y=target.reindex(x.index).astype(int)
    dates=pd.Index(x.index.get_level_values("date").unique()).sort_values()
    purge=pd.Timedelta(days=max(0,int(horizon_days)))
    rows=[]
    used=0
    skipped=0

    start=min_train_dates
    fold=0
    while start < len(dates):
        test_dates=dates[start:start+max(1,test_dates_per_fold)]
        if len(test_dates)==0:
            break
        test_start=pd.Timestamp(test_dates[0])
        train_end=test_start-purge
        eligible_dates=dates[dates < train_end]
        if len(eligible_dates)<min_train_dates:
            skipped+=1
            start+=test_dates_per_fold
            continue

        cal_dates=eligible_dates[-max(1,calibration_dates):]
        base_dates=eligible_dates[:-len(cal_dates)]
        if len(base_dates)<max(10,min_train_dates-calibration_dates):
            skipped+=1
            start+=test_dates_per_fold
            continue

        base_mask=x.index.get_level_values("date").isin(base_dates)
        cal_mask=x.index.get_level_values("date").isin(cal_dates)
        test_mask=x.index.get_level_values("date").isin(test_dates)
        xb,yb=x.loc[base_mask],y.loc[base_mask]
        xc,yc=x.loc[cal_mask],y.loc[cal_mask]
        xt,yt=x.loc[test_mask],y.loc[test_mask]
        if len(xt)==0 or len(xb)<30 or yb.nunique()<2:
            skipped+=1
            start+=test_dates_per_fold
            continue

        baseline=_base_rate(pd.concat([yb,yc]))
        model=_model(random_state+fold)
        model.fit(xb,yb)
        raw_test=model.predict_proba(xt)[:,1]
        probs=raw_test
        if len(xc)>=20 and yc.nunique()>1:
            raw_cal=model.predict_proba(xc)[:,1]
            logit=lambda z: np.log(np.clip(z,1e-6,1-1e-6)/(1-np.clip(z,1e-6,1-1e-6))).reshape(-1,1)
            calibrator=LogisticRegression(max_iter=1000,random_state=random_state+fold)
            calibrator.fit(logit(raw_cal),yc)
            probs=calibrator.predict_proba(logit(raw_test))[:,1]

        for idx,actual,prob in zip(xt.index,yt.to_numpy(),probs):
            rows.append({
                "date":idx[x.index.names.index("date")],
                "entity_id":idx[x.index.names.index("entity_id")],
                "fold":fold,
                "actual":int(actual),
                "probability":float(np.clip(prob,1e-6,1-1e-6)),
                "baseline_probability":baseline,
                "training_cutoff":train_end,
            })
        used+=1
        fold+=1
        start+=test_dates_per_fold

    predictions=pd.DataFrame(rows)
    if not predictions.empty:
        predictions=predictions.sort_values(["date","entity_id"]).reset_index(drop=True)
    return PanelBacktestResult(predictions,_metrics(predictions),used,skipped)
