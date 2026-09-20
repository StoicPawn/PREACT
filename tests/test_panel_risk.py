import numpy as np
import pandas as pd

from preact.models.panel_risk import panel_walk_forward_backtest
from preact.models.governance import PromotionPolicy,evaluate_promotion

def test_panel_backtest_splits_by_date_and_purges_horizon():
    dates=pd.date_range("2020-01-01",periods=80,freq="D")
    entities=["a","b","c","d"]
    idx=pd.MultiIndex.from_product([dates,entities],names=["date","entity_id"])
    signal=np.tile(np.linspace(-1,1,len(dates)),len(entities)).reshape(len(entities),len(dates)).T.ravel()
    x=pd.DataFrame({"signal":signal,"noise":np.sin(np.arange(len(idx))/7)},index=idx)
    y=pd.Series((x["signal"]>0.4).astype(int),index=idx)
    result=panel_walk_forward_backtest(x,y,horizon_days=7,min_train_dates=25,test_dates_per_fold=5)
    assert result.folds_used>0
    assert not result.predictions.empty
    assert (result.predictions["training_cutoff"] <= result.predictions["date"]-pd.Timedelta(days=7)).all()

def test_promotion_gate_requires_real_oos_evidence():
    class M:
        rows=100
        events=5
        brier_skill=0.2
        calibration_gap=0.01
    decision=evaluate_promotion(M(),PromotionPolicy())
    assert decision.promotable is False
    assert "enough_rows" in decision.reasons
