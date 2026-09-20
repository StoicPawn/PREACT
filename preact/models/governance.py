"""Evidence-based model promotion gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PromotionPolicy:
    min_oos_rows:int=500
    min_oos_events:int=30
    min_brier_skill:float=0.0
    max_abs_calibration_gap:float=0.05


@dataclass(frozen=True)
class PromotionDecision:
    promotable:bool
    checks:Mapping[str,bool]
    reasons:tuple[str,...]


def evaluate_promotion(metrics,policy:PromotionPolicy=PromotionPolicy())->PromotionDecision:
    checks={
        "enough_rows":int(metrics.rows)>=policy.min_oos_rows,
        "enough_events":int(metrics.events)>=policy.min_oos_events,
        "beats_base_rate":(
            metrics.brier_skill is not None
            and float(metrics.brier_skill)>policy.min_brier_skill
        ),
        "calibrated":(
            metrics.calibration_gap is not None
            and abs(float(metrics.calibration_gap))<=policy.max_abs_calibration_gap
        ),
    }
    reasons=tuple(name for name,passed in checks.items() if not passed)
    return PromotionDecision(all(checks.values()),checks,reasons)
