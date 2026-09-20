from datetime import datetime, timezone

import pytest

from preact.intelligence.risk import RiskDimension, RiskEstimate, RiskVector


def test_risk_vector_keeps_dimensions_separate() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    estimate = RiskEstimate(
        dimension=RiskDimension.INTERSTATE_CONFLICT,
        value=0.2,
        lower=0.1,
        upper=0.3,
        as_of=now,
        model_id="test",
        evidence_count=12,
    )
    vector = RiskVector("country:x", now, {estimate.dimension: estimate})
    assert vector.as_dict()["interstate_conflict"]["value"] == 0.2


def test_risk_estimate_rejects_invalid_probability() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        RiskEstimate(
            dimension=RiskDimension.POLITICAL_VIOLENCE,
            value=1.2,
            lower=None,
            upper=None,
            as_of=now,
            model_id="bad",
        )
