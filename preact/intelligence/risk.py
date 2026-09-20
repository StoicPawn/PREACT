"""Interpretable multidimensional risk representation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Mapping


class RiskDimension(str, Enum):
    INTERSTATE_CONFLICT = "interstate_conflict"
    INTERNAL_ARMED_CONFLICT = "internal_armed_conflict"
    POLITICAL_VIOLENCE = "political_violence"
    INSTITUTIONAL_STRESS = "institutional_stress"
    MACRO_FISCAL_STRESS = "macro_fiscal_stress"
    FINANCIAL_STRESS = "financial_stress"
    HUMANITARIAN_STRESS = "humanitarian_stress"
    SOCIAL_STRESS = "social_stress"
    DEMOGRAPHIC_STRESS = "demographic_stress"
    FOOD_ENERGY_STRESS = "food_energy_stress"
    CLIMATE_DISASTER_STRESS = "climate_disaster_stress"
    EXTERNAL_DEPENDENCY = "external_dependency"
    INFORMATION_ENVIRONMENT = "information_environment"


@dataclass(frozen=True)
class RiskEstimate:
    dimension: RiskDimension
    value: float
    lower: float | None
    upper: float | None
    as_of: datetime
    model_id: str
    calibration_id: str | None = None
    evidence_count: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.value) <= 1.0:
            raise ValueError("risk value must be in [0, 1]")
        if self.lower is not None and not 0.0 <= float(self.lower) <= 1.0:
            raise ValueError("lower bound must be in [0, 1]")
        if self.upper is not None and not 0.0 <= float(self.upper) <= 1.0:
            raise ValueError("upper bound must be in [0, 1]")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("lower bound cannot exceed upper bound")


@dataclass(frozen=True)
class RiskVector:
    entity_id: str
    as_of: datetime
    estimates: Mapping[RiskDimension, RiskEstimate]

    def get(self, dimension: RiskDimension) -> RiskEstimate | None:
        return self.estimates.get(dimension)

    def as_dict(self) -> dict[str, dict]:
        return {
            dimension.value: {
                "value": estimate.value,
                "lower": estimate.lower,
                "upper": estimate.upper,
                "model_id": estimate.model_id,
                "calibration_id": estimate.calibration_id,
                "evidence_count": estimate.evidence_count,
                "notes": estimate.notes,
            }
            for dimension, estimate in self.estimates.items()
        }
