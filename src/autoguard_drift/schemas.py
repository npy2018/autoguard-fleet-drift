from __future__ import annotations

from pydantic import BaseModel, Field


class FleetObservation(BaseModel):
    timestamp: int
    vehicle_id: str
    version: str
    metric_name: str
    metric_value: float
    vehicle_model: str
    city: str
    weather: str
    road_type: str


class DriftAssessment(BaseModel):
    context_key: str
    observed_value: float
    expected_mean: float
    expected_scale: float
    z_score: float
    conformal_p_value: float
    changepoint_probability: float
    drift_probability: float
    sample_count: int
    status: str
    evidence: list[str] = Field(default_factory=list)
