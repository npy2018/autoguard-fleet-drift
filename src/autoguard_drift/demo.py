from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .monitor import FleetDriftMonitor
from .schemas import FleetObservation


def observation(timestamp: int, value: float, version: str = "V1") -> FleetObservation:
    return FleetObservation(
        timestamp=timestamp,
        vehicle_id=f"car-{timestamp % 20:02d}",
        version=version,
        metric_name="hard_brakes_per_1000km",
        metric_value=value,
        vehicle_model="A",
        city="Shenzhen",
        weather="rain",
        road_type="urban",
    )


def run(output: Path | None = None) -> dict[str, object]:
    rng = np.random.default_rng(11)
    baseline = [observation(i, float(rng.normal(0.8, 0.08))) for i in range(240)]
    monitor = FleetDriftMonitor()
    monitor.fit_baseline(baseline)
    assessments = []
    for i in range(60):
        value = float(rng.normal(0.82 if i < 25 else 1.25, 0.07))
        assessments.append(monitor.update(observation(1000 + i, value, "V2")))
    payload = {
        "last": assessments[-1].model_dump(),
        "peak_drift_probability": max(item.drift_probability for item in assessments),
        "recommendation": "pause rollout" if max(item.drift_probability for item in assessments[-10:]) >= 0.72 else "continue monitoring",
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
