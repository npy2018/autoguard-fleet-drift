from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .bocpd import BayesianOnlineChangePoint
from .conformal import AdaptiveConformalAlarm
from .fingerprint import ContextualFingerprint
from .schemas import DriftAssessment, FleetObservation


class FleetDriftMonitor:
    def __init__(self) -> None:
        self.fingerprint = ContextualFingerprint()
        self.change_models: dict[str, BayesianOnlineChangePoint] = {}
        self.conformal: dict[str, AdaptiveConformalAlarm] = defaultdict(AdaptiveConformalAlarm)
        self.counts: dict[str, int] = defaultdict(int)

    def fit_baseline(self, observations: list[FleetObservation]) -> None:
        frame = pd.DataFrame([item.model_dump() for item in observations])
        self.fingerprint.fit(frame)

    def update(self, item: FleetObservation) -> DriftAssessment:
        context = item.model_dump()
        key = self.fingerprint.key(context)
        stats = self.fingerprint.predict(context)
        residual = (item.metric_value - stats.mean) / stats.scale
        p_value = self.conformal[key].update(residual)
        model = self.change_models.setdefault(
            key,
            BayesianOnlineChangePoint(
                prior_mean=0.0,
                prior_precision=1.0,
                observation_variance=1.0,
                hazard=1 / 60,
            ),
        )
        cp = model.update(residual)
        z_component = 1 / (1 + np.exp(-(abs(residual) - 2.5)))
        drift = float(np.clip(0.5 * z_component + 0.3 * (1 - p_value) + 0.2 * min(cp * 10, 1), 0, 1))
        self.counts[key] += 1
        status = "shadow-mode"
        if self.counts[key] >= 20:
            status = "alert" if drift >= 0.68 else "monitor"
        return DriftAssessment(
            context_key=key,
            observed_value=item.metric_value,
            expected_mean=round(stats.mean, 5),
            expected_scale=round(stats.scale, 5),
            z_score=round(float(residual), 4),
            conformal_p_value=round(p_value, 4),
            changepoint_probability=round(cp, 4),
            drift_probability=round(drift, 4),
            sample_count=self.counts[key],
            status=status,
            evidence=[
                f"fleet://context/{key}",
                f"fleet://version/{item.version}/metric/{item.metric_name}/t/{item.timestamp}",
            ],
        )
