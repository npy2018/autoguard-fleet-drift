from __future__ import annotations

from collections import deque


class AdaptiveConformalAlarm:
    def __init__(self, calibration_size: int = 200) -> None:
        self.residuals: deque[float] = deque(maxlen=calibration_size)

    def update(self, residual: float) -> float:
        score = abs(float(residual))
        if not self.residuals:
            p_value = 1.0
        else:
            greater = sum(item >= score for item in self.residuals)
            p_value = (greater + 1) / (len(self.residuals) + 1)
        self.residuals.append(score)
        return float(p_value)
