from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CONTEXT_COLUMNS = ["vehicle_model", "city", "weather", "road_type"]


@dataclass
class ContextStats:
    mean: float
    scale: float
    count: int


class ContextualFingerprint:
    """Robust context baselines with hierarchical shrinkage for cold starts."""

    def __init__(self, shrinkage: float = 20.0) -> None:
        self.shrinkage = shrinkage
        self.global_stats: ContextStats | None = None
        self.contexts: dict[str, ContextStats] = {}

    @staticmethod
    def key(row: pd.Series | dict[str, object]) -> str:
        return "|".join(str(row[column]) for column in CONTEXT_COLUMNS)

    @staticmethod
    def _robust(values: np.ndarray) -> tuple[float, float]:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median, max(1.4826 * mad, 1e-3)

    def fit(self, frame: pd.DataFrame, value_column: str = "metric_value") -> "ContextualFingerprint":
        values = frame[value_column].to_numpy(dtype=float)
        global_mean, global_scale = self._robust(values)
        self.global_stats = ContextStats(global_mean, global_scale, len(values))
        self.contexts = {}
        for key, group in frame.groupby(CONTEXT_COLUMNS, dropna=False):
            local_values = group[value_column].to_numpy(dtype=float)
            local_mean, local_scale = self._robust(local_values)
            weight = len(local_values) / (len(local_values) + self.shrinkage)
            mean = weight * local_mean + (1 - weight) * global_mean
            scale = weight * local_scale + (1 - weight) * global_scale
            self.contexts["|".join(map(str, key))] = ContextStats(mean, max(scale, 1e-3), len(local_values))
        return self

    def predict(self, context: dict[str, object]) -> ContextStats:
        if self.global_stats is None:
            raise RuntimeError("fingerprint is not fitted")
        return self.contexts.get(self.key(context), self.global_stats)
