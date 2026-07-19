from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobustLocalForecaster:
    window: int = 30

    def forecast(self, history: list[float]) -> tuple[float, float]:
        values = np.asarray(history[-self.window :], dtype=float)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        return center, max(1.4826 * mad, 1e-3)


class ChronosForecaster:
    """Optional zero-shot foundation-model backend loaded only when requested."""

    def __init__(self, model_id: str = "amazon/chronos-bolt-tiny") -> None:
        try:
            from chronos import BaseChronosPipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the foundation extra: pip install -e '.[foundation]'") from exc
        self.pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")

    def forecast(self, history: list[float]) -> tuple[float, float]:  # pragma: no cover
        import torch

        context = torch.tensor(history, dtype=torch.float32)
        quantiles, mean = self.pipeline.predict_quantiles(
            inputs=context,
            prediction_length=1,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        median = float(quantiles[0, 0, 1])
        scale = max(float(quantiles[0, 0, 2] - quantiles[0, 0, 0]) / 2.56, 1e-3)
        return median, scale
