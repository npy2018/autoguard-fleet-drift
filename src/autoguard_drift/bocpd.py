from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


@dataclass
class BayesianOnlineChangePoint:
    """Compact BOCPD with a normal likelihood and constant hazard."""

    hazard: float = 1 / 100
    prior_mean: float = 0.0
    prior_precision: float = 1.0
    observation_variance: float = 1.0
    max_run_length: int = 300
    run_probs: np.ndarray = field(init=False)
    means: np.ndarray = field(init=False)
    precisions: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.run_probs = np.array([1.0])
        self.means = np.array([self.prior_mean])
        self.precisions = np.array([self.prior_precision])

    def update(self, value: float) -> float:
        predictive_variance = self.observation_variance + 1.0 / self.precisions
        predictive = norm.pdf(value, loc=self.means, scale=np.sqrt(predictive_variance)) + 1e-15
        growth = self.run_probs * predictive * (1 - self.hazard)
        changepoint = float(np.sum(self.run_probs * predictive * self.hazard))
        new_probs = np.concatenate(([changepoint], growth))[: self.max_run_length]
        new_probs /= new_probs.sum()

        old_means = self.means[: len(new_probs) - 1]
        old_precisions = self.precisions[: len(new_probs) - 1]
        updated_precisions = old_precisions + 1.0 / self.observation_variance
        updated_means = (
            old_precisions * old_means + value / self.observation_variance
        ) / updated_precisions
        self.means = np.concatenate(([self.prior_mean], updated_means))
        self.precisions = np.concatenate(([self.prior_precision], updated_precisions))
        self.run_probs = new_probs
        return float(new_probs[0])
