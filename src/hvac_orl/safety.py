"""OOD detection with fallback to the rule-based controller.

The detector scores each (state, proposed action) pair by the dynamics-ensemble
disagreement on the next indoor-temperature change: high disagreement means the
logged data never covered that situation, so the learned policy's choice is not
trustworthy there. The threshold is a high quantile of the score on the logged
(state, action) pairs, i.e. calibrated purely from offline data.
"""
from __future__ import annotations

import numpy as np

from .controllers import RuleBasedPolicy


class SafePolicy:
    def __init__(self, policy, detector, threshold: float, fallback=None, name: str | None = None):
        self.policy, self.detector, self.threshold = policy, detector, threshold
        self.fallback = fallback or RuleBasedPolicy()
        self.name = name or f"safe_{getattr(policy, 'name', 'policy')}"
        self.last_fallback = None

    def reset(self):
        for p in (self.policy, self.fallback):
            if hasattr(p, "reset"):
                p.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        a = self.policy.act(obs)
        score = self.detector.disagreement(obs, a)
        fb = score > self.threshold
        self.last_fallback = fb
        return np.where(fb, self.fallback.act(obs), a)


def calibrate_threshold(detector, dataset: dict, quantile: float = 0.99, max_n: int = 50000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset["actions"]), min(max_n, len(dataset["actions"])), replace=False)
    s = detector.disagreement(dataset["observations"][idx], dataset["actions"][idx])
    return float(np.quantile(s, quantile))
