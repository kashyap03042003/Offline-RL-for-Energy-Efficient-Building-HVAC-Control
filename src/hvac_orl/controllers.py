"""Hand-written controllers: the rule-based BMS baseline and the behaviour policy
that generates the 'historical' BMS log used as the offline dataset."""
from __future__ import annotations

import numpy as np

from .building import N_ACTIONS, OBS_NAMES, action_index

IDX = {name: i for i, name in enumerate(OBS_NAMES)}


class RuleBasedPolicy:
    """Typical commercial BMS schedule: fixed occupied setpoint with a fixed
    pre-start, HVAC off overnight unless the zone drifts out of the protection band."""

    name = "rule_based"

    def __init__(self, occ_setpoint: float = 22.0, prestart_h: float = 1.0,
                 setback_low: float = 16.0, setback_high: float = 29.0):
        self.a_occ = action_index(occ_setpoint)
        self.prestart_h = prestart_h
        self.low, self.high = setback_low, setback_high
        self.a_heat, self.a_cool = action_index(17.0), action_index(27.0)

    def act(self, obs: np.ndarray) -> np.ndarray:
        t_in = obs[:, IDX["T_in"]]
        occ = obs[:, IDX["occupied"]] > 0.5
        weekday = obs[:, IDX["weekday"]] > 0.5
        to_start = obs[:, IDX["hours_to_occ_start"]]
        active = occ | (weekday & (to_start <= self.prestart_h))
        a = np.where(active, self.a_occ, 0)
        a = np.where(~active & (t_in < self.low), self.a_heat, a)
        a = np.where(~active & (t_in > self.high), self.a_cool, a)
        return a.astype(int)


class BehaviorPolicy:
    """Heterogeneous logging policy that mimics years of real BMS operation:
    * ~55% of days: fixed-setpoint schedule (setpoint / pre-start vary by facility manager)
    * ~30% of days: deadband ("free-float") mode with randomly chosen, often poorly
      tuned thresholds: HVAC off while the zone is inside [lo, hi], else heat/cool
    * ~15% of days: commissioning / test days with sticky random actions
    All modes also get sticky operator overrides."""

    def __init__(self, n_envs: int, rng: np.random.Generator, random_day_frac: float = 0.15,
                 deadband_frac: float = 0.30, override_max: float = 0.25):
        self.n, self.rng = n_envs, rng
        self.random_day_frac, self.deadband_frac, self.override_max = random_day_frac, deadband_frac, override_max

    def reset(self):
        r, n = self.rng, self.n
        u = r.random(n)
        self.random_day = u < self.random_day_frac
        self.deadband_day = (u >= self.random_day_frac) & (u < self.random_day_frac + self.deadband_frac)
        self.db_lo = r.uniform(20.2, 21.5, n)
        self.db_hi = r.uniform(22.5, 23.8, n)
        self.db_heat = np.array([action_index(x) for x in r.choice([21.0, 22.0], n)])
        self.db_cool = np.array([action_index(x) for x in r.choice([22.0, 23.0], n)])
        setpoints = r.choice([21.0, 22.0, 23.0], size=n, p=[0.25, 0.5, 0.25])
        prestarts = r.choice([0.5, 1.0, 1.5, 2.0], size=n)
        self.rule = RuleBasedPolicy()
        self.rule.a_occ = np.array([action_index(sp) for sp in setpoints])
        self.rule.prestart_h = prestarts
        self.override_prob = r.uniform(0.0, self.override_max, n)
        self.hold = np.zeros(n, dtype=int)
        self.held_action = np.zeros(n, dtype=int)

    def act(self, obs: np.ndarray) -> np.ndarray:
        rule_a = self.rule.act(obs)  # per-env setpoint / pre-start arrays broadcast
        t_in = obs[:, IDX["T_in"]]
        db_a = np.where(t_in < self.db_lo, self.db_heat, np.where(t_in > self.db_hi, self.db_cool, 0))
        rule_a = np.where(self.deadband_day & (rule_a == self.rule.a_occ), db_a, rule_a)
        start = (self.hold <= 0) & ((self.rng.random(self.n) < self.override_prob) | self.random_day)
        self.held_action = np.where(start, self.rng.integers(0, N_ACTIONS, self.n), self.held_action)
        self.hold = np.where(start, self.rng.integers(2, 9, self.n), self.hold - 1)
        return np.where(self.hold > 0, self.held_action, rule_a).astype(int)
