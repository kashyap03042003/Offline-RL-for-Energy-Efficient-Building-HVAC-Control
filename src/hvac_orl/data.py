"""Generate the offline 'historical BMS' dataset by logging the behaviour policy."""
from __future__ import annotations

import numpy as np

from .building import HVACEnv
from .controllers import BehaviorPolicy

# Behaviour mixes for dataset-quality ablations
BEHAVIOR_MIXES = {
    "mixed": dict(random_day_frac=0.15, deadband_frac=0.30, override_max=0.25),        # default
    "expert": dict(random_day_frac=0.0, deadband_frac=0.30, override_max=0.03),        # clean schedules only
    "random_heavy": dict(random_day_frac=0.50, deadband_frac=0.20, override_max=0.40),
}
INFO_KEYS = ("energy_kwh", "cost", "energy_cost", "demand_cost", "constraint_cost", "comfort_ok", "occupied")


def generate_dataset(climate, n_days: int, seed: int, batch: int = 500, mix: str = "mixed") -> dict:
    rng = np.random.default_rng(seed)
    keys = ["observations", "actions", "rewards", "next_observations", "terminals", "timeouts", *INFO_KEYS]
    buf = {k: [] for k in keys}
    done_days = 0
    while done_days < n_days:
        b = min(batch, n_days - done_days)
        env = HVACEnv(climate, b, seed=int(rng.integers(1 << 31)))
        pol = BehaviorPolicy(b, np.random.default_rng(int(rng.integers(1 << 31))), **BEHAVIOR_MIXES[mix])
        obs, done = env.reset(), False
        pol.reset()
        ep = {k: [] for k in keys}
        while not done:
            a = pol.act(obs)
            nobs, r, done, info = env.step(a)
            ep["observations"].append(obs); ep["actions"].append(a); ep["rewards"].append(r)
            ep["next_observations"].append(nobs)
            # Days are time-limit truncations, not true terminals: we still bootstrap.
            ep["terminals"].append(np.zeros(b, dtype=bool)); ep["timeouts"].append(np.full(b, done))
            for k in INFO_KEYS:
                ep[k].append(info[k])
            obs = nobs
        for k in keys:  # (T, B, ...) -> (B*T, ...) keeping each day contiguous
            arr = np.stack(ep[k], axis=0)
            buf[k].append(np.swapaxes(arr, 0, 1).reshape(-1, *arr.shape[2:]))
        done_days += b
    out = {k: np.concatenate(v) for k, v in buf.items()}
    for k in ("observations", "next_observations", "rewards", "cost", "constraint_cost"):
        out[k] = out[k].astype(np.float32)
    out["actions"] = out["actions"].astype(np.int64)
    return out


def load_dataset(path: str) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def add_n_step(d: dict, n: int, gamma: float, keys=("rewards", "constraint_cost")) -> dict:
    """n-step returns within each logged day (days are contiguous blocks of 96).
    Adds '<key>_n', 'next_observations_n' and 'discount_n' (= gamma^k, k<=n truncated at day end)."""
    T = 96
    N = len(d["actions"])
    t_in_day = np.arange(N) % T
    out = dict(d)
    k_eff = np.minimum(n, T - t_in_day)                 # steps available before the day ends
    for key in keys:
        r = d[key].reshape(-1, T).astype(np.float64)
        acc = np.zeros_like(r)
        for j in range(n):
            shifted = np.zeros_like(r)
            shifted[:, : T - j] = r[:, j:]
            acc += (gamma ** j) * shifted               # entries beyond day end are zero-padded
        out[f"{key}_n"] = acc.reshape(-1).astype(np.float32)
    nxt_idx = np.arange(N) + k_eff - 1                  # transition whose next_obs is n steps ahead
    out["next_observations_n"] = d["next_observations"][nxt_idx]
    out["discount_n"] = (gamma ** k_eff).astype(np.float32)
    return out
