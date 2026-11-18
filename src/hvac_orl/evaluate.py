"""Monte Carlo rollouts in the simulator. Every policy is evaluated on the same
sampled days (common random numbers), so comparisons are paired per day."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .building import ACTION_SETPOINTS, OBS_NAMES, HVACEnv


def rollout(policy, climate, n_days: int, seed: int, record_trajectories: bool = False,
            gamma: float = 0.95, **env_kwargs):
    """Run `policy` for n_days in parallel. env_kwargs go to HVACEnv (obs noise/bias, params).
    Pass a Scenario's .env_kwargs() to reproduce a scenario."""
    env = HVACEnv(climate, n_days, seed=seed, **env_kwargs)
    obs, done = env.reset(), False
    if hasattr(policy, "reset"):
        policy.reset()
    keys = ("energy_kwh", "cost", "energy_cost", "demand_cost", "reward", "disc_return", "occ_steps",
            "comfort_ok_steps", "viol_degh", "constraint_cost", "fallback_steps")
    acc = {k: np.zeros(n_days) for k in keys}
    traj, disc = [], 1.0
    while not done:
        a = policy.act(obs)
        nobs, r, done, info = env.step(a)
        for k in ("energy_kwh", "cost", "energy_cost", "demand_cost", "constraint_cost"):
            acc[k] += info[k]
        acc["reward"] += r
        acc["disc_return"] += disc * r
        disc *= gamma
        acc["occ_steps"] += info["occupied"]
        acc["comfort_ok_steps"] += info["comfort_ok"]
        acc["viol_degh"] += info["viol_degh_occ"]
        if getattr(policy, "last_fallback", None) is not None:
            acc["fallback_steps"] += policy.last_fallback
        if record_trajectories:
            traj.append({"t": env.t - 1, "T_in": info["T_in"], "T_out": info["T_out"], "action": a,
                         "setpoint": ACTION_SETPOINTS[a], "price": obs[:, OBS_NAMES.index("price")],
                         "occupied": info["occupied"], "energy_kwh": info["energy_kwh"]})
        obs = nobs
    df = pd.DataFrame(acc)
    df["peak_kw"] = env.peak_kw
    df["day"] = np.arange(n_days)
    df["weekday"] = env.weekday
    df["doy"] = env.w["doy"]
    df["mean_T_out"] = env.w["T_out"][:, :96].mean(axis=1)
    return (df, traj) if record_trajectories else df


def rollout_scenario(policy, scenario, n_days: int, seed: int, **kw):
    return rollout(policy, scenario.climate, n_days, seed, **scenario.env_kwargs(), **kw)


def summarize(df: pd.DataFrame) -> dict:
    occ = df["occ_steps"].sum()
    return {
        "energy_kwh_per_day": df["energy_kwh"].mean(),
        "cost_per_day": df["cost"].mean(),
        "demand_cost_per_day": df["demand_cost"].mean(),
        "peak_kw": df["peak_kw"].mean(),
        "comfort_compliance": df["comfort_ok_steps"].sum() / occ if occ > 0 else np.nan,
        "viol_degh_per_day": df["viol_degh"].mean(),
        "return": df["reward"].mean(),
        "fallback_rate": df["fallback_steps"].sum() / (96 * len(df)),
    }
