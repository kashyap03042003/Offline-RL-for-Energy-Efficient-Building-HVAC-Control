"""Three-zone office (south perimeter, north perimeter, interior core) with inter-zone
heat exchange, one HVAC unit per zone and a building-level demand charge.

Control is decentralised with parameter sharing: every zone is one "row" with its own
observation (the single-zone features + zone one-hot + mean neighbour temperature offset),
and a single policy acts for all zones. Rewards are per zone (zone energy cost + its share
of the building demand-charge increment + zone comfort penalty); they sum to the building reward.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .building import (ACTION_SETPOINTS, COMFORT_OCC, COMFORT_UNOCC, OCC_END_H, OCC_START_H, OBS_NAMES,
                       RewardConfig, cop_cooling, cop_heating)
from .weather import DT_H, FORECAST_STEPS, STEPS_PER_DAY, tou_price

ZONES = ("south", "north", "core")
Z = len(ZONES)
MZ_OBS_NAMES = OBS_NAMES + [f"zone_{z}" for z in ZONES] + ["neighbor_dT"]
MZ_OBS_DIM = len(MZ_OBS_NAMES)


@dataclass(frozen=True)
class ZoneParams:
    C_in: tuple = (1.0e6, 1.0e6, 1.2e6)
    C_w: tuple = (1.6e7, 1.6e7, 0.6e7)
    R_iw: tuple = (1 / 700.0, 1 / 700.0, 1 / 600.0)
    R_wo: tuple = (1 / 170.0, 1 / 170.0, 1 / 20.0)      # core has almost no exterior envelope
    R_win: tuple = (1 / 60.0, 1 / 60.0, 1 / 10.0)
    solar_gain: tuple = (8.0, 2.0, 0.5)                  # m² effective aperture
    solar_wall: tuple = (12.0, 6.0, 0.0)
    q_int_occ: tuple = (2500.0, 2500.0, 3500.0)
    q_int_unocc: float = 300.0
    q_max: float = 15e3
    kp: float = 10000.0
    fan_kw: float = 0.5
    u_interzone: float = 250.0                           # W/K between each perimeter zone and the core
    substeps: int = 15


class MultiZoneEnv:
    def __init__(self, climate, n_envs: int, seed: int, zp: ZoneParams = ZoneParams(),
                 reward_cfg: RewardConfig = RewardConfig()):
        self.climate, self.n, self.zp, self.rc = climate, n_envs, zp, reward_cfg
        self.rng = np.random.default_rng(seed)
        self.P = {k: np.asarray(getattr(zp, k))[None, :] for k in
                  ("C_in", "C_w", "R_iw", "R_wo", "R_win", "solar_gain", "solar_wall", "q_int_occ")}

    def _hour(self, t):
        return (t * DT_H) % 24.0

    def _occupied(self, t):
        h = self._hour(t)
        return self.weekday & (h >= OCC_START_H) & (h < OCC_END_H) & (t < STEPS_PER_DAY)

    def _obs(self):
        t, n = self.t, self.n
        h = self._hour(t)
        occ = self._occupied(t)
        to_start = np.where(self.weekday & (h < OCC_START_H), OCC_START_H - h, 12.0)
        price = tou_price(np.full(n, h), self.weekday)
        price_next = np.mean([tou_price(np.full(n, self._hour(t + k)), self.weekday) for k in range(1, 9)], 0)
        d_t = self.T_in - (self._last if t > 0 else self.T_in)
        self._last = self.T_in.copy()
        shared = np.stack([self.w["T_out"][:, t], self.w["solar"][:, t] / 1000.0,
                           np.full(n, np.sin(2 * np.pi * h / 24)), np.full(n, np.cos(2 * np.pi * h / 24)),
                           occ, self.weekday, to_start, price, price_next,
                           self.w["T_out"][:, t + FORECAST_STEPS], self.peak_kw / Z], 1)      # (n, 11)
        neigh = np.stack([(self.T_in.sum(1) - self.T_in[:, z]) / (Z - 1) - self.T_in[:, z] for z in range(Z)], 1)
        rows = []
        for z in range(Z):
            onehot = np.zeros((n, Z)); onehot[:, z] = 1
            rows.append(np.concatenate([self.T_in[:, z:z + 1], d_t[:, z:z + 1], shared, onehot, neigh[:, z:z + 1]], 1))
        return np.stack(rows, 1).reshape(n * Z, -1).astype(np.float32)   # row = (day, zone)

    def reset(self):
        self.w = self.climate.sample(self.n, STEPS_PER_DAY + 1 + FORECAST_STEPS, self.rng)
        self.weekday = self.rng.random(self.n) < 5 / 7
        self.occ_level = self.rng.uniform(0.7, 1.0, (self.n, 1))
        t_mean = self.w["T_out"][:, :STEPS_PER_DAY].mean(1, keepdims=True)
        self.T_in = np.clip(21.0 + 0.15 * (t_mean - 11.0) + self.rng.normal(0, 1.0, (self.n, Z)), 14, 30)
        self.T_w = 0.8 * self.T_in + 0.2 * t_mean
        self.peak_kw = np.zeros(self.n)
        self.t = 0
        return self._obs()

    def step(self, actions):
        zp, P, t = self.zp, self.P, self.t
        a = np.asarray(actions, dtype=int).reshape(self.n, Z)
        on = a != 0
        sp = np.where(on, ACTION_SETPOINTS[np.where(on, a, 1)], 0.0)
        t_out = self.w["T_out"][:, t:t + 1]
        sol = self.w["solar"][:, t:t + 1]
        q_int = np.where(self._occupied(t)[:, None], self.occ_level * P["q_int_occ"], zp.q_int_unocc)
        dt = DT_H * 3600 / zp.substeps
        eh, ec = np.zeros((self.n, Z)), np.zeros((self.n, Z))
        T_in, T_w = self.T_in, self.T_w
        for _ in range(zp.substeps):
            q = np.where(on, np.clip(zp.kp * (sp - T_in), -zp.q_max, zp.q_max), 0.0)
            eh += np.maximum(q, 0) * dt; ec += np.maximum(-q, 0) * dt
            core = T_in[:, 2:3]
            inter = np.concatenate([zp.u_interzone * (core - T_in[:, :2]),
                                    zp.u_interzone * (T_in[:, :2] - core).sum(1, keepdims=True)], 1)
            d_in = ((T_w - T_in) / P["R_iw"] + (t_out - T_in) / P["R_win"] + q + q_int + P["solar_gain"] * sol + inter) / P["C_in"]
            d_w = ((t_out - T_w) / P["R_wo"] + (T_in - T_w) / P["R_iw"] + P["solar_wall"] * sol) / P["C_w"]
            T_in, T_w = T_in + dt * d_in, T_w + dt * d_w
        self.T_in, self.T_w = T_in, T_w
        e = (eh / cop_heating(t_out) + ec / cop_cooling(t_out)) / 3.6e6 + on * zp.fan_kw * DT_H   # (n, Z)
        price = tou_price(np.full(self.n, self._hour(t)), self.weekday)[:, None]
        e_cost = e * price
        kw_tot = e.sum(1) / DT_H
        dem = self.rc.demand_rate * np.maximum(kw_tot - self.peak_kw, 0)
        self.peak_kw = np.maximum(self.peak_kw, kw_tot)
        share = np.where(e.sum(1, keepdims=True) > 0, e / e.sum(1, keepdims=True).clip(1e-9), 1 / Z)
        cost = e_cost + share * dem[:, None]
        self.t = t + 1
        occ = self._occupied(self.t)[:, None] & np.ones((1, Z), bool)
        lo = np.where(occ, COMFORT_OCC[0], COMFORT_UNOCC[0]); hi = np.where(occ, COMFORT_OCC[1], COMFORT_UNOCC[1])
        viol = np.maximum(lo - T_in, 0) + np.maximum(T_in - hi, 0)
        reward = -(cost + np.where(occ, self.rc.occ_penalty, self.rc.unocc_penalty) * viol)
        flat = lambda x: x.reshape(-1)  # noqa: E731
        info = {"energy_kwh": flat(e), "cost": flat(cost), "occupied": flat(occ), "comfort_ok": flat(occ & (viol == 0)),
                "constraint_cost": flat(viol * np.where(occ, 1.0, self.rc.unocc_penalty / self.rc.occ_penalty)),
                "energy_cost": flat(e_cost), "demand_cost": flat(share * dem[:, None]), "T_in": flat(T_in)}
        return self._obs(), flat(reward).astype(np.float32), self.t >= STEPS_PER_DAY, info


def rollout_multizone(policy, climate, n_days: int, seed: int) -> pd.DataFrame:
    env = MultiZoneEnv(climate, n_days, seed)
    obs, done = env.reset(), False
    if hasattr(policy, "reset"):
        policy.reset()
    acc = {k: np.zeros(n_days * Z) for k in ("energy_kwh", "cost", "reward", "occ_steps", "comfort_ok_steps")}
    while not done:
        obs, r, done, info = env.step(policy.act(obs))
        acc["energy_kwh"] += info["energy_kwh"]; acc["cost"] += info["cost"]; acc["reward"] += r
        acc["occ_steps"] += info["occupied"]; acc["comfort_ok_steps"] += info["comfort_ok"]
    df = pd.DataFrame({k: v.reshape(n_days, Z).sum(1) for k, v in acc.items()})   # building totals per day
    df["peak_kw"] = env.peak_kw
    for z, name in enumerate(ZONES):
        df[f"comfort_{name}"] = acc["comfort_ok_steps"].reshape(n_days, Z)[:, z] / acc["occ_steps"].reshape(n_days, Z)[:, z].clip(1)
    return df


def generate_multizone_dataset(climate, n_days: int, seed: int, batch: int = 300) -> dict:
    from .controllers import BehaviorPolicy
    rng = np.random.default_rng(seed)
    keys = ["observations", "actions", "rewards", "next_observations", "terminals", "energy_kwh", "cost",
            "constraint_cost", "comfort_ok", "occupied"]
    buf = {k: [] for k in keys}
    done_days = 0
    while done_days < n_days:
        b = min(batch, n_days - done_days)
        env = MultiZoneEnv(climate, b, int(rng.integers(1 << 31)))
        pol = BehaviorPolicy(b * Z, np.random.default_rng(int(rng.integers(1 << 31))))
        obs, done = env.reset(), False
        pol.reset()
        ep = {k: [] for k in keys}
        while not done:
            a = pol.act(obs)
            nobs, r, done, info = env.step(a)
            for k, v in (("observations", obs), ("actions", a), ("rewards", r), ("next_observations", nobs),
                         ("terminals", np.zeros(b * Z, bool))):
                ep[k].append(v)
            for k in ("energy_kwh", "cost", "constraint_cost", "comfort_ok", "occupied"):
                ep[k].append(info[k])
            obs = nobs
        for k in keys:   # (T, rows, ...) -> rows-major so each (day, zone) is a contiguous 96-step block
            arr = np.stack(ep[k], 0)
            buf[k].append(np.swapaxes(arr, 0, 1).reshape(-1, *arr.shape[2:]))
        done_days += b
    out = {k: np.concatenate(v) for k, v in buf.items()}
    for k in ("observations", "next_observations", "rewards", "cost", "constraint_cost"):
        out[k] = out[k].astype(np.float32)
    out["actions"] = out["actions"].astype(np.int64)
    return out
