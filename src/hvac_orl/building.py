"""Batched single-zone office building simulator (2R2C thermal network).

State (hidden):  indoor air/furniture temperature T_in, envelope/wall temperature T_w.
Action:          discrete zone setpoint (or HVAC off). A P-controller with finite
                 heating/cooling capacity tracks the setpoint inside each 15-min step.
Energy:          heat-pump electricity = thermal / COP(T_out) + fan power when on.

    C_in dT_in/dt = (T_w - T_in)/R_iw + (T_out - T_in)/R_win + Q_hvac + Q_int + A_in * I_sol
    C_w  dT_w/dt  = (T_out - T_w)/R_wo + (T_in - T_w)/R_iw + A_w * I_sol

All arrays are batched over n_envs independent days, so Monte Carlo rollouts of
hundreds of days take well under a second.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .weather import DT_H, FORECAST_STEPS, STEPS_PER_DAY, tou_price

# Action 0 = HVAC off; others are zone setpoints in °C.
ACTION_SETPOINTS = np.array([np.nan, 17.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 27.0])
N_ACTIONS = len(ACTION_SETPOINTS)
COMFORT_OCC = (20.0, 24.0)       # occupied comfort band, °C
COMFORT_UNOCC = (15.0, 30.0)     # unoccupied building-protection band, °C
OCC_START_H, OCC_END_H = 7.0, 19.0

OBS_NAMES = [
    "T_in", "dT_in", "T_out", "solar_kw_m2", "hour_sin", "hour_cos", "occupied", "weekday",
    "hours_to_occ_start", "price", "price_next2h", "T_out_fc3h", "peak_kw_today",
]
OBS_DIM = len(OBS_NAMES)


def action_index(setpoint: float) -> int:
    return int(np.nanargmin(np.abs(ACTION_SETPOINTS - setpoint)))


@dataclass(frozen=True)
class BuildingParams:
    C_in: float = 3.0e6        # J/K
    C_w: float = 4.0e7         # J/K
    R_iw: float = 1 / 2000.0   # K/W
    R_wo: float = 1 / 400.0    # K/W
    R_win: float = 1 / 150.0   # K/W (windows + infiltration)
    q_int_occ: float = 8000.0  # W, people + equipment at full occupancy
    q_int_unocc: float = 1000.0
    a_sol_in: float = 12.0     # m², effective solar aperture to the zone
    a_sol_w: float = 25.0      # m², solar absorbed by the envelope
    q_heat_max: float = 40e3   # W thermal
    q_cool_max: float = 40e3
    kp: float = 30000.0        # W/K, setpoint-tracking gain
    fan_kw: float = 1.5
    cop_scale: float = 1.0     # <1 models degraded / fouled equipment
    substeps: int = 15


@dataclass(frozen=True)
class RewardConfig:
    occ_penalty: float = 2.0     # $ per °C of occupied comfort violation per step
    unocc_penalty: float = 0.5
    demand_rate: float = 0.5     # $ per kW of the day's peak 15-min demand (~$15/kW-month)


def cop_heating(t_out):
    return np.clip(2.0 + 0.08 * (t_out + 10.0), 1.5, 4.5)


def cop_cooling(t_out):
    return np.clip(5.5 - 0.1 * (t_out - 15.0), 2.0, 5.5)


class HVACEnv:
    """Vectorised environment: n_envs independent days, all reset/terminate together."""

    def __init__(self, climate, n_envs: int, seed: int,
                 params: BuildingParams = BuildingParams(),
                 reward_cfg: RewardConfig = RewardConfig(),
                 obs_noise_std: float = 0.0, obs_bias_T_in: float = 0.0):
        """climate: any weather source with .sample(n_days, n_steps, rng) (synthetic Climate or EPWWeather).
        obs_noise_std: Gaussian sensor noise (°C); obs_bias_T_in: constant indoor-sensor drift (°C)."""
        self.climate = climate
        self.obs_bias_T_in = obs_bias_T_in
        self.n = n_envs
        self.p = params
        self.rc = reward_cfg
        self.obs_noise_std = obs_noise_std
        self.rng = np.random.default_rng(seed)
        self.noise_rng = np.random.default_rng(seed + 7919)

    # ---- helpers -------------------------------------------------------
    def _hour(self, t):
        return (t * DT_H) % 24.0

    def _occupied(self, t):
        h = self._hour(t)
        return self.weekday & (h >= OCC_START_H) & (h < OCC_END_H) & (t < STEPS_PER_DAY)

    def _obs(self):
        t = self.t
        h = self._hour(t)
        hv = np.full(self.n, h)
        occ = self._occupied(t)
        to_start = np.where(self.weekday & (h < OCC_START_H), OCC_START_H - h, 12.0)
        price = tou_price(hv, self.weekday)
        price_next = np.mean([tou_price(np.full(self.n, self._hour(t + k)), self.weekday) for k in range(1, 9)], axis=0)
        t_in, t_out, t_fc = self.T_in + self.obs_bias_T_in, self.w["T_out"][:, t].copy(), self.w["T_out"][:, t + FORECAST_STEPS].copy()
        if self.obs_noise_std > 0:
            s = self.obs_noise_std
            t_in += self.noise_rng.normal(0, s, self.n)
        # 15-min change of the *measured* indoor temperature (exposes hidden thermal-mass drift)
        d_t_in = t_in - (self._last_meas if self.t > 0 else t_in)
        self._last_meas = t_in
        if self.obs_noise_std > 0:
            t_out += self.noise_rng.normal(0, s, self.n)
            t_fc += self.noise_rng.normal(0, 2 * s, self.n)
        return np.stack([
            t_in, d_t_in, t_out, self.w["solar"][:, t] / 1000.0,
            np.full(self.n, np.sin(2 * np.pi * h / 24)), np.full(self.n, np.cos(2 * np.pi * h / 24)),
            occ.astype(float), self.weekday.astype(float), to_start, price, price_next, t_fc, self.peak_kw,
        ], axis=1).astype(np.float32)

    # ---- API ------------------------------------------------------------
    def reset(self) -> np.ndarray:
        n_steps = STEPS_PER_DAY + 1 + FORECAST_STEPS
        self.w = self.climate.sample(self.n, n_steps, self.rng)
        self.weekday = self.rng.random(self.n) < 5 / 7
        self.occ_level = self.rng.uniform(0.7, 1.0, self.n)
        t_mean = self.w["T_out"][:, :STEPS_PER_DAY].mean(axis=1)
        self.T_in = np.clip(21.0 + 0.15 * (t_mean - 11.0) + self.rng.normal(0, 1.0, self.n), 14.0, 30.0)
        self.T_w = 0.83 * self.T_in + 0.17 * t_mean
        self.peak_kw = np.zeros(self.n)
        self.t = 0
        return self._obs()

    def step(self, actions: np.ndarray):
        p, t = self.p, self.t
        actions = np.asarray(actions, dtype=int)
        on = actions != 0
        sp = np.where(on, ACTION_SETPOINTS[np.where(on, actions, 1)], 0.0)
        t_out = self.w["T_out"][:, t]
        solar = self.w["solar"][:, t]
        q_int = np.where(self._occupied(t), self.occ_level * p.q_int_occ, p.q_int_unocc)

        dt = DT_H * 3600.0 / p.substeps
        e_heat = np.zeros(self.n)
        e_cool = np.zeros(self.n)
        T_in, T_w = self.T_in, self.T_w
        for _ in range(p.substeps):
            q = np.where(on, np.clip(p.kp * (sp - T_in), -p.q_cool_max, p.q_heat_max), 0.0)
            e_heat += np.maximum(q, 0) * dt
            e_cool += np.maximum(-q, 0) * dt
            d_in = ((T_w - T_in) / p.R_iw + (t_out - T_in) / p.R_win + q + q_int + p.a_sol_in * solar) / p.C_in
            d_w = ((t_out - T_w) / p.R_wo + (T_in - T_w) / p.R_iw + p.a_sol_w * solar) / p.C_w
            T_in = T_in + dt * d_in
            T_w = T_w + dt * d_w
        self.T_in, self.T_w = T_in, T_w

        energy = (e_heat / (p.cop_scale * cop_heating(t_out)) + e_cool / (p.cop_scale * cop_cooling(t_out))) / 3.6e6 \
            + on * p.fan_kw * DT_H
        energy_cost = energy * tou_price(np.full(self.n, self._hour(t)), self.weekday)
        kw = energy / DT_H
        demand_cost = self.rc.demand_rate * np.maximum(kw - self.peak_kw, 0.0)  # incremental => Markov w/ peak in obs
        self.peak_kw = np.maximum(self.peak_kw, kw)
        cost = energy_cost + demand_cost

        self.t = t + 1
        occ_next = self._occupied(self.t)
        lo = np.where(occ_next, COMFORT_OCC[0], COMFORT_UNOCC[0])
        hi = np.where(occ_next, COMFORT_OCC[1], COMFORT_UNOCC[1])
        viol = np.maximum(lo - T_in, 0) + np.maximum(T_in - hi, 0)
        penalty = np.where(occ_next, self.rc.occ_penalty, self.rc.unocc_penalty) * viol
        # constraint signal for constrained RL (°C of violation, unoccupied down-weighted)
        constraint_cost = viol * np.where(occ_next, 1.0, self.rc.unocc_penalty / self.rc.occ_penalty)
        reward = -(cost + penalty)

        done = self.t >= STEPS_PER_DAY
        info = {
            "energy_kwh": energy, "cost": cost, "energy_cost": energy_cost, "demand_cost": demand_cost,
            "constraint_cost": constraint_cost, "peak_kw": self.peak_kw.copy(), "occupied": occ_next,
            "comfort_ok": occ_next & (viol == 0), "viol_degh_occ": viol * occ_next * DT_H,
            "T_in": T_in.copy(), "T_out": t_out.copy(),
        }
        return self._obs(), reward.astype(np.float32), done, info
