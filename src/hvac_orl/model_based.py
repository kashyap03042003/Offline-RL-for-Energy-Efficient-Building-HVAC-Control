"""Model-based offline control: a learned dynamics ensemble + uncertainty-penalised MPC.

DynamicsEnsemble  K bootstrap MLPs predicting, from (obs, action):
                    * change of indoor temperature over the next step
                    * electricity used in the step (kWh)
                  Trained only on the logged data. Ensemble disagreement = epistemic
                  uncertainty (also used for OOD detection in safety.py).

MPCPolicy         Enumerates two-segment setpoint plans (a1 held for h1 steps, then a2),
                  rolls each through the ensemble, and scores
                      sum_k gamma^k r_k  -  kappa * sum_k std_k  +  gamma^H V(s_H)
                  where r_k uses the known tariff, demand charge and comfort band, std_k is
                  the ensemble disagreement (MOPO-style pessimism) and V is an optional
                  learned terminal value (from IQL). Receding horizon: only a1 is applied.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .agents import mlp
from .building import COMFORT_OCC, COMFORT_UNOCC, N_ACTIONS, OBS_NAMES, OCC_END_H, OCC_START_H, RewardConfig
from .weather import DT_H, FORECAST_STEPS, tou_price

I = {n: i for i, n in enumerate(OBS_NAMES)}


class DynamicsEnsemble(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, k: int = 5, hidden: int = 128,
                 obs_mean=None, obs_std=None, y_mean=None, y_std=None):
        super().__init__()
        self.k, self.n_actions = k, n_actions
        self.members = nn.ModuleList(mlp(obs_dim + n_actions, 2, hidden, layers=2) for _ in range(k))
        for name, v, dim in (("obs_mean", obs_mean, obs_dim), ("obs_std", obs_std, obs_dim),
                             ("y_mean", y_mean, 2), ("y_std", y_std, 2)):
            self.register_buffer(name, torch.as_tensor(v if v is not None else np.zeros(dim) if "mean" in name
                                                       else np.ones(dim), dtype=torch.float32))

    def _inp(self, obs, act):
        return torch.cat([(obs - self.obs_mean) / self.obs_std, nn.functional.one_hot(act, self.n_actions).float()], -1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """(K, B, 2) predictions in physical units: [dT_in, energy_kwh]."""
        x = self._inp(obs, act)
        return torch.stack([m(x) for m in self.members]) * self.y_std + self.y_mean

    @torch.no_grad()
    def disagreement(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        """Std across members of the predicted indoor-temperature change (°C)."""
        p = self(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(act, dtype=torch.int64))
        return p[..., 0].std(0).numpy()

    def save(self, path):
        torch.save({"k": self.k, "n_actions": self.n_actions, "hidden": self.members[0][0].out_features,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        m = cls(sd["obs_mean"].shape[0], ck["n_actions"], ck["k"], ck["hidden"])
        m.load_state_dict(sd)
        return m.eval()


def train_dynamics(d: dict, k: int = 5, hidden: int = 128, steps: int = 30000, batch: int = 512,
                   lr: float = 1e-3, seed: int = 0, log_every: int = 5000) -> DynamicsEnsemble:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, act = d["observations"], d["actions"]
    y = np.stack([d["next_observations"][:, I["T_in"]] - obs[:, I["T_in"]], d["energy_kwh"]], 1).astype(np.float32)
    n = len(act)
    perm = rng.permutation(n)
    val, tr = perm[: n // 20], perm[n // 20:]
    model = DynamicsEnsemble(obs.shape[1], N_ACTIONS, k, hidden, obs.mean(0), obs.std(0) + 1e-3, y.mean(0), y.std(0) + 1e-6)
    O, A, Y = torch.as_tensor(obs), torch.as_tensor(act), torch.as_tensor(y)
    boot = [torch.as_tensor(rng.choice(tr, len(tr))) for _ in range(k)]     # bootstrap resample per member
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for step in range(1, steps + 1):
        loss = 0.0
        for j, mem in enumerate(model.members):
            idx = boot[j][torch.randint(0, len(tr), (batch,))]
            pred = mem(model._inp(O[idx], A[idx]))
            loss = loss + ((pred - (Y[idx] - model.y_mean) / model.y_std) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                vi = torch.as_tensor(val)
                p = model(O[vi], A[vi]).mean(0)
                rmse = ((p - Y[vi]) ** 2).mean(0).sqrt().numpy()
            print(f"[dynamics] step {step} train_loss={loss.item() / k:.4f} val RMSE dT={rmse[0]:.4f}°C "
                  f"energy={rmse[1]:.4f}kWh", flush=True)
    return model.eval()


def _hour(obs):
    return (np.arctan2(obs[:, I["hour_sin"]], obs[:, I["hour_cos"]]) * 24 / (2 * np.pi)) % 24


class MPCPolicy:
    name = "mpc"

    def __init__(self, model: DynamicsEnsemble, value_agent=None, h1: int = 4, h2: int = 4,
                 gamma: float = 0.95, kappa: float = 0.0, reward_cfg: RewardConfig = RewardConfig(),
                 second_actions=(0, 3, 4, 5, 6, 7), comfort_margin: float = 0.3):
        self.m, self.v, self.h1, self.h2 = model, value_agent, h1, h2
        self.margin = comfort_margin      # plan against a tightened band to absorb model error
        self.gamma, self.kappa, self.rc = gamma, kappa, reward_cfg
        plans = [(a1, a2) for a1 in range(N_ACTIONS) for a2 in sorted(set(second_actions) | {a1})]
        self.plans = np.array(plans)                                  # (C, 2)
        self.H = h1 + h2

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        B, C = len(obs), len(self.plans)
        o = np.repeat(obs.astype(np.float64), C, axis=0)              # (B*C, D)
        plan = np.tile(self.plans, (B, 1))
        h = _hour(o)
        weekday = o[:, I["weekday"]] > 0.5
        t_out0, fc = o[:, I["T_out"]].copy(), o[:, I["T_out_fc3h"]].copy()
        score = np.zeros(B * C)
        disc = 1.0
        for k in range(self.H):
            a = np.where(k < self.h1, plan[:, 0], plan[:, 1])
            pred = self.m(torch.as_tensor(o, dtype=torch.float32), torch.as_tensor(a, dtype=torch.int64)).numpy()
            d_t, e = pred[..., 0].mean(0), np.clip(pred[..., 1].mean(0), 0, None)
            unc = pred[..., 0].std(0)
            e = np.where(a == 0, 0.0, e)                              # HVAC off uses no energy (known)
            kw = e / DT_H
            peak = o[:, I["peak_kw_today"]]
            cost = e * tou_price(h, weekday) + self.rc.demand_rate * np.maximum(kw - peak, 0)
            # advance the observation with known schedule / tariff and interpolated weather
            h = (h + DT_H) % 24
            t_in = o[:, I["T_in"]] + d_t
            occ = weekday & (h >= OCC_START_H) & (h < OCC_END_H)
            lo = np.where(occ, COMFORT_OCC[0] + self.margin, COMFORT_UNOCC[0])
            hi = np.where(occ, COMFORT_OCC[1] - self.margin, COMFORT_UNOCC[1])
            viol = np.maximum(lo - t_in, 0) + np.maximum(t_in - hi, 0)
            r = -(cost + np.where(occ, self.rc.occ_penalty, self.rc.unocc_penalty) * viol) - self.kappa * unc
            score += disc * r
            disc *= self.gamma
            o = o.copy()
            o[:, I["T_in"]], o[:, I["dT_in"]] = t_in, d_t
            o[:, I["T_out"]] = t_out0 + (fc - t_out0) * (k + 1) / FORECAST_STEPS
            o[:, I["hour_sin"]], o[:, I["hour_cos"]] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
            o[:, I["occupied"]] = occ
            o[:, I["hours_to_occ_start"]] = np.where(weekday & (h < OCC_START_H), OCC_START_H - h, 12.0)
            o[:, I["price"]] = tou_price(h, weekday)
            o[:, I["price_next2h"]] = np.mean([tou_price((h + DT_H * j) % 24, weekday) for j in range(1, 9)], 0)
            o[:, I["peak_kw_today"]] = np.maximum(peak, kw)
        if self.v is not None:
            score += disc * self.v.value(torch.as_tensor(o, dtype=torch.float32)).numpy()
        best = score.reshape(B, C).argmax(1)
        return self.plans[best, 0].astype(int)
