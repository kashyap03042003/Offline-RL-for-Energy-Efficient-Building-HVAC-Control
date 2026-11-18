"""Off-policy evaluation (OPE): estimate a policy's value from logged data only.

In a real building there is no simulator to roll policies out in, so the value of a
candidate controller must be estimated from the BMS logs. Estimators:

* FQE    - Fitted Q Evaluation (Le et al., 2019): regress Q^pi with targets
           r + gamma * Q^pi(s', pi(s')); value = mean_s0 Q^pi(s0, pi(s0)).
* PDWIS  - per-decision weighted importance sampling with an estimated behaviour
           policy (BC softmax). The evaluated (deterministic) policy is epsilon-smoothed.
* WDR    - weighted doubly robust (Thomas & Brunskill, 2016), using FQE as the model.

Everything is in units of the environment reward with discount gamma, so estimates
can be compared with Monte Carlo discounted returns from the simulator.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .agents import mlp

T_DAY = 96


def policy_actions(policy, obs: np.ndarray, batch: int = 4000) -> np.ndarray:
    if hasattr(policy, "reset"):
        policy.reset()
    return np.concatenate([policy.act(obs[i:i + batch]) for i in range(0, len(obs), batch)])


def fqe(d: dict, pi_next: np.ndarray, pi_s0: np.ndarray, gamma: float, n_actions: int,
        steps: int = 30000, batch: int = 512, lr: float = 3e-4, reward_scale: float = 0.1,
        seed: int = 0, hidden: int = 256):
    """Returns (value estimate, q_fn) where q_fn(obs) -> (N, A) Q-values in reward units."""
    torch.manual_seed(seed)
    obs = torch.as_tensor(d["observations"]); nobs = torch.as_tensor(d["next_observations"])
    mean, std = obs.mean(0), obs.std(0) + 1e-3
    norm = lambda x: (x - mean) / std  # noqa: E731
    act = torch.as_tensor(d["actions"]); rew = torch.as_tensor(d["rewards"], dtype=torch.float32) * reward_scale
    a2 = torch.as_tensor(pi_next)
    q = mlp(obs.shape[1], n_actions, hidden)
    q_t = copy.deepcopy(q).requires_grad_(False)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    N = len(act)
    for step in range(steps):
        i = torch.randint(0, N, (batch,))
        with torch.no_grad():
            y = rew[i] + gamma * q_t(norm(nobs[i])).gather(1, a2[i, None]).squeeze(1)
        loss = F.smooth_l1_loss(q(norm(obs[i])).gather(1, act[i, None]).squeeze(1), y)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for tp, sp in zip(q_t.parameters(), q.parameters()):
                tp.mul_(0.995).add_(sp, alpha=0.005)

    @torch.no_grad()
    def q_fn(o: np.ndarray) -> np.ndarray:
        return q(norm(torch.as_tensor(o, dtype=torch.float32))).numpy() / reward_scale

    s0 = d["observations"][::T_DAY]
    v = q_fn(s0)[np.arange(len(s0)), pi_s0].mean()
    return float(v), q_fn


def _traj(x):
    return x.reshape(-1, T_DAY, *x.shape[1:])


def importance_estimators(d: dict, pi_a: np.ndarray, behavior_probs: np.ndarray, gamma: float,
                          q_fn=None, n_actions: int = 10, eps: float = 0.1) -> dict:
    """PDWIS and (if q_fn given) WDR. pi_a: evaluated policy's action at every logged state."""
    a = _traj(d["actions"]); r = _traj(d["rewards"].astype(np.float64)); pa = _traj(pi_a)
    pb = _traj(behavior_probs)                                         # (n, T, A)
    n, T = a.shape
    pe = np.where(a == pa, 1 - eps + eps / n_actions, eps / n_actions)  # smoothed target prob of logged action
    pb_a = np.take_along_axis(pb, a[..., None], 2).squeeze(2).clip(1e-3, 1)
    rho = pe / pb_a
    w = np.cumprod(rho, axis=1)                                        # (n, T)
    wn = w / w.sum(0, keepdims=True).clip(1e-300)
    disc = gamma ** np.arange(T)
    out = {"pdwis": float((disc * (wn * r).sum(0)).sum()),
           "ess_final": float(w[:, -1].sum() ** 2 / (w[:, -1] ** 2).sum().clip(1e-300))}
    if q_fn is not None:
        obs = d["observations"]
        q_all = _traj(q_fn(obs))                                       # (n, T, A)
        pe_full = np.full(q_all.shape, eps / n_actions)
        np.put_along_axis(pe_full, pa[..., None], 1 - eps + eps / n_actions, 2)
        v_hat = (pe_full * q_all).sum(2)                               # V^pi(s_t) under smoothed policy
        q_hat = np.take_along_axis(q_all, a[..., None], 2).squeeze(2)
        wn_prev = np.concatenate([np.full((n, 1), 1.0 / n), wn[:, :-1]], 1)
        out["wdr"] = float((disc * ((wn * r).sum(0) - (wn * q_hat).sum(0) + (wn_prev * v_hat).sum(0))).sum())
    return out
