"""From-scratch PyTorch implementations of discrete-action offline RL agents.

* BC    - behaviour cloning (cross-entropy on logged actions); with --filter-top: filtered BC
* CQL   - Conservative Q-Learning, CQL(H) (Kumar et al., 2020) on a Double-DQN backbone,
          with an optional Q-ensemble (pessimistic min-over-ensemble targets) and n-step returns
* IQL   - Implicit Q-Learning (Kostrikov et al., 2021): expectile value regression,
          SARSA-style Q backup through V, advantage-weighted policy extraction
* LCQL  - Lagrangian (constrained) CQL: separate critics for energy cost and comfort
          violation; the policy maximises  Q_r - lambda * Q_c  and lambda is adapted by dual
          ascent so that the estimated per-step comfort violation stays below `cost_limit`.

Batches contain: obs, act, rew, next_obs, disc (= gamma^k * (1 - terminal), k = n-step length),
and for LCQL also cost (the constraint signal, n-step summed like rew).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, out_dim: int, hidden: int = 256, layers: int = 2, layer_norm: bool = True) -> nn.Sequential:
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden)] + ([nn.LayerNorm(hidden)] if layer_norm else []) + [nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class QEnsemble(nn.Module):
    """K independent Q-heads; forward returns (K, B, n_actions)."""

    def __init__(self, in_dim, n_actions, hidden, k):
        super().__init__()
        self.members = nn.ModuleList(mlp(in_dim, n_actions, hidden) for _ in range(k))

    def forward(self, x):
        return torch.stack([m(x) for m in self.members])


def polyak(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(sp, alpha=tau)


@dataclass
class AgentConfig:
    algo: str = "iql"
    gamma: float = 0.95
    lr: float = 3e-4
    hidden: int = 256
    batch_size: int = 256
    tau: float = 0.005            # target-network Polyak rate
    reward_scale: float = 0.1
    n_critics: int = 2            # Q-ensemble size (CQL / IQL / LCQL)
    n_step: int = 1
    # CQL / LCQL
    cql_alpha: float = 0.1
    # IQL
    expectile: float = 0.9
    beta: float = 30.0            # inverse temperature for advantage weights
    max_weight: float = 100.0
    # LCQL
    cost_limit: float = 0.002     # allowed mean comfort violation (°C) per step
    lambda_init: float = 2.0
    lambda_lr: float = 0.02
    lambda_max: float = 50.0


class Agent(nn.Module):
    """Common plumbing: observation normalisation, acting, save/load."""
    uses_cost = False

    def __init__(self, obs_dim: int, n_actions: int, cfg: AgentConfig, obs_mean: np.ndarray, obs_std: np.ndarray):
        super().__init__()
        self.cfg, self.obs_dim, self.n_actions = cfg, obs_dim, n_actions
        self.register_buffer("obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_std", torch.as_tensor(obs_std, dtype=torch.float32))

    def norm(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / self.obs_std

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_actions) action scores on normalised obs; argmax = greedy action."""
        raise NotImplementedError

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = self.norm(torch.as_tensor(obs, dtype=torch.float32))
        return self.scores(x).argmax(dim=1).numpy()

    @torch.no_grad()
    def action_scores(self, obs: np.ndarray) -> np.ndarray:
        return self.scores(self.norm(torch.as_tensor(obs, dtype=torch.float32))).numpy()

    def save(self, path: str, extra: dict | None = None):
        torch.save({"cfg": asdict(self.cfg), "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                    "state_dict": self.state_dict(), "extra": extra or {}}, path)


class BC(Agent):
    name = "bc"

    def __init__(self, obs_dim, n_actions, cfg, obs_mean, obs_std):
        super().__init__(obs_dim, n_actions, cfg, obs_mean, obs_std)
        self.pi = mlp(obs_dim, n_actions, cfg.hidden)
        self.opt = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr)

    def scores(self, x):
        return self.pi(x)

    @torch.no_grad()
    def probs(self, obs: np.ndarray) -> np.ndarray:
        return torch.softmax(self.scores(self.norm(torch.as_tensor(obs, dtype=torch.float32))), 1).numpy()

    def update(self, b: dict) -> dict:
        loss = F.cross_entropy(self.pi(self.norm(b["obs"])), b["act"])
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {"bc_loss": loss.item()}


class CQL(Agent):
    name = "cql"

    def __init__(self, obs_dim, n_actions, cfg, obs_mean, obs_std):
        super().__init__(obs_dim, n_actions, cfg, obs_mean, obs_std)
        self.q = QEnsemble(obs_dim, n_actions, cfg.hidden, cfg.n_critics)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)

    def scores(self, x):
        return self.q(x).mean(0)

    @torch.no_grad()
    def q_std(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        q = self.q(self.norm(torch.as_tensor(obs, dtype=torch.float32)))
        return q.std(0).gather(1, torch.as_tensor(actions)[:, None]).squeeze(1).numpy()

    def update(self, b: dict) -> dict:
        c = self.cfg
        s, s2, a = self.norm(b["obs"]), self.norm(b["next_obs"]), b["act"]
        q_all = self.q(s)                                             # (K, B, A)
        q_sa = q_all.gather(2, a[None, :, None].expand(len(q_all), -1, 1)).squeeze(2)
        with torch.no_grad():  # Double DQN: online-mean argmax, pessimistic min over target ensemble
            a2 = self.q(s2).mean(0).argmax(1)
            q2 = self.q_target(s2).gather(2, a2[None, :, None].expand(c.n_critics, -1, 1)).squeeze(2).min(0).values
            target = c.reward_scale * b["rew"] + b["disc"] * q2
        td = F.smooth_l1_loss(q_sa, target[None].expand_as(q_sa))
        conservative = (torch.logsumexp(q_all, dim=2) - q_sa).mean()
        loss = td + c.cql_alpha * conservative
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        polyak(self.q_target, self.q, c.tau)
        return {"td_loss": td.item(), "cql_gap": conservative.item(), "q_mean": q_sa.mean().item()}


class LCQL(Agent):
    """Constrained CQL with a learned Lagrange multiplier."""
    name = "lcql"
    uses_cost = True

    def __init__(self, obs_dim, n_actions, cfg, obs_mean, obs_std):
        super().__init__(obs_dim, n_actions, cfg, obs_mean, obs_std)
        self.qr = QEnsemble(obs_dim, n_actions, cfg.hidden, cfg.n_critics)   # energy-cost return (negative $)
        self.qc = QEnsemble(obs_dim, n_actions, cfg.hidden, cfg.n_critics)   # discounted comfort violation
        self.qr_t = copy.deepcopy(self.qr).requires_grad_(False)
        self.qc_t = copy.deepcopy(self.qc).requires_grad_(False)
        self.opt = torch.optim.Adam(list(self.qr.parameters()) + list(self.qc.parameters()), lr=cfg.lr)
        self.register_buffer("lam", torch.tensor(float(cfg.lambda_init)))

    def scores(self, x):
        return self.qr(x).mean(0) - self.lam * self.qc(x).mean(0)

    @torch.no_grad()
    def q_std(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = self.norm(torch.as_tensor(obs, dtype=torch.float32))
        q = self.qr(x) - self.lam * self.qc(x)
        return q.std(0).gather(1, torch.as_tensor(actions)[:, None]).squeeze(1).numpy()

    def update(self, b: dict) -> dict:
        c, K = self.cfg, self.cfg.n_critics
        s, s2, a = self.norm(b["obs"]), self.norm(b["next_obs"]), b["act"]
        idx = a[None, :, None].expand(K, -1, 1)
        qr_all, qc_all = self.qr(s), self.qc(s)
        qr_sa, qc_sa = qr_all.gather(2, idx).squeeze(2), qc_all.gather(2, idx).squeeze(2)
        with torch.no_grad():
            a2 = (self.qr(s2).mean(0) - self.lam * self.qc(s2).mean(0)).argmax(1)
            idx2 = a2[None, :, None].expand(K, -1, 1)
            qr2 = self.qr_t(s2).gather(2, idx2).squeeze(2).min(0).values      # pessimistic reward
            qc2 = self.qc_t(s2).gather(2, idx2).squeeze(2).max(0).values      # pessimistic cost
            tr = c.reward_scale * b["rew"] + b["disc"] * qr2
            tc = b["cost"] + b["disc"] * qc2
        td = F.smooth_l1_loss(qr_sa, tr[None].expand_as(qr_sa)) + F.smooth_l1_loss(qc_sa, tc[None].expand_as(qc_sa))
        # conservatism: push DOWN reward-Q and push UP cost-Q on actions not in the data
        cons_r = (torch.logsumexp(qr_all, 2) - qr_sa).mean()
        cons_c = (torch.logsumexp(-qc_all, 2) + qc_sa).mean()
        loss = td + c.cql_alpha * (cons_r + cons_c)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(self.qr.parameters()) + list(self.qc.parameters()), 10.0)
        self.opt.step()
        polyak(self.qr_t, self.qr, c.tau); polyak(self.qc_t, self.qc, c.tau)

        # dual ascent on lambda: per-step violation of the greedy policy, J_c ~ (1-gamma) Q_c(s, pi(s))
        with torch.no_grad():
            pi_a = (qr_all.mean(0) - self.lam * qc_all.mean(0)).argmax(1)
            jc = (1 - c.gamma) * qc_all.mean(0).gather(1, pi_a[:, None]).mean()
            self.lam.add_(c.lambda_lr * (jc - c.cost_limit) / max(c.cost_limit, 1e-6)).clamp_(0.0, c.lambda_max)
        return {"td_loss": td.item(), "jc": jc.item(), "lambda": self.lam.item(), "qr_mean": qr_sa.mean().item()}


class IQL(Agent):
    name = "iql"

    def __init__(self, obs_dim, n_actions, cfg, obs_mean, obs_std):
        super().__init__(obs_dim, n_actions, cfg, obs_mean, obs_std)
        self.q = QEnsemble(obs_dim, n_actions, cfg.hidden, cfg.n_critics)
        self.q_t = copy.deepcopy(self.q).requires_grad_(False)
        self.v = mlp(obs_dim, 1, cfg.hidden)
        self.pi = mlp(obs_dim, n_actions, cfg.hidden)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=cfg.lr)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr)

    def scores(self, x):
        return self.pi(x)

    @torch.no_grad()
    def q_std(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        q = self.q(self.norm(torch.as_tensor(obs, dtype=torch.float32)))
        return q.std(0).gather(1, torch.as_tensor(actions)[:, None]).squeeze(1).numpy()

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """V(s) in *unscaled* reward units (used as MPC terminal value)."""
        return self.v(self.norm(obs)).squeeze(-1) / self.cfg.reward_scale

    def update(self, b: dict) -> dict:
        c, K = self.cfg, self.cfg.n_critics
        s, s2, a = self.norm(b["obs"]), self.norm(b["next_obs"]), b["act"]
        idx = a[None, :, None].expand(K, -1, 1)
        with torch.no_grad():
            q_t = self.q_t(s).gather(2, idx).squeeze(2).min(0).values

        # 1) V: expectile regression towards in-sample Q (never queries unseen actions)
        v = self.v(s).squeeze(1)
        diff = q_t - v
        w = torch.where(diff > 0, c.expectile, 1 - c.expectile)
        v_loss = (w * diff.pow(2)).mean()
        self.v_opt.zero_grad(); v_loss.backward(); self.v_opt.step()

        # 2) Q: backup through V(s')
        with torch.no_grad():
            target = c.reward_scale * b["rew"] + b["disc"] * self.v(s2).squeeze(1)
        q_sa = self.q(s).gather(2, idx).squeeze(2)
        q_loss = F.mse_loss(q_sa, target[None].expand_as(q_sa)) * K
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()
        polyak(self.q_t, self.q, c.tau)

        # 3) Policy: advantage-weighted behaviour cloning
        adv = q_t - v.detach()
        weights = torch.exp(c.beta * adv).clamp(max=c.max_weight)
        pi_loss = (weights * F.cross_entropy(self.pi(s), a, reduction="none")).mean()
        self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()
        return {"v_loss": v_loss.item(), "q_loss": q_loss.item(), "pi_loss": pi_loss.item(),
                "adv_mean": adv.mean().item(), "q_mean": q_t.mean().item()}


ALGOS = {"bc": BC, "cql": CQL, "iql": IQL, "lcql": LCQL}


def make_agent(cfg: AgentConfig, obs_dim: int, n_actions: int, obs_mean, obs_std) -> Agent:
    return ALGOS[cfg.algo](obs_dim, n_actions, cfg, obs_mean, obs_std)


def load_agent(path: str) -> Agent:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    known = AgentConfig.__dataclass_fields__
    cfg = AgentConfig(**{k: v for k, v in ckpt["cfg"].items() if k in known})
    sd = ckpt["state_dict"]
    agent = make_agent(cfg, ckpt["obs_dim"], ckpt["n_actions"], sd["obs_mean"].numpy(), sd["obs_std"].numpy())
    agent.load_state_dict(sd)
    agent.eval()
    return agent


def make_batch_tensors(d: dict, cfg: AgentConfig) -> dict:
    """Convert a dataset dict to the tensors agents train on (handles n-step and LCQL's reward split)."""
    from .data import add_n_step
    if cfg.n_step > 1:
        d = add_n_step(d, cfg.n_step, cfg.gamma, keys=("rewards", "cost", "constraint_cost"))
        sfx, nxt, disc = "_n", d["next_observations_n"], d["discount_n"]
    else:
        sfx, nxt = "", d["next_observations"]
        disc = cfg.gamma * (1.0 - d["terminals"].astype(np.float32))
    T = {
        "obs": torch.as_tensor(d["observations"]), "next_obs": torch.as_tensor(nxt),
        "act": torch.as_tensor(d["actions"]), "disc": torch.as_tensor(disc, dtype=torch.float32),
    }
    if cfg.algo == "lcql":   # reward critic sees energy cost only; comfort goes to the constraint critic
        T["rew"] = -torch.as_tensor(d["cost" + sfx], dtype=torch.float32)
        T["cost"] = torch.as_tensor(d["constraint_cost" + sfx], dtype=torch.float32)
    else:
        T["rew"] = torch.as_tensor(d["rewards" + sfx], dtype=torch.float32)
    return T
