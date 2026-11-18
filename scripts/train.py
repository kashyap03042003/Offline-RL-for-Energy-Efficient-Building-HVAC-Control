"""Train BC / CQL / IQL on the offline dataset.

Validation rollouts use held-out seeds (never the test seeds in evaluate_all.py),
and the FINAL checkpoint is what gets evaluated, so no test-set model selection."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702

import argparse
import json
import os
import time
from dataclasses import asdict, fields

import numpy as np
import pandas as pd
import torch

from hvac_orl.agents import AgentConfig, make_agent, make_batch_tensors
from hvac_orl.building import N_ACTIONS
from hvac_orl.data import load_dataset
from hvac_orl.evaluate import rollout, summarize
from hvac_orl.scenarios import TRAIN_CLIMATE

VAL_SEED = 50_000

ap = argparse.ArgumentParser()
ap.add_argument("--algo", choices=["bc", "cql", "iql", "lcql"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=100_000)
ap.add_argument("--eval-every", type=int, default=10_000)
ap.add_argument("--val-days", type=int, default=100)
ap.add_argument("--data", default="data/bms_dataset.npz")
ap.add_argument("--out-dir", default="runs")
ap.add_argument("--filter-top", type=float, default=1.0,
                help="keep only the top fraction of logged days by return (filtered BC, e.g. 0.1)")
ap.add_argument("--run-name", default=None)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--env", choices=["single", "multizone"], default="single")
for f in fields(AgentConfig):
    if f.name != "algo":
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.set_num_threads(args.threads)
cfg = AgentConfig(algo=args.algo, **{f.name: getattr(args, f.name) for f in fields(AgentConfig) if f.name != "algo"})

d = load_dataset(args.data)
if args.filter_top < 1.0:
    # Rank days *within* strata of (weekday/weekend x outdoor-temperature quintile); otherwise
    # "best days" are just mild weekends with no comfort constraint.
    day_ret = d["rewards"].reshape(-1, 96).sum(1)
    obs0 = d["observations"].reshape(-1, 96, d["observations"].shape[1])
    weekday = obs0[:, 0, 7] > 0.5
    t_out = obs0[:, :, 2].mean(1)
    stratum = weekday * 10 + np.digitize(t_out, np.quantile(t_out, [0.2, 0.4, 0.6, 0.8]))
    mask = np.zeros(len(day_ret), dtype=bool)
    for st in np.unique(stratum):
        ids = np.where(stratum == st)[0]
        mask[ids[np.argsort(day_ret[ids])[::-1][: max(1, int(len(ids) * args.filter_top))]]] = True
    mask = np.repeat(mask, 96)
    d = {k: v[mask] for k, v in d.items()}
    print(f"filtered to top {args.filter_top:.0%} days: {mask.sum()} transitions")
obs_mean, obs_std = d["observations"].mean(0), d["observations"].std(0) + 1e-3
T = make_batch_tensors(d, cfg)
N = len(T["act"])
agent = make_agent(cfg, d["observations"].shape[1], N_ACTIONS, obs_mean, obs_std)


def validate(agent):
    if args.env == "multizone":
        from hvac_orl.multizone import rollout_multizone
        df = rollout_multizone(agent, TRAIN_CLIMATE, args.val_days, seed=VAL_SEED)
        return {"energy_kwh_per_day": df.energy_kwh.mean(), "cost_per_day": df.cost.mean(),
                "comfort_compliance": df.comfort_ok_steps.sum() / df.occ_steps.sum(), "return": df.reward.mean()}
    return summarize(rollout(agent, TRAIN_CLIMATE, args.val_days, seed=VAL_SEED, gamma=cfg.gamma))

run_dir = os.path.join(args.out_dir, f"{args.run_name or args.algo}_seed{args.seed}")
os.makedirs(run_dir, exist_ok=True)
log, t0 = [], time.time()
for step in range(1, args.steps + 1):
    idx = torch.randint(0, N, (cfg.batch_size,))
    metrics = agent.update({k: v[idx] for k, v in T.items()})
    if step % args.eval_every == 0 or step == args.steps:
        agent.eval()
        val = validate(agent)
        agent.train()
        row = {"step": step, **metrics, **{f"val_{k}": v for k, v in val.items()}}
        log.append(row)
        print(f"[{args.algo} s{args.seed}] step {step:>7d} {time.time()-t0:6.0f}s "
              f"val E={val['energy_kwh_per_day']:.1f} $={val['cost_per_day']:.2f} "
              f"comfort={val['comfort_compliance']:.3f} R={val['return']:.2f}", flush=True)
agent.save(os.path.join(run_dir, "model.pt"))
pd.DataFrame(log).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)
with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump({**asdict(cfg), "seed": args.seed, "steps": args.steps, "filter_top": args.filter_top}, f, indent=2)
