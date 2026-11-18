"""Monte Carlo evaluation of every trained policy + the rule-based baseline on all
scenarios (in-distribution, synthetic & real-weather OOD, sensor issues, equipment faults).

Policies discovered in --runs:
  <name>_seed<k>/model.pt          offline-RL agents (bc, fbc, cql, iql, lcql, ...)
  dynamics_seed<k>/model.pt        -> "mpc" (learned-dynamics MPC)
  --safe a,b                       -> "safe_<a>" = agent <a> with OOD detection + rule-based fallback

Test seeds are disjoint from data / validation seeds. All policies see identical days."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702
import argparse
import glob
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from hvac_orl.agents import load_agent
from hvac_orl.controllers import RuleBasedPolicy
from hvac_orl.data import load_dataset
from hvac_orl.evaluate import rollout_scenario, summarize
from hvac_orl.model_based import DynamicsEnsemble, MPCPolicy
from hvac_orl.safety import SafePolicy, calibrate_threshold
from hvac_orl.scenarios import SCENARIOS

TEST_SEED = 900_000

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--data", default="data/bms_dataset.npz")
ap.add_argument("--days", type=int, default=500, help="Monte Carlo days per scenario")
ap.add_argument("--out", default="results")
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--safe", default="", help="comma-separated agent names to also evaluate with OOD fallback")
ap.add_argument("--safe-quantile", type=float, default=0.99)
ap.add_argument("--mpc-margin", type=float, default=0.3)
ap.add_argument("--scenarios", default="", help="comma-separated subset (default: all)")
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)
os.makedirs(args.out, exist_ok=True)

agents = {}
for path in sorted(glob.glob(os.path.join(args.runs, "*_seed*", "model.pt"))):
    name, seed = re.match(r"(.+)_seed(\d+)", os.path.basename(os.path.dirname(path))).groups()
    if name != "dynamics":
        agents[(name, int(seed))] = load_agent(path)
policies = [("rule_based", 0, RuleBasedPolicy())] + [(n, s, a) for (n, s), a in agents.items()]

dyn = {int(re.search(r"seed(\d+)", p).group(1)): DynamicsEnsemble.load(p)
       for p in sorted(glob.glob(os.path.join(args.runs, "dynamics_seed*", "model.pt")))}
for s, m in dyn.items():
    policies.append(("mpc", s, MPCPolicy(m, None, comfort_margin=args.mpc_margin)))  # no terminal value: best on validation

if args.safe and dyn:
    d = load_dataset(args.data)
    for s, m in dyn.items():
        thr = calibrate_threshold(m, d, args.safe_quantile)
        print(f"OOD threshold (dynamics seed {s}, q={args.safe_quantile}): {thr:.4f} °C")
        for name in args.safe.split(","):
            if (name, s) in agents:
                policies.append((f"safe_{name}", s, SafePolicy(agents[(name, s)], m, thr)))
print("policies:", sorted({p[0] for p in policies}))

scen_names = args.scenarios.split(",") if args.scenarios else list(SCENARIOS)
rows = []
for si, scen in enumerate(SCENARIOS):
    if scen not in scen_names:
        continue
    sc = SCENARIOS[scen]
    for name, seed, pol in policies:
        t0 = time.time()
        df = rollout_scenario(pol, sc, args.days, seed=TEST_SEED + 1000 * si)
        df["scenario"], df["group"], df["policy"], df["train_seed"] = scen, sc.group, name, seed
        rows.append(df)
        m = summarize(df)
        print(f"{scen:18s} {name:14s} s{seed}  E={m['energy_kwh_per_day']:6.1f}  $={m['cost_per_day']:6.2f}  "
              f"peak={m['peak_kw']:5.1f}  comfort={m['comfort_compliance']:.3f}  fb={m['fallback_rate']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
per_day = pd.concat(rows, ignore_index=True)
per_day.to_csv(os.path.join(args.out, "per_day.csv"), index=False)

# ---- summary: mean/std over training seeds, paired bootstrap CI over days ----
rng = np.random.default_rng(0)
out = []
for scen, g in per_day.groupby("scenario", sort=False):
    base = g[g.policy == "rule_based"].set_index("day")
    for name, gp in g.groupby("policy", sort=False):
        seed_stats = []
        for seed, gs in gp.groupby("train_seed"):
            s = summarize(gs)
            gs = gs.set_index("day").loc[base.index]
            s["energy_red"] = 1 - gs.energy_kwh.sum() / base.energy_kwh.sum()
            s["cost_red"] = 1 - gs.cost.sum() / base.cost.sum()
            s["peak_red"] = 1 - gs.peak_kw.sum() / base.peak_kw.sum()
            seed_stats.append(s)
        ss = pd.DataFrame(seed_stats)
        avg = gp.groupby("day")[["energy_kwh", "cost"]].mean().loc[base.index]
        idx = rng.integers(0, len(base), (args.n_boot, len(base)))
        e_boot = 1 - avg.energy_kwh.values[idx].sum(1) / base.energy_kwh.values[idx].sum(1)
        c_boot = 1 - avg.cost.values[idx].sum(1) / base.cost.values[idx].sum(1)
        out.append({
            "scenario": scen, "group": g.group.iloc[0], "policy": name, "n_seeds": len(ss),
            "energy_kwh_per_day": ss.energy_kwh_per_day.mean(), "cost_per_day": ss.cost_per_day.mean(),
            "peak_kw": ss.peak_kw.mean(), "peak_reduction": ss.peak_red.mean(),
            "comfort_compliance": ss.comfort_compliance.mean(), "comfort_compliance_std": ss.comfort_compliance.std(ddof=0),
            "viol_degh_per_day": ss.viol_degh_per_day.mean(), "fallback_rate": ss.fallback_rate.mean(),
            "energy_reduction": ss.energy_red.mean(), "energy_reduction_seed_std": ss.energy_red.std(ddof=0),
            "energy_reduction_ci_lo": np.percentile(e_boot, 2.5), "energy_reduction_ci_hi": np.percentile(e_boot, 97.5),
            "cost_reduction": ss.cost_red.mean(), "cost_reduction_seed_std": ss.cost_red.std(ddof=0),
            "cost_reduction_ci_lo": np.percentile(c_boot, 2.5), "cost_reduction_ci_hi": np.percentile(c_boot, 97.5),
        })
summary = pd.DataFrame(out)
summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

lines = ["| Scenario | Policy | Energy kWh/day | Energy red. | Cost $/day | Cost red. | Peak red. | Comfort | Fallback |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in summary.itertuples():
    pm = lambda m, s: f"{100 * m:+.1f}% ± {100 * s:.1f}" if r.n_seeds > 1 else f"{100 * m:+.1f}%"  # noqa: E731
    lines.append(f"| {r.scenario} | {r.policy} | {r.energy_kwh_per_day:.1f} | {pm(r.energy_reduction, r.energy_reduction_seed_std)} "
                 f"| {r.cost_per_day:.2f} | {pm(r.cost_reduction, r.cost_reduction_seed_std)} | {100 * r.peak_reduction:+.1f}% "
                 f"| {100 * r.comfort_compliance:.1f}% | {100 * r.fallback_rate:.1f}% |")
open(os.path.join(args.out, "summary.md"), "w").write(
    "Reductions are relative to the rule-based BMS schedule on identical days (positive = saves). "
    "± is std across training seeds. Cost includes energy (TOU) and demand charges.\n\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
