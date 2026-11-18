"""Off-policy evaluation study.

For every candidate policy (all checkpoints found in the given run dirs + rule-based + MPC):
  * estimate its discounted value from the LOGGED DATA ONLY with FQE, PDWIS and WDR
  * compute the ground truth with Monte Carlo rollouts in the simulator
Then report how good each estimator is for *policy selection* (the realistic use case in
a building without a simulator): Spearman rank correlation, regret@1, and absolute error.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hvac_orl.agents import load_agent
from hvac_orl.building import N_ACTIONS
from hvac_orl.controllers import RuleBasedPolicy
from hvac_orl.data import load_dataset
from hvac_orl.evaluate import rollout
from hvac_orl.model_based import DynamicsEnsemble, MPCPolicy
from hvac_orl.ope import fqe, importance_estimators, policy_actions
from hvac_orl.scenarios import TRAIN_CLIMATE

ap = argparse.ArgumentParser()
ap.add_argument("--run-dirs", default="runs")
ap.add_argument("--data", default="data/bms_dataset.npz")
ap.add_argument("--behavior", default="runs/bc_seed0/model.pt", help="BC model used as behaviour-policy estimate")
ap.add_argument("--gamma", type=float, default=0.95)
ap.add_argument("--fqe-steps", type=int, default=20000)
ap.add_argument("--truth-days", type=int, default=1000)
ap.add_argument("--out", default="results/ope")
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)
os.makedirs(args.out, exist_ok=True)

d = load_dataset(args.data)
beh = load_agent(args.behavior)
pb = np.concatenate([beh.probs(d["observations"][i:i + 20000]) for i in range(0, len(d["actions"]), 20000)])
print(f"behaviour model: mean prob of logged action = {np.take_along_axis(pb, d['actions'][:, None], 1).mean():.3f}")

cands = [("rule_based", RuleBasedPolicy())]
for rd in args.run_dirs.split(","):
    for p in sorted(glob.glob(os.path.join(rd, "*_seed*", "model.pt"))):
        tag = f"{os.path.basename(rd)}/{os.path.basename(os.path.dirname(p))}"
        if "dynamics" in p:
            seed = int(re.search(r"seed(\d+)", p).group(1))
            iql = os.path.join(rd, f"iql_seed{seed}", "model.pt")
            cands.append((tag.replace("dynamics", "mpc"),
                          MPCPolicy(DynamicsEnsemble.load(p), None,
                                    comfort_margin=0.3)))
        else:
            cands.append((tag, load_agent(p)))
print(f"{len(cands)} candidate policies")

rows = []
for name, pol in cands:
    truth = rollout(pol, TRAIN_CLIMATE, args.truth_days, seed=123_456, gamma=args.gamma)["disc_return"].mean()
    pi_a = policy_actions(pol, d["observations"])
    pi_next = policy_actions(pol, d["next_observations"])
    v_fqe, q_fn = fqe(d, pi_next, pi_a[::96], args.gamma, N_ACTIONS, steps=args.fqe_steps)
    iw = importance_estimators(d, pi_a, pb, args.gamma, q_fn=q_fn)
    agree = float((pi_a == d["actions"]).mean())
    rows.append({"policy": name, "true_value": truth, "fqe": v_fqe, "pdwis": iw["pdwis"], "wdr": iw["wdr"],
                 "ess_final": iw["ess_final"], "agreement_with_data": agree})
    print(f"{name:34s} true={truth:8.2f}  FQE={v_fqe:8.2f}  PDWIS={iw['pdwis']:8.2f}  WDR={iw['wdr']:8.2f}  "
          f"ESS={iw['ess_final']:.1f}  agree={agree:.2f}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.out, "ope_estimates.csv"), index=False)

best_true = df.true_value.max()
summ = []
for est in ("fqe", "pdwis", "wdr"):
    pick = df.loc[df[est].idxmax()]
    summ.append({"estimator": est,
                 "spearman": df[[est, "true_value"]].rank().corr().iloc[0, 1],
                 "mae": (df[est] - df.true_value).abs().mean(),
                 "regret_at_1": best_true - pick.true_value,
                 "picked": pick.policy})
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(args.out, "ope_summary.csv"), index=False)
print(summ.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
lo, hi = df[["true_value", "fqe", "wdr"]].min().min(), df[["true_value", "fqe", "wdr"]].max().max()
for ax, est in zip(axes, ("fqe", "wdr", "pdwis")):
    ax.scatter(df.true_value, df[est], s=40, color="#2a78d6", edgecolor="#fcfcfb", lw=1.5, zorder=3)
    ax.plot([lo, hi], [lo, hi], color="#8a8985", lw=1, ls="--")
    r = summ.set_index("estimator").loc[est]
    ax.set_title(f"{est.upper()}  (Spearman {r.spearman:.2f}, regret@1 {r.regret_at_1:.2f})", loc="left", fontsize=10)
    ax.set_xlabel("True value (simulator Monte Carlo)"); ax.set_ylabel("Estimate from logged data")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "ope_scatter.png"), dpi=150); plt.close(fig)
