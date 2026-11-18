"""Evaluate ablation runs (dataset size / behaviour quality) on in-distribution test days."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hvac_orl.agents import load_agent
from hvac_orl.controllers import RuleBasedPolicy
from hvac_orl.evaluate import rollout, summarize
from hvac_orl.model_based import DynamicsEnsemble, MPCPolicy
from hvac_orl.scenarios import TRAIN_CLIMATE

DAYS, SEED = 400, 900_000
base = summarize(rollout(RuleBasedPolicy(), TRAIN_CLIMATE, DAYS, seed=SEED))
rows = []
for ds_dir in sorted(glob.glob("runs_ablation/*")):
    ds = os.path.basename(ds_dir)
    for p in sorted(glob.glob(f"{ds_dir}/*_seed0/model.pt")):
        algo = re.match(r"(.+)_seed0", os.path.basename(os.path.dirname(p))).group(1)
        if algo == "dynamics":
            iql = f"{ds_dir}/iql_seed0/model.pt"
            pol, algo = MPCPolicy(DynamicsEnsemble.load(p), None,
                                  comfort_margin=0.3), "mpc"
        else:
            pol = load_agent(p)
        m = summarize(rollout(pol, TRAIN_CLIMATE, DAYS, seed=SEED))
        rows.append({"dataset": ds, "algo": algo, **m,
                     "energy_reduction": 1 - m["energy_kwh_per_day"] / base["energy_kwh_per_day"],
                     "cost_reduction": 1 - m["cost_per_day"] / base["cost_per_day"]})
        print(f"{ds:18s} {algo:6s} E-red={rows[-1]['energy_reduction']:+.3f} $-red={rows[-1]['cost_reduction']:+.3f} "
              f"comfort={m['comfort_compliance']:.3f}", flush=True)
df = pd.DataFrame(rows)
os.makedirs("results/ablation", exist_ok=True)
df.to_csv("results/ablation/ablation.csv", index=False)

COL = {"cql": "#eda100", "iql": "#e87ba4", "lcql": "#6250d6", "mpc": "#008300"}
sizes = df[df.dataset.str.startswith("size")].copy()
sizes["days"] = sizes.dataset.str.replace("size", "").astype(int)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for algo, g in sizes.groupby("algo"):
    g = g.sort_values("days")
    ax[0].plot(g.days, 100 * g.cost_reduction, "-o", color=COL.get(algo, "#52514e"), lw=2, ms=6, label=algo.upper())
    ax[1].plot(g.days, 100 * g.comfort_compliance, "-o", color=COL.get(algo, "#52514e"), lw=2, ms=6)
ax[0].set_title("Cost reduction vs rule-based (%)", loc="left", fontsize=10)
ax[1].set_title("Comfort compliance (%)", loc="left", fontsize=10)
for a in ax:
    a.set_xscale("log"); a.set_xticks([250, 500, 1000, 2000], ["250", "500", "1000", "2000"])
    a.set_xlabel("Logged building-days"); a.grid(alpha=0.3); a.spines[["top", "right"]].set_visible(False)
ax[0].axhline(0, color="#52514e", lw=0.8); ax[0].legend(frameon=False)
fig.tight_layout(); fig.savefig("results/ablation/dataset_size.png", dpi=150); plt.close(fig)
print(df.pivot_table(index="dataset", columns="algo", values=["cost_reduction", "comfort_compliance"]).round(3))
