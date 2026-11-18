"""Figures: reductions & comfort per scenario, example day trajectories, learning curves."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hvac_orl.agents import load_agent
from hvac_orl.building import COMFORT_OCC
from hvac_orl.controllers import RuleBasedPolicy
from hvac_orl.evaluate import rollout
from hvac_orl.scenarios import SCENARIOS

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--runs", default="runs")
args = ap.parse_args()
fig_dir = os.path.join(args.results, "figures"); os.makedirs(fig_dir, exist_ok=True)

# Fixed categorical order (validated reference palette); color follows the policy, never its rank.
COLORS = {"rule_based": "#2a78d6", "cql": "#eb6834", "iql": "#1baf7a", "lcql": "#eda100", "mpc": "#e87ba4",
          "safe_lcql": "#008300", "bc": "#6250d6", "fbc": "#e34948"}
LABELS = {"rule_based": "Rule-based", "cql": "CQL", "iql": "IQL", "lcql": "Lagrangian CQL", "mpc": "Model-based MPC",
          "safe_lcql": "Lagrangian CQL + OOD fallback", "bc": "BC", "fbc": "Filtered BC"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})

s = pd.read_csv(os.path.join(args.results, "summary.csv"))
scen = [x for x in SCENARIOS if x in set(s.scenario)]
# BC / filtered BC are reported in the table; their comfort failures would swamp the axes here
main = [p for p in ("cql", "iql", "lcql", "safe_lcql", "mpc") if p in set(s.policy)]

# 1) Dot plot: scenarios on y, one marker per method, three metrics side by side
fig, axes = plt.subplots(1, 3, figsize=(14, 0.42 * len(scen) + 1.8), sharey=True)
y = np.arange(len(scen))[::-1]
specs = [("energy_reduction", "Energy reduction vs rule-based (%)", main),
         ("cost_reduction", "Cost reduction vs rule-based (%)", main),
         ("comfort_compliance", "Occupied comfort compliance (%)", ["rule_based"] + main)]
for ax, (col, title, pols) in zip(axes, specs):
    off = np.linspace(-0.28, 0.28, len(pols))
    for i, p in enumerate(pols):
        d = s[s.policy == p].set_index("scenario").reindex(scen)
        err = d.get(col + "_seed_std", d.get(col + "_std"))
        ax.errorbar(100 * d[col], y + off[i], xerr=None if err is None else 100 * err, fmt="o", ms=6,
                    color=COLORS[p], ecolor=COLORS[p], elinewidth=1, mec=SURFACE, mew=1, label=LABELS[p])
    ax.set_title(title, loc="left", fontsize=10.5)
    if col != "comfort_compliance":
        ax.axvline(0, color=MUTED, lw=0.8)
    else:
        ax.axvline(99, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(max(80, 100 * s[s.policy.isin(pols)][col].min() - 1), 100.3)
axes[0].set_yticks(y, [x.replace("_", " ") for x in scen])
for k in range(len(scen) - 1):
    for ax in axes:
        ax.axhline(y[k] - 0.5, color=GRID, lw=0.6)
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, ncol=len(handles), loc="upper left", bbox_to_anchor=(0.01, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(fig_dir, "scenario_results.png"), dpi=160); plt.close(fig)

# 2) Example days: winter & summer weekday, rule-based vs Lagrangian CQL vs MPC (seed 0)
climate = SCENARIOS["in_distribution"].climate
pols = {"rule_based": RuleBasedPolicy()}
if os.path.exists(f"{args.runs}/lcql_seed0/model.pt"):
    pols["lcql"] = load_agent(f"{args.runs}/lcql_seed0/model.pt")
if os.path.exists(f"{args.runs}/dynamics_seed0/model.pt"):
    from hvac_orl.model_based import DynamicsEnsemble, MPCPolicy
    pols["mpc"] = MPCPolicy(DynamicsEnsemble.load(f"{args.runs}/dynamics_seed0/model.pt"))
trajs = {p: rollout(pol, climate, 400, seed=777, record_trajectories=True) for p, pol in pols.items()}
meta = trajs["rule_based"][0]
pick = {"Winter weekday": meta[(meta.weekday) & (meta.mean_T_out < 2)].day.iloc[0],
        "Summer weekday": meta[(meta.weekday) & (meta.mean_T_out > 22)].day.iloc[0]}
fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
hours = np.arange(96) * 0.25 + 0.25
for j, (title, day) in enumerate(pick.items()):
    ax, ax2 = axes[0, j], axes[1, j]
    ax.axhspan(*COMFORT_OCC, xmin=7 / 24, xmax=19 / 24, color=GRID, alpha=0.8, lw=0, label="Occupied comfort band")
    for p, (_, tr) in trajs.items():
        ax.plot(hours, [x["T_in"][day] for x in tr], color=COLORS[p], lw=2, label=LABELS[p])
        ax2.step(hours, np.cumsum([x["energy_kwh"][day] for x in tr]), where="post", color=COLORS[p], lw=2)
    t_out = np.mean([x["T_out"][day] for x in trajs["rule_based"][1]])
    ax.set_title(f"{title}  (mean outdoor {t_out:.1f} °C)", loc="left", fontsize=11)
    ax.set_ylabel("Indoor temperature (°C)"); ax2.set_ylabel("Cumulative energy (kWh)")
    ax2.set_xlabel("Hour of day"); ax2.set_xticks(range(0, 25, 3))
axes[0, 0].legend(loc="lower left", ncol=2, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "example_days.png"), dpi=160); plt.close(fig)

# 3) Learning curves (validation return, held-out seeds)
fig, ax = plt.subplots(figsize=(8, 4))
for p in ("bc", "fbc", "cql", "iql", "lcql"):
    logs = [pd.read_csv(f) for f in glob.glob(f"{args.runs}/{p}_seed*/train_log.csv")]
    if not logs:
        continue
    m = pd.concat(logs).groupby("step").val_return.agg(["mean", "min", "max"])
    ax.plot(m.index, m["mean"], color=COLORS[p], lw=2, label=LABELS[p])
    ax.fill_between(m.index, m["min"], m["max"], color=COLORS[p], alpha=0.15, lw=0)
from hvac_orl.evaluate import summarize
rb = summarize(rollout(RuleBasedPolicy(), climate, 100, seed=50_000))["return"]  # same validation days as training
ax.axhline(rb, color=COLORS["rule_based"], lw=1.5, ls="--", label="Rule-based (val days)")
ax.set_title("Validation return per day (held-out days; band = min–max over seeds)", loc="left", fontsize=11)
ax.set_xlabel("Gradient steps"); ax.set_ylabel("Return"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "learning_curves.png"), dpi=160); plt.close(fig)
print("figures written to", fig_dir)
