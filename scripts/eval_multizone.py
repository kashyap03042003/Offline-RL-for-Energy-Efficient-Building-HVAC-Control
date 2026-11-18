"""Evaluate shared per-zone policies on the 3-zone building vs the rule-based schedule (all zones)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702
import glob
import os
import re

import pandas as pd

from hvac_orl.agents import load_agent
from hvac_orl.controllers import RuleBasedPolicy
from hvac_orl.multizone import ZONES, rollout_multizone
from hvac_orl.scenarios import SCENARIOS

DAYS, SEED = 400, 900_000
SCEN = ["in_distribution", "heat_wave", "cold_snap", "real_boston", "real_phoenix"]
pols = [("rule_based", 0, RuleBasedPolicy())]
for p in sorted(glob.glob("runs_multizone/*_seed*/model.pt")):
    n, s = re.match(r"(.+)_seed(\d+)", os.path.basename(os.path.dirname(p))).groups()
    pols.append((n, int(s), load_agent(p)))
rows = []
for si, sc in enumerate(SCEN):
    if sc not in SCENARIOS:
        continue
    for name, seed, pol in pols:
        df = rollout_multizone(pol, SCENARIOS[sc].climate, DAYS, seed=SEED + 1000 * si)
        occ = df.occ_steps > 0
        rows.append({"scenario": sc, "policy": name, "seed": seed, "energy_kwh_per_day": df.energy_kwh.mean(),
                     "cost_per_day": df.cost.mean(), "peak_kw": df.peak_kw.mean(),
                     "comfort_compliance": df.comfort_ok_steps.sum() / df.occ_steps.sum(),
                     **{f"comfort_{z}": df[f"comfort_{z}"][occ].mean() for z in ZONES}})
        print(rows[-1], flush=True)
r = pd.DataFrame(rows)
g = r.groupby(["scenario", "policy"], sort=False).mean(numeric_only=True).reset_index()
base = g[g.policy == "rule_based"].set_index("scenario")
g["energy_reduction"] = 1 - g.energy_kwh_per_day / g.scenario.map(base.energy_kwh_per_day)
g["cost_reduction"] = 1 - g.cost_per_day / g.scenario.map(base.cost_per_day)
os.makedirs("results/multizone", exist_ok=True)
g.drop(columns="seed").to_csv("results/multizone/summary.csv", index=False)
print(g[["scenario", "policy", "energy_reduction", "cost_reduction", "comfort_compliance"] +
        [f"comfort_{z}" for z in ZONES]].round(3).to_string(index=False))
