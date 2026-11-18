"""Explainability for a learned controller.

1. Permutation importance: how often the policy's action changes when one observation
   feature is shuffled across states (higher = the policy relies on it more).
2. Distilled decision tree: a shallow tree fitted to imitate the policy on states it
   visits; reports fidelity (agreement with the policy) and prints human-readable rules.
3. Behaviour summary: when the policy runs HVAC vs. lets the zone float, by hour & season.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from hvac_orl.agents import load_agent
from hvac_orl.building import ACTION_SETPOINTS, OBS_NAMES, HVACEnv
from hvac_orl.controllers import RuleBasedPolicy
from hvac_orl.evaluate import rollout
from hvac_orl.model_based import DynamicsEnsemble, MPCPolicy
from hvac_orl.scenarios import TRAIN_CLIMATE

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True, help="path to agent model.pt, or 'mpc:<dynamics.pt>[:<iql.pt>]'")
ap.add_argument("--name", default=None)
ap.add_argument("--days", type=int, default=300)
ap.add_argument("--out", default="results/explain")
ap.add_argument("--depth", type=int, default=5)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

if args.policy.startswith("mpc:"):
    parts = args.policy.split(":")[1:]
    pol = MPCPolicy(DynamicsEnsemble.load(parts[0]), load_agent(parts[1]) if len(parts) > 1 else None, comfort_margin=0.3)
else:
    pol = load_agent(args.policy)
name = args.name or getattr(pol, "name", "policy")

# states the policy actually visits (on-policy distribution, held-out days)
df, traj = rollout(pol, TRAIN_CLIMATE, args.days, seed=424242, record_trajectories=True)
env_obs = []  # re-run the same days to collect the observations the policy saw
env = HVACEnv(TRAIN_CLIMATE, args.days, seed=424242)
o, done = env.reset(), False
while not done:
    env_obs.append(o)
    o, _, done, _ = env.step(pol.act(o))
X = np.concatenate(env_obs)
y = np.concatenate([pol.act(X[i:i + 5000]) for i in range(0, len(X), 5000)])

# 1) permutation importance
rng = np.random.default_rng(0)
sub = rng.choice(len(X), min(20000, len(X)), replace=False)
Xs, ys = X[sub], y[sub]
imp = {}
for j, f in enumerate(OBS_NAMES):
    Xp = Xs.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
    imp[f] = float((pol.act(Xp) != ys).mean())
imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

# 2) distilled tree
tr, te = sub[: len(sub) // 2], sub[len(sub) // 2:]
tree = DecisionTreeClassifier(max_depth=args.depth, min_samples_leaf=50, random_state=0).fit(X[tr], y[tr])
fidelity = float((tree.predict(X[te]) == y[te]).mean())
labels = ["OFF" if np.isnan(s) else f"SP{int(s)}" for s in ACTION_SETPOINTS]
rules = export_text(tree, feature_names=OBS_NAMES, class_names=[labels[c] for c in tree.classes_], decimals=2)

# 3) behaviour summary vs rule-based
hours = np.arange(96) * 0.25
A = np.array([t["action"] for t in traj])                           # (96, days)
cold, hot = df.mean_T_out.values < 8, df.mean_T_out.values > 18
wk = df.weekday.values

rb_tr = rollout(RuleBasedPolicy(), TRAIN_CLIMATE, args.days, seed=424242, record_trajectories=True)[1]
A_rb = np.array([t["action"] for t in rb_tr])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for k, (mask, title) in enumerate(((cold & wk, "Cold weekdays"), (hot & wk, "Hot weekdays"))):
    on_rate = (A[:, mask] != 0).mean(1)
    rb_on = (A_rb[:, mask] != 0).mean(1)
    ax[k].step(hours, rb_on, where="post", color="#2a78d6", lw=2, label="Rule-based")
    ax[k].step(hours, on_rate, where="post", color="#e87ba4", lw=2, label=name)
    ax[k].set_title(f"{title}: share of days with HVAC on", loc="left", fontsize=10)
    ax[k].set_xlabel("Hour of day"); ax[k].set_xticks(range(0, 25, 3))
    ax[k].spines[["top", "right"]].set_visible(False); ax[k].grid(alpha=0.3)
ax[0].set_ylabel("Fraction of days"); ax[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{args.out}/{name}_hvac_on_profile.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ks = list(imp)[::-1]
ax.barh(ks, [imp[k] for k in ks], color="#2a78d6")
ax.set_xlabel("Fraction of actions changed when feature is shuffled")
ax.set_title(f"{name}: permutation importance", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{args.out}/{name}_importance.png", dpi=150); plt.close(fig)

with open(f"{args.out}/{name}_rules.txt", "w") as f:
    f.write(f"Distilled decision tree (depth {args.depth}) for {name}; fidelity on held-out visited states = {fidelity:.3f}\n\n")
    f.write(rules)
json.dump({"policy": name, "permutation_importance": imp, "tree_fidelity": fidelity,
           "action_hist": np.bincount(y, minlength=len(ACTION_SETPOINTS)).tolist()},
          open(f"{args.out}/{name}_explain.json", "w"), indent=2)
print(json.dumps({"fidelity": fidelity, "top_features": list(imp.items())[:5]}, indent=2))
