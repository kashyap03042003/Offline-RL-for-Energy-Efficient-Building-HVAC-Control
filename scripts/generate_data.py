"""Generate the offline BMS dataset from the behaviour policy."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702

import argparse
import json
import os

import numpy as np

from hvac_orl.data import generate_dataset
from hvac_orl.scenarios import TRAIN_CLIMATE

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=2000, help="logged building-days")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/bms_dataset.npz")
ap.add_argument("--mix", default="mixed", choices=["mixed", "expert", "random_heavy"])
args = ap.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
d = generate_dataset(TRAIN_CLIMATE, args.days, args.seed, mix=args.mix)
np.savez_compressed(args.out, **d)
n_days = args.days
stats = {
    "transitions": int(len(d["actions"])),
    "days": n_days,
    "mix": args.mix,
    "behavior_energy_kwh_per_day": float(d["energy_kwh"].sum() / n_days),
    "behavior_cost_per_day": float(d["cost"].sum() / n_days),
    "behavior_comfort_compliance": float(d["comfort_ok"].sum() / d["occupied"].sum()),
    "action_counts": np.bincount(d["actions"], minlength=10).tolist(),
}
with open(args.out.replace(".npz", "_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
