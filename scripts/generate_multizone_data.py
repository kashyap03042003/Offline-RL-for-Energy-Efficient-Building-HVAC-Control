"""Generate the offline dataset for the 3-zone building (one row per zone per step)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702
import argparse
import os

import numpy as np

from hvac_orl.multizone import generate_multizone_dataset
from hvac_orl.scenarios import TRAIN_CLIMATE

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=1500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/bms_multizone.npz")
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
d = generate_multizone_dataset(TRAIN_CLIMATE, args.days, args.seed)
np.savez_compressed(args.out, **d)
print(f"{len(d['actions'])} zone-transitions; behaviour comfort {d['comfort_ok'].sum() / d['occupied'].sum():.3f}")
