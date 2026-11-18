"""Datasets for ablations: size subsets of the main log (first N days) and alternative behaviour mixes."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702
import os

import numpy as np

from hvac_orl.data import generate_dataset, load_dataset
from hvac_orl.scenarios import TRAIN_CLIMATE

os.makedirs("data/ablation", exist_ok=True)
d = load_dataset("data/bms_dataset.npz")
for n in (250, 500, 1000):
    np.savez_compressed(f"data/ablation/size{n}.npz", **{k: v[: n * 96] for k, v in d.items()})
for mix in ("expert", "random_heavy"):
    dm = generate_dataset(TRAIN_CLIMATE, 2000, seed=0, mix=mix)
    np.savez_compressed(f"data/ablation/mix_{mix}.npz", **dm)
    print(mix, "behaviour comfort", round(dm["comfort_ok"].sum() / dm["occupied"].sum(), 3),
          "energy/day", round(dm["energy_kwh"].sum() / 2000, 1))
print("ablation datasets written")
