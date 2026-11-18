"""Train the dynamics ensemble used by MPC (model-based offline RL) and by the OOD detector."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702
import argparse
import os

import torch

from hvac_orl.data import load_dataset
from hvac_orl.model_based import train_dynamics

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/bms_dataset.npz")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--out-dir", default="runs")
ap.add_argument("--threads", type=int, default=2)
args = ap.parse_args()
torch.set_num_threads(args.threads)
m = train_dynamics(load_dataset(args.data), k=args.k, steps=args.steps, seed=args.seed)
os.makedirs(f"{args.out_dir}/dynamics_seed{args.seed}", exist_ok=True)
m.save(f"{args.out_dir}/dynamics_seed{args.seed}/model.pt")
