"""Print validation curves for every run in a sweep directory, plus the rule-based reference."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702
import glob
import os

import pandas as pd

from hvac_orl.controllers import RuleBasedPolicy
from hvac_orl.evaluate import rollout, summarize
from hvac_orl.scenarios import TRAIN_CLIMATE

out = sys.argv[1]
rb = summarize(rollout(RuleBasedPolicy(), TRAIN_CLIMATE, 100, seed=50_000))
print(f"{'rule_based':14s} E={rb['energy_kwh_per_day']:.1f} $={rb['cost_per_day']:.2f} c={rb['comfort_compliance']:.3f} R={rb['return']:.2f}")
for f in sorted(glob.glob(f"{out}/*/train_log.csv")):
    d = pd.read_csv(f)
    name = os.path.basename(os.path.dirname(f)).replace("_seed0", "")
    print(f"{name:14s}", " | ".join(f"{r.step // 1000}k E={r.val_energy_kwh_per_day:.1f} $={r.val_cost_per_day:.2f} "
                                    f"c={r.val_comfort_compliance:.3f} R={r.val_return:.2f}" for r in d.itertuples()))
