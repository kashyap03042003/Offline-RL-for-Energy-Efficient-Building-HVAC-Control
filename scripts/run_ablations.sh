#!/usr/bin/env bash
# Dataset size (250/500/1000/2000 days) and behaviour-quality (expert / mixed / random-heavy) ablations, seed 0.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=${STEPS:-60000}
$PY scripts/make_ablation_data.py
jobs_file=$(mktemp)
for ds in size250 size500 size1000 mix_expert mix_random_heavy; do
  D=data/ablation/$ds.npz; O=runs_ablation/$ds
  T="$PY scripts/train.py --seed 0 --steps $STEPS --eval-every $STEPS --threads 1 --data $D --out-dir $O"
  echo "$T --algo cql --cql-alpha 0.1 --n-step 4"                        >> "$jobs_file"
  echo "$T --algo iql ${IQL_ARGS:---beta 30.0 --expectile 0.7}"                                     >> "$jobs_file"
  echo "$T --algo lcql --cost-limit 0.003 --reward-scale 1.0 --n-step 4" >> "$jobs_file"
  echo "$PY scripts/train_dynamics.py --seed 0 --steps 20000 --threads 1 --data $D --out-dir $O" >> "$jobs_file"
done
xargs -P ${J:-10} -I{} sh -c '{} > /dev/null' < "$jobs_file"
rm "$jobs_file"
# the full 2000-day 'mixed' dataset = main runs (seed 0)
mkdir -p runs_ablation/size2000
for a in cql iql lcql dynamics; do ln -sfn "$PWD/runs/${a}_seed0" "runs_ablation/size2000/${a}_seed0"; done
$PY scripts/eval_ablations.py
