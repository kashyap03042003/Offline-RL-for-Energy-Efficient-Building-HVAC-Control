#!/usr/bin/env bash
# Full reproducible pipeline. Hyperparameters were chosen on validation days only (scripts/sweep.sh).
# Runtime on a 10-core laptop CPU: roughly 2-3 hours.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-100000}
J=${J:-10}                                  # parallel training jobs

$PY -m pytest -q tests
$PY -I scripts/download_weather.py           # real TMYx weather (Boston, Chicago, Phoenix, Miami)
$PY scripts/generate_data.py --days 2000     # main offline dataset
$PY scripts/generate_multizone_data.py --days 1500

jobs_file=$(mktemp)
for s in $SEEDS; do
  T="$PY scripts/train.py --seed $s --steps $STEPS --eval-every 20000 --threads 1"
  echo "$T --algo bc"                                                          >> "$jobs_file"
  echo "$T --algo bc --filter-top 0.25 --run-name fbc"                         >> "$jobs_file"
  echo "$T --algo cql --cql-alpha 0.1 --n-step 4"                              >> "$jobs_file"
  echo "$T --algo iql ${IQL_ARGS:---beta 30.0 --expectile 0.7}"                                           >> "$jobs_file"
  echo "$T --algo lcql --cost-limit 0.003 --reward-scale 1.0 --n-step 4"       >> "$jobs_file"
  echo "$PY scripts/train_dynamics.py --seed $s --steps 30000 --threads 1"     >> "$jobs_file"
  # multi-zone (shared per-zone policy)
  echo "$T --env multizone --data data/bms_multizone.npz --out-dir runs_multizone --algo lcql --cost-limit 0.003 --reward-scale 1.0 --n-step 4" >> "$jobs_file"
  echo "$T --env multizone --data data/bms_multizone.npz --out-dir runs_multizone --algo cql --cql-alpha 0.1 --n-step 4" >> "$jobs_file"
done
xargs -P "$J" -I{} sh -c '{} > /dev/null' < "$jobs_file"
rm "$jobs_file"

$PY scripts/evaluate_all.py --days 500 --safe lcql,cql,iql
$PY scripts/make_report.py
$PY scripts/run_ope.py --run-dirs runs,runs_sweep3
$PY scripts/eval_multizone.py
bash scripts/run_ablations.sh
for p in lcql iql; do $PY scripts/explain.py --policy runs/${p}_seed0/model.pt --name $p; done
$PY scripts/explain.py --policy mpc:runs/dynamics_seed0/model.pt --name mpc
echo "done: see results/"
