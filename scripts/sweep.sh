#!/usr/bin/env bash
# Validation-only hyperparameter sweep (validation seed 50_000; test seeds are never used here).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=${OUT:-runs_sweep3}
STEPS=${STEPS:-80000}
mkdir -p $OUT
run() { name=$1; shift; $PY scripts/train.py --steps $STEPS --eval-every 20000 --threads 1 --out-dir $OUT --run-name $name "$@" > "$OUT/$name.log" 2>&1 && echo "done $name" || echo "FAILED $name"; }
run bc        --algo bc &
run fbc25     --algo bc --filter-top 0.25 &
run cql_n1    --algo cql --cql-alpha 0.1 &
run cql_n4    --algo cql --cql-alpha 0.1 --n-step 4 &
run cql_k5    --algo cql --cql-alpha 0.1 --n-critics 5 &
run iql_n1    --algo iql &
run iql_n4    --algo iql --n-step 4 &
run lcql_l1   --algo lcql --cost-limit 0.001 --reward-scale 1.0 &
run lcql_l3   --algo lcql --cost-limit 0.003 --reward-scale 1.0 &
run lcql_l3n4 --algo lcql --cost-limit 0.003 --reward-scale 1.0 --n-step 4 &
wait
$PY scripts/summarize_sweep.py $OUT
