#!/usr/bin/env bash
# Two complete synthetic pools. No training or cloud logging is started here.
set -uo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_DIR"
export PYTHONPATH="$(dirname "$PROJECT_DIR")${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline
PYTHON_BIN=${PYTHON_BIN:-/home/users/zsy/.conda/envs/neuralop/bin/python}
DATA_DIR=${DATA_DIR:-/home/users/zsy/mydata/budding_yeast}
POOL_ROOT=${POOL_ROOT:-$DATA_DIR/offline_pools}
GENERATION_SEED=${GENERATION_SEED:-42}
POOL_SAMPLES=${POOL_SAMPLES:-100000}
POOL_WORKERS=${POOL_WORKERS:-24}
LHS_CONFIG=${LHS_CONFIG:-$DATA_DIR/analysis/offline_pool_plan_20261112_exact_zero/mixed_lhs_domain.json}
mkdir -p "$POOL_ROOT"
run_pool() {
    local kind=$1
    local label=$kind
    if [[ $kind == lhs ]]; then label=lhs_matched; fi
    local output="$POOL_ROOT/${label}_${POOL_SAMPLES}_seed${GENERATION_SEED}"
    local log="$POOL_ROOT/${label}_${POOL_SAMPLES}_seed${GENERATION_SEED}.log"
    local attempt=1 code=0
    local args=(--kind "$kind" --output "$output" --samples "$POOL_SAMPLES"
                --seed "$GENERATION_SEED" --workers "$POOL_WORKERS" --block-size 256 --resume)
    if [[ $kind == lhs ]]; then args+=(--lhs-config "$LHS_CONFIG"); fi
    while (( attempt <= 3 )); do
        printf '[%s] Starting %s, attempt %s\n' "$(date '+%F %T')" "$kind" "$attempt" >> "$log"
        "$PYTHON_BIN" -m budding_yeast_v2.data.systems.budding.offline_pool "${args[@]}" >> "$log" 2>&1
        code=$?
        if (( code == 0 )); then return 0; fi
        printf '[%s] %s exited %s; durable progress retained\n' "$(date '+%F %T')" "$kind" "$code" >> "$log"
        attempt=$((attempt+1))
        if (( attempt <= 3 )); then sleep 5; fi
    done
    return "$code"
}
run_pool mutation & mutation_pid=$!
run_pool lhs & lhs_pid=$!
status=0
wait "$mutation_pid" || status=1
wait "$lhs_pid" || status=1
printf '[%s] Pool preparation finished with status %s\n' "$(date '+%F %T')" "$status"
exit "$status"
