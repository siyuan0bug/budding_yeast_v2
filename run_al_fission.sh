#!/usr/bin/env bash
# Fission AL: reuse a native trajectory initial dataset and a warmed input pool.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(dirname -- "$TASK_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
: "${INIT_DATASET:?Set INIT_DATASET to a Fission trajectory NPZ}"
: "${POOL_DATASET:?Set POOL_DATASET to a Fission warmed input-pool NPZ}"
TASK_PYTHON="${PYTHON_BIN:-python}"
read -r -a TASK_STRATEGIES <<< "${STRATEGIES:-al4pde_random al4pde_uncertainty al4pde_power al4pde_lcmd al4pde_bait}"
for TASK_STRATEGY in "${TASK_STRATEGIES[@]}"; do
    case "$TASK_STRATEGY" in
        al4pde_random|al4pde_uncertainty|al4pde_power|al4pde_lcmd|al4pde_bait) ;;
        *) echo "Unsupported Fission strategy: $TASK_STRATEGY" >&2; exit 2 ;;
    esac
done
for TASK_STRATEGY in "${TASK_STRATEGIES[@]}"; do
    "$TASK_PYTHON" "$TASK_ROOT/train.py" \
        --system fission --model "${MODEL:-cross_fno}" \
        --dataset_name "$INIT_DATASET" --al_pool_dataset "$POOL_DATASET" \
        --al_strategy "$TASK_STRATEGY" --num_al_rounds "${AL_ROUNDS:-3}" \
        --al_initial_train_size "${ADD_BASE:-1000}" \
        --max_epochs "${EPOCHS:-100}" --batch_size "${BATCH_SIZE:-128}" \
        --width "${WIDTH:-128}" --modes "${MODES:-24}" \
        --loss_type hybrid_mse_rel --norm_type log1p --seed "${SEED:-42}" \
        --exp_root "${EXP_ROOT:-$TASK_ROOT/experiments}" --dataset_tag fission_al \
        "$@"
done
