#!/usr/bin/env bash
# Fission AL batch pipeline: edit SEEDS and EXPERIMENTS below, then run once.
# Same GPU: sequential queue. Different GPUs: concurrent queues.
# DRY_RUN=1 prints all tasks without creating files or starting training.
set -euo pipefail
TASK_ROOT="${AL_SCRIPT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
TASK_SYSTEM=fission
export PYTHONPATH="$(dirname -- "$TASK_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
TASK_PYTHON="${PYTHON_BIN:-/home/users/zsy/.conda/envs/neuralop/bin/python}"

# ==================== Shared experiment settings ====================
SEEDS=(42)                         # e.g. (42 1024 3407); SEED_LIST also overrides
INIT_TRAIN="${INIT_TRAIN:-5000}"
POOL_SIZE="${POOL_SIZE:-500000}"; POOL_SEED="${POOL_SEED:-20000}"
AL_ROUNDS="${AL_ROUNDS:-4}"
EXP_ROOT="${EXP_ROOT:-/home/users/zsy/mydata/budding_yeast/experiments}"
DATA_DIR="${DATA_DIR:-/home/users/zsy/mydata/fission_yeast}"
POOL_FILE="${POOL_FILE:-${POOL_DATASET:-$DATA_DIR/fission_pool_${POOL_SIZE}_seed${POOL_SEED}_al.npz}}"
# Optional INIT_DATASET template: /path/initial_seed{seed}.npz (one file per seed).
TASK_DEFAULT_INITIAL="$DATA_DIR/fission_initial_train${INIT_TRAIN}_seed{seed}_al.npz"
TASK_INITIAL_TEMPLATE="${INIT_DATASET:-$TASK_DEFAULT_INITIAL}"
export WANDB="${WANDB:-0}"
# AL monitoring is local-only, including when the parent shell requests online.
export WANDB_MODE=offline
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"  # record failed tasks, then run next

# ==================== Editable experiment list ======================
# Each row is ONE complete four-round AL experiment for EACH seed.
# MODEL LOSS_TYPE AL_STRATEGY MODES WIDTH N_BLOCKS BATCH_SIZE LR ROUND_EPOCHS FINAL_EPOCHS ROUND_WARMUP FINAL_WARMUP GPU_ID
# modes is unused by DeepONet/UNet/Transformer; keep a positive placeholder.
# RTX A6000 (48GB class), bf16-mixed; starting configurations, not a VRAM guarantee.
# Default reference: standard FNO, total width 128, modes 24, 4 operator blocks.
# Standard FNO: batch1024, epochs60/120, warmup6/12; approximate update counts preserved.
# Optional BioRefinerPlus keeps batch512, epochs30/60 and warmup3/6.
EXPERIMENTS=(
    # GPU 0: random
    "fno hybrid_mse_rel al4pde_random 24 128 4 1024 1e-3 60 120 6 12 0"
    # GPU 1: uncertainty
    "fno hybrid_mse_rel al4pde_uncertainty 24 128 4 1024 1e-3 60 120 6 12 1"
    # GPU 2: power
    "fno hybrid_mse_rel al4pde_power 24 128 4 1024 1e-3 60 120 6 12 2"
    # GPU 3: bait
    "fno hybrid_mse_rel al4pde_bait 24 128 4 1024 1e-3 60 120 6 12 3"
    # Optional additional model (uncomment to add a task):
    # BioRefinerPlus: coarse stage is fixed at 4 blocks; n_blocks controls refinement.
    # "biom_refiner_plus hybrid_mse_rel al4pde_random 24 128 4 512 8.8e-4 30 60 3 6 0"
)

# Optional CLI/environment filtering. The list above remains the source of defaults.
# MODEL=fno GPU_ID=2 STRATEGIES="al4pde_random al4pde_power" bash run_al_fission.sh
# WIDTH/BATCH_SIZE/N_BLOCKS/LR override selected rows; add new models as new rows.
if [[ -n "${SEED_LIST:-${SEED:-}}" ]]; then read -r -a SEEDS <<< "${SEED_LIST:-$SEED}"; fi
if [[ -n "${EXPERIMENTS_TEXT:-}" ]]; then mapfile -t EXPERIMENTS <<< "$EXPERIMENTS_TEXT"; fi
if [[ ${#SEEDS[@]} == 0 ]]; then echo 'No seeds configured' >&2; exit 2; fi
for task_seed in "${SEEDS[@]}"; do
    if [[ ! "$task_seed" =~ ^[0-9]+$ ]]; then echo "Invalid seed: $task_seed" >&2; exit 2; fi
done
if [[ ${#SEEDS[@]} -gt 1 && "$TASK_INITIAL_TEMPLATE" != *'{seed}'* ]]; then
    echo 'Multiple seeds require {seed} in INIT_DATASET to avoid mixing splits' >&2; exit 2
fi

TASK_ROWS=(); TASK_STRATEGIES=()
for task_row in "${EXPERIMENTS[@]}"; do
    read -r -a task_columns <<< "$task_row"
    if [[ ${#task_columns[@]} == 0 || "${task_columns[0]}" == \#* ]]; then continue; fi
    if [[ ${#task_columns[@]} != 13 ]]; then echo "Expected 13 experiment columns: $task_row" >&2; exit 2; fi
    read -r task_model task_loss task_strategy task_modes task_width task_blocks task_batch task_lr task_epochs task_final task_warmup task_finalwarm task_gpu <<< "$task_row"
    if [[ -n "${MODEL:-${MODELS:-}}" && " ${MODEL:-${MODELS:-}} " != *" $task_model "* ]]; then continue; fi
    if [[ -n "${STRATEGIES:-}" && " $STRATEGIES " != *" $task_strategy "* ]]; then continue; fi
    task_row="$task_model $task_loss $task_strategy ${MODES:-$task_modes} ${WIDTH:-$task_width} ${N_BLOCKS:-$task_blocks} ${BATCH_SIZE:-$task_batch} ${LR:-$task_lr} ${ROUND_EPOCHS:-$task_epochs} ${FINAL_EPOCHS:-$task_final} ${ROUND_WARMUP:-$task_warmup} ${FINAL_WARMUP:-$task_finalwarm} ${GPU_ID:-$task_gpu}"
    TASK_ROWS+=("$task_row")
    if [[ " ${TASK_STRATEGIES[*]} " != *" $task_strategy "* ]]; then TASK_STRATEGIES+=("$task_strategy"); fi
done
if [[ ${#TASK_ROWS[@]} == 0 ]]; then echo 'No experiments match the configured list/filters' >&2; exit 2; fi
# Backward-compatible GPU_IDS shortcut; arbitrary row-level assignments need no shortcut.
if [[ -n "${GPU_IDS:-}" ]]; then
    read -r -a task_override_gpus <<< "$GPU_IDS"
    if [[ ${#task_override_gpus[@]} != 1 && ${#task_override_gpus[@]} != ${#TASK_STRATEGIES[@]} ]]; then
        echo 'GPU_IDS must contain one GPU or one per selected strategy' >&2; exit 2
    fi
    for task_i in "${!TASK_ROWS[@]}"; do
        read -r -a task_columns <<< "${TASK_ROWS[$task_i]}"
        for task_j in "${!TASK_STRATEGIES[@]}"; do
            if [[ "${task_columns[2]}" == "${TASK_STRATEGIES[$task_j]}" ]]; then
                task_columns[12]="${task_override_gpus[$((task_j % ${#task_override_gpus[@]}))]}"; break
            fi
        done
        TASK_ROWS[$task_i]="${task_columns[*]}"
    done
fi

initial_path() { echo "${TASK_INITIAL_TEMPLATE//\{seed\}/$1}"; }
experiment_path() {
    local seed="$1" model loss strategy modes width blocks batch lr epochs final warmup finalwarm gpu
    read -r model loss strategy modes width blocks batch lr epochs final warmup finalwarm gpu <<< "$2"
    echo "$EXP_ROOT/al_${TASK_SYSTEM}/$model/${loss}_${strategy}/w${width}_n${blocks}_m${modes}_b${batch}_e${epochs}-${final}_lr${lr}_wm${warmup}-${finalwarm}_al${AL_ROUNDS}_refit_v2/seed_${seed}"
}

declare -A TASK_GPU_LIST TASK_GPU_COUNT TASK_OUTPUTS
ALL_GPUS=(); TOTAL_TASKS=0
for task_seed in "${SEEDS[@]}"; do
    for task_row in "${TASK_ROWS[@]}"; do
        read -r -a task_columns <<< "$task_row"; task_gpu="${task_columns[12]}"
        for task_col in 3 4 5 6 8 9; do
            if [[ ! "${task_columns[$task_col]}" =~ ^[0-9]+$ || "${task_columns[$task_col]}" -lt 1 ]]; then
                echo "Expected a positive integer in experiment column $((task_col+1)): $task_row" >&2; exit 2
            fi
        done
        for task_col in 10 11; do
            if [[ ! "${task_columns[$task_col]}" =~ ^[0-9]+$ ]]; then
                echo "Expected a nonnegative warmup in experiment column $((task_col+1)): $task_row" >&2; exit 2
            fi
        done
        if [[ ! "${task_columns[7]}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || ! awk -v rate="${task_columns[7]}" 'BEGIN {exit !(rate>0)}'; then
            echo "Invalid learning rate: ${task_columns[7]}" >&2; exit 2
        fi
        if [[ ! "$task_gpu" =~ ^[0-9]+$ ]]; then echo "Invalid GPU: $task_gpu" >&2; exit 2; fi
        task_output="$(experiment_path "$task_seed" "$task_row")"
        if [[ -n "${TASK_OUTPUTS[$task_output]:-}" ]]; then echo "Duplicate experiment output: $task_output" >&2; exit 2; fi
        TASK_OUTPUTS[$task_output]=1
        if [[ -z "${TASK_GPU_COUNT[$task_gpu]:-}" ]]; then
            ALL_GPUS+=("$task_gpu"); TASK_GPU_COUNT[$task_gpu]=0; TASK_GPU_LIST[$task_gpu]=''
        fi
        TASK_GPU_COUNT[$task_gpu]=$((TASK_GPU_COUNT[$task_gpu]+1))
        TASK_GPU_LIST[$task_gpu]+="${task_seed}|${task_row}"$'\n'
        TOTAL_TASKS=$((TOTAL_TASKS+1))
    done
done
echo "Plan: ${#TASK_ROWS[@]} experiments x ${#SEEDS[@]} seeds = $TOTAL_TASKS tasks; ${#ALL_GPUS[@]} GPU queues"
for task_gpu in "${ALL_GPUS[@]}"; do echo "Queue GPU $task_gpu: ${TASK_GPU_COUNT[$task_gpu]} tasks"; done
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    for task_gpu in "${ALL_GPUS[@]}"; do
        while IFS='|' read -r task_seed task_row; do
            [[ -n "$task_seed" ]] || continue
            read -r model loss strategy modes width blocks batch lr epochs final warmup finalwarm gpu <<< "$task_row"
            echo "GPU=$gpu strategy=$strategy model=$model seed=$task_seed width=$width blocks=$blocks batch=$batch lr=$lr modes=$modes epochs=$epochs/$final loss=$loss"
        done <<< "${TASK_GPU_LIST[$task_gpu]}"
    done
    exit 0
fi

# Prepare a single shared pool and one initial train/val/test split per seed.
mkdir -p "$DATA_DIR"
if [[ "${SKIP_PREPARE:-0}" == 1 && ! -f "$POOL_FILE" ]]; then
    echo "SKIP_PREPARE requires existing pool: $POOL_FILE" >&2; exit 2
fi
if [[ ! -f "$POOL_FILE" ]]; then
    "$TASK_PYTHON" -m budding_yeast_v2.data.systems.fission.al_pool --samples "$POOL_SIZE" --seed "$POOL_SEED" \
        --output "$POOL_FILE" --workers "${SIM_WORKERS:-32}"
fi
for task_seed in "${SEEDS[@]}"; do
    if [[ "${SKIP_PREPARE:-0}" == 1 && ! -f "$(initial_path "$task_seed")" ]]; then
        echo "SKIP_PREPARE requires existing initial dataset: $(initial_path "$task_seed")" >&2; exit 2
    fi
    "$TASK_PYTHON" -m budding_yeast_v2.data.systems.al_initial --pool "$POOL_FILE" --output "$(initial_path "$task_seed")" \
        --train-samples "$INIT_TRAIN" --seed "$task_seed" --workers "${SIM_WORKERS:-32}" --check-existing
done
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi

TASK_LOG_ROOT="$EXP_ROOT/al_${TASK_SYSTEM}/batch_logs/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$TASK_LOG_ROOT"
# Stop only this launcher's tracked process trees when explicitly interrupted.
stop_tree() {
    local parent="$1" child
    while read -r child; do [[ -n "$child" ]] && stop_tree "$child"; done < <(pgrep -P "$parent" || true)
    kill -TERM "$parent" 2>/dev/null || true
}
TASK_CHILD_PID=''; TASK_PIDS=()
cleanup_all() {
    trap '' INT TERM
    local pid
    for pid in "${TASK_PIDS[@]}"; do stop_tree "$pid"; done
    exit 130
}
trap cleanup_all INT TERM

run_task() {
    local seed="$1" row="$2" model loss strategy modes width blocks batch lr epochs final warmup finalwarm gpu output status
    shift 2
    read -r model loss strategy modes width blocks batch lr epochs final warmup finalwarm gpu <<< "$row"
    output="$(experiment_path "$seed" "$row")"
    echo "START GPU=$gpu model=$model strategy=$strategy seed=$seed"
    if [[ -f "$output/config.json" ]]; then
        echo "Refusing existing run: $output (choose a new EXP_ROOT or seed)" >&2; return 2
    fi
    mkdir -p "$output" || return 2
    local args=(--system "$TASK_SYSTEM" --model "$model" --loss_type "$loss" --dataset_name "$(initial_path "$seed")"
        --al_pool_dataset "$POOL_FILE" --al_strategy "$strategy" --al_initial_train_size "$INIT_TRAIN" --num_al_rounds "$AL_ROUNDS"
        --al_round_epochs "$epochs" --al_final_epochs "$final" --al_round_warmup "$warmup" --al_final_warmup "$finalwarm"
        --width "$width" --n_blocks "$blocks" --modes "$modes" --batch_size "$batch" --val_batch_size "$batch" --lr "$lr"
        --norm_type log1p --seed "$seed" --num_workers "${LOADER_WORKERS:-2}" --devices 1 --exp_dir "$output" --dataset_tag "al_${TASK_SYSTEM}")
    if [[ "$WANDB" == 0 ]]; then args+=(--no_wandb); fi
    CUDA_VISIBLE_DEVICES="$gpu" "$TASK_PYTHON" -u "$TASK_ROOT/train.py" "${args[@]}" "$@" > "$output/train.log" 2>&1 &
    TASK_CHILD_PID=$!
    if wait "$TASK_CHILD_PID"; then status=0; else status=$?; fi
    TASK_CHILD_PID=''
    echo "END GPU=$gpu model=$model strategy=$strategy seed=$seed exit=$status log=$output/train.log"
    return "$status"
}

wait_for_gpu() {
    local gpu="$1" memory
    if [[ "${WAIT_FOR_GPU:-0}" != 1 ]]; then return 0; fi
    while true; do
        if ! memory=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits); then
            echo "Cannot query GPU $gpu" >&2; return 2
        fi
        memory=${memory//[[:space:]]/}
        if [[ "$memory" =~ ^[0-9]+$ && "$memory" -le "${GPU_IDLE_MAX_MIB:-512}" ]]; then return 0; fi
        echo "WAIT GPU=$gpu memory=${memory}MiB; waiting for idle GPU"
        sleep "${GPU_WAIT_SECONDS:-30}"
    done
}

run_worker() {
    local gpu="$1" seed row status failed=0
    shift
    trap 'trap "" INT TERM; if [[ -n "$TASK_CHILD_PID" ]]; then stop_tree "$TASK_CHILD_PID"; fi; exit 143' INT TERM
    printf 'seed\tmodel\tstrategy\texit_code\texperiment_dir\n' > "$TASK_LOG_ROOT/gpu${gpu}.tsv"
    while IFS='|' read -r seed row; do
        [[ -n "$seed" ]] || continue
        if ! wait_for_gpu "$gpu"; then failed=1; break; fi
        if run_task "$seed" "$row" "$@"; then status=0; else status=$?; fi
        read -r -a task_columns <<< "$row"
        printf '%s\t%s\t%s\t%s\t%s\n' "$seed" "${task_columns[0]}" "${task_columns[2]}" "$status" "$(experiment_path "$seed" "$row")" >> "$TASK_LOG_ROOT/gpu${gpu}.tsv"
        if [[ "$status" != 0 ]]; then
            failed=1
            if [[ "$CONTINUE_ON_ERROR" != 1 || "$status" == 130 || "$status" == 143 ]]; then break; fi
        fi
    done <<< "${TASK_GPU_LIST[$gpu]}"
    return "$failed"
}
for task_gpu in "${ALL_GPUS[@]}"; do
    run_worker "$task_gpu" "$@" > "$TASK_LOG_ROOT/gpu${task_gpu}.log" 2>&1 &
    TASK_PIDS+=("$!")
done
TASK_STATUS=0
for task_pid in "${TASK_PIDS[@]}"; do
    if wait "$task_pid"; then :; else TASK_STATUS=1; fi
done
trap - INT TERM
{ printf 'seed\tmodel\tstrategy\texit_code\texperiment_dir\n'; for task_gpu in "${ALL_GPUS[@]}"; do tail -n +2 "$TASK_LOG_ROOT/gpu${task_gpu}.tsv"; done; } > "$TASK_LOG_ROOT/tasks.tsv"
echo "Batch finished: exit=$TASK_STATUS; task status table: $TASK_LOG_ROOT/tasks.tsv"
exit "$TASK_STATUS"
