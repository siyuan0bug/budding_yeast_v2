#!/usr/bin/env bash
# Budding AL batch pipeline: edit SEEDS and EXPERIMENTS below, then run once.
# Same GPU: bounded concurrent queue. Different GPUs: concurrent queues.
# DRY_RUN=1 prints all tasks without creating files or starting training.
set -euo pipefail
TASK_ROOT="${AL_SCRIPT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
TASK_SYSTEM=budding
export PYTHONPATH="$(dirname -- "$TASK_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
TASK_PYTHON="${PYTHON_BIN:-/home/users/zsy/.conda/envs/neuralop/bin/python}"

# ==================== Shared experiment settings ====================
# Current cohort: balanced module/allele pool; regenerate an independent initial
# dataset from this pool once, then share it across all model/strategy rows.
SEEDS=(42)                         # e.g. (42 1024 3407); SEED_LIST also overrides
INIT_TRAIN="${INIT_TRAIN:-10000}"
POOL_SIZE="${POOL_SIZE:-1000000}"; POOL_SEED="${POOL_SEED:-42}"
AL_ROUNDS="${AL_ROUNDS:-4}"
TRAINING_PROFILE="${TRAINING_PROFILE:-full}"
case "$TRAINING_PROFILE" in
    screen) TASK_PRESET_EPOCHS=50; TASK_PRESET_FINAL=100; TASK_PRESET_WARMUP=4; TASK_PRESET_FINAL_WARMUP=8 ;;
    full) TASK_PRESET_EPOCHS=100; TASK_PRESET_FINAL=200; TASK_PRESET_WARMUP=5; TASK_PRESET_FINAL_WARMUP=10 ;;
    *) echo 'TRAINING_PROFILE must be screen or full' >&2; exit 2 ;;
esac
MIN_LR="${MIN_LR:-1e-5}"
if [[ ! "$MIN_LR" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
    echo 'MIN_LR must be a finite nonnegative number' >&2; exit 2
fi
EXP_ROOT="${EXP_ROOT:-/home/users/zsy/mydata/budding_yeast/experiments}"
DATA_DIR="${DATA_DIR:-/home/users/zsy/mydata/budding_yeast}"
DATA_COHORT="${DATA_COHORT:-gene_ranges}"
AL_SUITE="${AL_SUITE:-mechanism}" # train-only Pattern evidence and canonical ODE response
if [[ "$AL_SUITE" != original && "$AL_SUITE" != research && "$AL_SUITE" != optimized && "$AL_SUITE" != mechanism ]]; then
    echo 'AL_SUITE must be original, research, optimized or mechanism' >&2; exit 2
fi
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
if [[ "$AL_SUITE" == research && -z "$EXPERIMENT_TAG" ]]; then EXPERIMENT_TAG=research_v1; fi
if [[ "$AL_SUITE" == optimized && -z "$EXPERIMENT_TAG" ]]; then EXPERIMENT_TAG=optimized_v2; fi
if [[ "$AL_SUITE" == mechanism && -z "$EXPERIMENT_TAG" ]]; then EXPERIMENT_TAG=mechanism_v1; fi
if [[ -n "$EXPERIMENT_TAG" && ! "$EXPERIMENT_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'Invalid EXPERIMENT_TAG' >&2; exit 2
fi
TASK_RESULT_COHORT="$DATA_COHORT${EXPERIMENT_TAG:+_$EXPERIMENT_TAG}_${TRAINING_PROFILE}"
if [[ ! "$DATA_COHORT" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'DATA_COHORT must contain only letters, numbers, underscores or hyphens' >&2; exit 2
fi
POOL_FILE="${POOL_FILE:-${POOL_DATASET:-$DATA_DIR/budding_pool_${POOL_SIZE}_seed${POOL_SEED}_${DATA_COHORT}.npz}}"
# Optional INIT_DATASET template: /path/initial_seed{seed}.npz (one file per seed).
TASK_DEFAULT_INITIAL="$DATA_DIR/budding_initial_train${INIT_TRAIN}_seed{seed}_${DATA_COHORT}.npz"
TASK_INITIAL_TEMPLATE="${INIT_DATASET:-$TASK_DEFAULT_INITIAL}"
export WANDB="${WANDB:-1}"
# AL monitoring is local-only, including when the parent shell requests online.
export WANDB_MODE=offline
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"  # record failed tasks, then run next
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"  # four cards concurrently; avoid host-RAM OOM
if [[ ! "$MAX_JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo 'MAX_JOBS_PER_GPU must be a positive integer' >&2; exit 2
fi
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
    echo 'This bounded task scheduler requires Bash 5.1 or newer' >&2; exit 2
fi
# Bound CPU oversubscription when several experiments query the ODE together.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export AL_SIM_WORKERS="${AL_SIM_WORKERS:-4}"

# ==================== Editable experiment list ======================
# Each row is ONE complete four-round AL experiment for EACH seed.
# MODEL LOSS_TYPE AL_STRATEGY MODES WIDTH N_BLOCKS BATCH_SIZE LR ROUND_EPOCHS FINAL_EPOCHS ROUND_WARMUP FINAL_WARMUP GPU_ID
# modes is unused by DeepONet/UNet/Transformer; keep a positive placeholder.
# RTX A6000 (48GB class), bf16-mixed; starting configurations, not a VRAM guarantee.
# Default full budget: 100/100/100/200, warmup 5/5/5/10.
# TRAINING_PROFILE=screen: 50/50/50/100, warmup 4/4/4/8.
# FNO: width192, modes24, blocks4, batch512, peak LR1e-3.
# Reference for other Budding models/AL methods: batch512, LR1e-3,
# Same profile across AL methods; retain architecture-specific capacity.
EXPERIMENTS=(
    # FNO only; preserve one model/training protocol across AL baselines.
    # GPU 0: random + output-trajectory MaxDist.
    "fno hybrid_mse_rel al4pde_random 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 0"
    "fno hybrid_mse_rel al4pde_max_dist 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 0"
    # GPU 1: ensemble Top-K + LCMD.
    "fno hybrid_mse_rel al4pde_uncertainty 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 1"
    "fno hybrid_mse_rel al4pde_lcmd 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 1"
    # GPU 2: power sampling + within-background response uncertainty.
    "fno hybrid_mse_rel al4pde_power 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 2"
    "fno hybrid_mse_rel ours_response 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 2"
    # GPU 3: BAIT.
    "fno hybrid_mse_rel al4pde_bait 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 3"
)
if [[ "$AL_SUITE" == research ]]; then
    EXPERIMENTS+=(
        "fno hybrid_mse_rel ours_response_marginal 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 0"
        "fno hybrid_mse_rel ours_response_feedback 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 1"
        "fno hybrid_mse_rel ours_interaction 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 2"
        "fno hybrid_mse_rel ours_invariance 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 3"
        "fno hybrid_mse_rel ours_event 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 0"
        "fno hybrid_mse_rel ours_error 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 1"
        "fno hybrid_mse_rel ours_coverage 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 3"
    )
fi
if [[ "$AL_SUITE" == optimized ]]; then
    # Finished v1 runs remain in their original directories. Every row below
    # starts from round 0 under an independent optimized_v2 result namespace.
    EXPERIMENTS=(
        "fno hybrid_mse_rel ours_global_interaction 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 0"
        "fno hybrid_mse_rel al4pde_uncertainty 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 0"
        "fno hybrid_mse_rel ours_event 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 0"
        "fno hybrid_mse_rel ours_global_response 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 1"
        "fno hybrid_mse_rel al4pde_power 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 1"
        "fno hybrid_mse_rel ours_invariance 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 1"
        "fno hybrid_mse_rel al4pde_lcmd 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 2"
        "fno hybrid_mse_rel ours_global_tail 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 2"
        "fno hybrid_mse_rel ours_response 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 2"
        "fno hybrid_mse_rel al4pde_bait 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 3"
        "fno hybrid_mse_rel ours_global_calibrated 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 3"
    )
fi
if [[ "$AL_SUITE" == mechanism ]]; then
    EXPERIMENTS=(
        "fno hybrid_mse_rel ours_sensitivity 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 0"
        "fno hybrid_mse_rel ours_pattern_balance 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 1"
        "fno hybrid_mse_rel ours_pattern_context 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 2"
        "fno hybrid_mse_rel ours_pattern_support 24 192 4 512 1e-3 $TASK_PRESET_EPOCHS $TASK_PRESET_FINAL $TASK_PRESET_WARMUP $TASK_PRESET_FINAL_WARMUP 3"
    )
fi

# Optional CLI/environment filtering. The list above remains the source of defaults.
# MODEL=fno GPU_ID=2 STRATEGIES="al4pde_random al4pde_power" bash run_al_budding.sh
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
    echo "$EXP_ROOT/al_${TASK_SYSTEM}/$model/${loss}_${strategy}/w${width}_n${blocks}_m${modes}_b${batch}_e${epochs}-${final}_lr${lr}_lmin${MIN_LR}_wm${warmup}-${finalwarm}_al${AL_ROUNDS}_refit_${TASK_RESULT_COHORT}/seed_${seed}"
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
        if ! awk -v floor="$MIN_LR" -v peak="${task_columns[7]}" 'BEGIN {exit !(floor>=0 && floor<=peak)}'; then
            echo "MIN_LR must be between zero and peak LR: $task_row" >&2; exit 2
        fi
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
echo "Concurrency: max $MAX_JOBS_PER_GPU experiments per GPU; AL_SIM_WORKERS=$AL_SIM_WORKERS each"
echo "Training profile=$TRAINING_PROFILE; screen is a finite-budget comparison, not a convergence guarantee"
for task_gpu in "${ALL_GPUS[@]}"; do echo "Queue GPU $task_gpu: ${TASK_GPU_COUNT[$task_gpu]} tasks"; done
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "Pool: $POOL_FILE"
    for task_seed in "${SEEDS[@]}"; do echo "Initial seed=$task_seed: $(initial_path "$task_seed")"; done
    echo "Results: $EXP_ROOT/al_${TASK_SYSTEM}/<model>/<loss_strategy>/<settings>_refit_${TASK_RESULT_COHORT}/seed_<seed>"
    echo "Suite=$AL_SUITE; primary round seed policy=$([[ "$AL_SUITE" != original ]] && echo fixed || echo legacy)"
    echo "W&B enabled=$WANDB mode=$WANDB_MODE"
    echo "Cosine minimum learning rate=$MIN_LR; convergence diagnostics window=10 epochs (no automatic stopping)"
    for task_gpu in "${ALL_GPUS[@]}"; do
        while IFS='|' read -r task_seed task_row; do
            [[ -n "$task_seed" ]] || continue
            read -r model loss strategy modes width blocks batch lr epochs final warmup finalwarm gpu <<< "$task_row"
            echo "GPU=$gpu strategy=$strategy model=$model seed=$task_seed width=$width blocks=$blocks batch=$batch lr=$lr modes=$modes epochs=$epochs/$final loss=$loss warmup=$warmup/$finalwarm min_lr=$MIN_LR"
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
    "$TASK_PYTHON" "$TASK_ROOT/data/generate.py" --recipe budding_pool --samples "$POOL_SIZE" --seed "$POOL_SEED" \
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
task_timestamp() { TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S%:z'; }
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
    echo "[$(task_timestamp)] START GPU=$gpu model=$model strategy=$strategy seed=$seed"
    if [[ -f "$output/config.json" ]]; then
        echo "Refusing existing run: $output (choose a new EXP_ROOT or seed)" >&2; return 2
    fi
    mkdir -p "$output" || return 2
    printf '[%s] START GPU=%s model=%s strategy=%s seed=%s\n' \
        "$(task_timestamp)" "$gpu" "$model" "$strategy" "$seed" > "$output/train.log"
    local args=(--system "$TASK_SYSTEM" --model "$model" --loss_type "$loss" --dataset_name "$(initial_path "$seed")"
        --al_pool_dataset "$POOL_FILE" --al_strategy "$strategy" --al_initial_train_size "$INIT_TRAIN" --num_al_rounds "$AL_ROUNDS"
        --al_round_epochs "$epochs" --al_final_epochs "$final" --al_round_warmup "$warmup" --al_final_warmup "$finalwarm"
        --width "$width" --n_blocks "$blocks" --modes "$modes" --batch_size "$batch" --val_batch_size "$batch" --lr "$lr" --min_lr "$MIN_LR"
        --norm_type log1p --seed "$seed" --num_workers "${LOADER_WORKERS:-2}" --devices 1 --exp_dir "$output" --dataset_tag "al_${TASK_SYSTEM}")
    if [[ "$WANDB" == 0 ]]; then args+=(--no_wandb); fi
    if [[ "$AL_SUITE" != original ]]; then args+=(--al_round_seed_policy fixed); fi
    CUDA_VISIBLE_DEVICES="$gpu" "$TASK_PYTHON" -u "$TASK_ROOT/train.py" "${args[@]}" "$@" >> "$output/train.log" 2>&1 &
    TASK_CHILD_PID=$!
    if wait "$TASK_CHILD_PID"; then status=0; else status=$?; fi
    TASK_CHILD_PID=''
    local finished="[$(task_timestamp)] END GPU=$gpu model=$model strategy=$strategy seed=$seed exit=$status log=$output/train.log"
    echo "$finished"
    printf '%s\n' "$finished" >> "$output/train.log"
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
    local gpu="$1" seed row status failed=0 next=0 halted=0 pid finished
    local -a active=() seeds=() rows=() remaining=()
    local -A job_seeds job_rows
    shift
    trap 'trap "" INT TERM; for pid in "${active[@]}"; do stop_tree "$pid"; done; exit 143' INT TERM
    printf 'seed\tmodel\tstrategy\texit_code\texperiment_dir\n' > "$TASK_LOG_ROOT/gpu${gpu}.tsv"
    while IFS='|' read -r seed row; do
        [[ -n "$seed" ]] || continue
        seeds+=("$seed"); rows+=("$row")
    done <<< "${TASK_GPU_LIST[$gpu]}"
    while (( next < ${#rows[@]} || ${#active[@]} > 0 )); do
        while (( halted == 0 && next < ${#rows[@]} && ${#active[@]} < MAX_JOBS_PER_GPU )); do
            # Idle checks apply before claiming the card. Existing jobs from this
            # worker must not block its second slot on the idle-memory threshold.
            if (( ${#active[@]} == 0 )) && ! wait_for_gpu "$gpu"; then
                failed=1; halted=1; break
            fi
            seed="${seeds[$next]}"; row="${rows[$next]}"
            run_task "$seed" "$row" "$@" &
            pid=$!; active+=("$pid"); job_seeds[$pid]="$seed"; job_rows[$pid]="$row"
            next=$((next+1))
        done
        if (( ${#active[@]} == 0 )); then break; fi
        finished=''
        # wait -n ignores children that already completed before it was called.
        # Harvest those explicitly, including fast failures, without losing status.
        for pid in "${active[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then finished="$pid"; break; fi
        done
        if [[ -n "$finished" ]]; then
            if wait "$finished"; then status=0; else status=$?; fi
        else
            if wait -n -p finished "${active[@]}"; then status=0; else status=$?; fi
            if [[ -z "${finished:-}" ]]; then
                finished="${active[0]}"
                if wait "$finished"; then status=0; else status=$?; fi
            fi
        fi
        seed="${job_seeds[$finished]}"; row="${job_rows[$finished]}"
        read -r -a task_columns <<< "$row"
        printf '%s\t%s\t%s\t%s\t%s\n' "$seed" "${task_columns[0]}" "${task_columns[2]}" "$status" "$(experiment_path "$seed" "$row")" >> "$TASK_LOG_ROOT/gpu${gpu}.tsv"
        remaining=()
        for pid in "${active[@]}"; do [[ "$pid" == "$finished" ]] || remaining+=("$pid"); done
        active=("${remaining[@]}")
        unset 'job_seeds[$finished]' 'job_rows[$finished]'
        if [[ "$status" != 0 ]]; then
            failed=1
            # Already-running experiments finish; pending experiments stay queued.
            if [[ "$CONTINUE_ON_ERROR" != 1 || "$status" == 130 || "$status" == 143 ]]; then halted=1; fi
        fi
    done
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
