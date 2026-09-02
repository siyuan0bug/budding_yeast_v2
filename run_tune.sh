#!/bin/bash
# ============================================================
# Optuna 多 GPU 分布式超参自动搜索系统
# ============================================================
#
# 【架构说明】
#   - 单个 Trial 在单张 GPU 上快速训练 (2~4 分钟, 显存 1~4GB)。
#   - 多个 GPU (如 8 张 A6000) 并行启动独立的 Worker 进程。
#   - 所有 Worker 共享一个 SQLite WAL 数据库并发读写，吞吐量直接翻倍 8 倍！
#   - 调参结束后自动输出 Top-K 最优超参及可直接粘贴到 run_pipeline.sh 的配置行。
# ============================================================

cd "$(dirname "$0")" || exit

export NCCL_P2P_DISABLE=1
export PYTHONPATH=$(pwd)/..:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN="/home/users/zsy/.conda/envs/neuralop/bin/python"

# ============================================================
# 用户配置区
# ============================================================

# --- 1. 目标模型与损失策略 ---
MODEL="biom_refiner_plus"              # 可选: biom_refiner_plus, biom_refiner_curvature, pure_wno, pure_fno, resnet_film, hyper_fno 等
LOSS_TYPE="hybrid_mse_rel"             # 可选: hybrid_mse_rel, triple_dynamic, mse_only
AL_STRATEGY="none"                     # 可选: none, al4pde_boundary, al4pde_diversity, rgs 等

# --- 2. 数据集配置 ---
DATA_DIR="/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast"
DATASET="${DATA_DIR}/mass_100000_random_dataset_210min_500steps_dual_labels.npz"
DATASET_TAG="mass_100000_random"

# --- 3. 多 GPU 分布式资源调度 ---
# 指定参与并行搜索的 GPU 编号列表 (例如 8 卡: 0 1 2 3 4 5 6 7; 或部分卡: 0 1 2 3)
GPUS=(0 1 2 3)
TOTAL_TRIALS=200                       # 总计划探索的 Trials 数 (由所有 GPU 共同分担)
SAVE_DIR="./tune_results/${DATASET_TAG}_${MODEL}_${LOSS_TYPE}_${AL_STRATEGY}"
STUDY_NAME="tune_${DATASET_TAG}_${MODEL}_${LOSS_TYPE}"
SEED=42

# --- 4. 开关控制 ---
CLEANUP_CKPTS=true                     # 评估后自动删除 trial 临时 checkpoint 省磁盘
RESUME=false                           # true=从断点续跑, false=全新搜索 (清空旧 study)
NO_PRUNE=false                         # true=关闭中途剪枝, false=开启中途剪枝 (推荐)
PRUNE_WARMUP=10                        # 前 N 个 epoch 不剪枝，让模型热身

# ============================================================
# 调度与执行逻辑 (以下无需修改)
# ============================================================

NUM_GPUS=${#GPUS[@]}
TRIALS_PER_GPU=$(( (TOTAL_TRIALS + NUM_GPUS - 1) / NUM_GPUS ))

mkdir -p "$SAVE_DIR"
DB_FILE="${SAVE_DIR}/${STUDY_NAME}.db"
STORAGE_URL="sqlite:///${DB_FILE}?timeout=60"

CLEANUP_FLAG=""
if [ "$CLEANUP_CKPTS" = "true" ]; then CLEANUP_FLAG="--cleanup_ckpts"; fi

RESUME_FLAG=""
if [ "$RESUME" = "true" ]; then RESUME_FLAG="--resume"; fi

NO_PRUNE_FLAG=""
if [ "$NO_PRUNE" = "true" ]; then NO_PRUNE_FLAG="--no_prune"; fi

echo "============================================================"
echo "  🚀 启动 Optuna 多 GPU 分布式超参搜索"
echo "============================================================"
echo "  模型架构:       $MODEL"
echo "  损失函数:       $LOSS_TYPE"
echo "  主动学习策略:   $AL_STRATEGY"
echo "  数据集:         $(basename $DATASET)"
echo "  分配 GPU 数量:  $NUM_GPUS (卡号: ${GPUS[*]})"
echo "  总搜索 Trials:  $TOTAL_TRIALS (每卡约 $TRIALS_PER_GPU trials)"
echo "  存储数据库:     $STORAGE_URL"
echo "  结果保存目录:   $SAVE_DIR"
echo "  中途剪枝:       $([ "$NO_PRUNE" = "true" ] && echo '关闭' || echo "开启 (热身 ${PRUNE_WARMUP} epoch)")"
echo "  断点续跑:       $([ "$RESUME" = "true" ] && echo '开启' || echo '全新搜索')"
echo "============================================================"
echo ""

# 步骤 0: 主进程单进程预初始化 SQLite 表结构与 Study (彻底避免多进程并发创建表时产生 DDL 冲突)
echo "  🔧 预初始化 Optuna 数据库表结构..."
$PYTHON_BIN -c "
import sqlite3, optuna
conn = sqlite3.connect('${DB_FILE}', timeout=60.0)
conn.execute('PRAGMA journal_mode=WAL;')
conn.close()
optuna.create_study(study_name='${STUDY_NAME}', storage='${STORAGE_URL}', load_if_exists=True)
" 2>/dev/null

echo ""

WORKER_PIDS=()

# 优雅退出处理: 捕获 Ctrl+C 并杀掉所有子 worker 进程
cleanup() {
    echo ""
    echo "⚠️ 捕获中断信号，正在安全终止所有调参 Worker..."
    for pid in "${WORKER_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
    exit 1
}
trap cleanup SIGINT SIGTERM

# 启动多卡并行 Worker 进程
for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}
    WORKER_ID=$i
    LOG_FILE="${SAVE_DIR}/worker_${WORKER_ID}_gpu${GPU_ID}.log"

    echo "  ▶️ 启动 Worker $WORKER_ID (GPU $GPU_ID) -> 日志: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN tune.py \
        --model "$MODEL" \
        --al_strategy "$AL_STRATEGY" \
        --loss_type "$LOSS_TYPE" \
        --dataset_name "$DATASET" \
        --study_name "$STUDY_NAME" \
        --storage "$STORAGE_URL" \
        --n_trials "$TRIALS_PER_GPU" \
        --save_dir "$SAVE_DIR" \
        --seed "$SEED" \
        --worker_id "$WORKER_ID" \
        --prune_warmup_epochs "$PRUNE_WARMUP" \
        $CLEANUP_FLAG \
        $RESUME_FLAG \
        $NO_PRUNE_FLAG > "$LOG_FILE" 2>&1 &

    WORKER_PIDS+=($!)
    sleep 1  # 错开启动，彻底杜绝瞬时并发争抢
done

echo ""
echo "⏳ 所有 ${NUM_GPUS} 个 Worker 正在并行搜索中..."
echo "   实时查看日志: tail -f ${SAVE_DIR}/worker_*.log"
echo "   启动 Dashboard 可视化: optuna-dashboard $STORAGE_URL"
echo ""

# 等待全部 Worker 完成
FAIL_COUNT=0
for pid in "${WORKER_PIDS[@]}"; do
    wait $pid
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
if [ $FAIL_COUNT -eq 0 ]; then
    echo "============================================================"
    echo "  🎉 所有 Worker 调参全部成功完成!"
    echo "============================================================"
    if [ -f "$SAVE_DIR/best_params.json" ]; then
        echo ""
        echo "📊 最优超参结果汇总:"
        cat "$SAVE_DIR/best_params.json"
    fi
else
    echo "⚠️ 调参过程中有 $FAIL_COUNT 个 Worker 异常退出，请检查 ${SAVE_DIR}/ 目录下的日志。"
fi
