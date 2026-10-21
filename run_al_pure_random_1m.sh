#!/bin/bash
# ==============================================================================
# 100万级纯随机主动学习 (AL4PDE 几何倍增 + K-Means 初始冷启动) 一键式四卡并行运行脚本
# ==============================================================================
# 硬件分配 (3 张 NVIDIA RTX A6000 48GB 独立显卡并行评估三大独立理论路线):
#   - GPU 0: al4pde_bifurcation   (路线一: 相空间极限环吸引子与分岔临界流形主动学习 BM-AL)
#   - GPU 1: al4pde_event_entropy (路线二: 生物检验点离散事件转移熵 HETE-AL)
#   - GPU 2: al4pde_fisher_oed    (路线三: 泛函空间贝叶斯最优实验设计 F-OED)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/users/zsy/.conda/envs/neuralop/bin/python}"
POOL_FILE="${POOL_FILE:-/home/users/zsy/mydata/budding_yeast/budding_pool_1000000_seed42_exact3.npz}"
INIT_DATASET="${INIT_DATASET:-/home/users/zsy/mydata/budding_yeast/al_iter0_10000_210m_500s_exact3.npz}"
EXP_ROOT="./experiments"
mkdir -p "$EXP_ROOT" /home/users/zsy/mydata/budding_yeast

# 实验超参数定义
MODEL="biom_refiner_plus"
LOSS_TYPE="hybrid_mse_rel"
NORM_TYPE="log1p"
BATCH_SIZE=256
EPOCHS=100
WIDTH=192
MODES=24
LR="8.8e-4"
WARMUP=8
SEED=42
DATASET_TAG="al_pure_random_1m"
HPARAM_SIG="w${WIDTH}_m${MODES}_b${BATCH_SIZE}_ep${EPOCHS}_lr${LR}_wm${WARMUP}"
if [ -n "$NORM_TYPE" ] && [ "$NORM_TYPE" != "log1p" ]; then
    HPARAM_SIG="${HPARAM_SIG}_${NORM_TYPE}"
fi

# 算法配置数组: "GPU_ID STRATEGY" (使用 GPU 3 独立运行，与 run_pipeline.sh 的 GPU 0/1/2 互不冲突)
AL_JOBS=(
    "3 al4pde_bifurcation_entropy"
    #"0 al4pde_bifurcation_entropy"
    #"1 al4pde_bifurcation"
    #"2 al4pde_event_entropy"
    #"3 al4pde_fisher_oed"
)

JOB_DESC=""
for JOB in "${AL_JOBS[@]}"; do
    read -r G_ID G_STRAT <<< "$JOB"
    JOB_DESC+="${JOB_DESC:+, }GPU $G_ID ($G_STRAT)"
done

echo "================================================================================"
echo " 🌟 100万级纯随机主动学习 (AL4PDE 几何倍增闭环) 一键式流水线 🌟"
echo " 当前工作目录 : $SCRIPT_DIR"
echo " 候选数据池   : $POOL_FILE"
echo " 初始训练集   : $INIT_DATASET"
echo " 显卡与任务   : $JOB_DESC"
echo " 任务总数量   : ${#AL_JOBS[@]} 个并发主动学习实验"
echo " 归一化规范   : $NORM_TYPE (变量 log1p + 参数线性 Z-score, 经典基准)"
echo " 模型架构     : $MODEL (w$WIDTH, m$MODES, b$BATCH_SIZE, ep$EPOCHS, lr$LR, wm$WARMUP)"
echo " 轮次机制     : 3 轮指数倍增 (Iter 0: 10k -> Iter 1: 20k -> Iter 2: 40k)"
echo "================================================================================"

# ==============================================================================
# 阶段 1: 检查并生成 100 万组纯随机参数候选池 (Sobol 准随机低差异映射)
# ==============================================================================
if [ ! -f "$POOL_FILE" ]; then
    echo -e "\n[阶段 1/3] 未找到 100 万候选池文件，启动秒级采样引擎..."
    $PYTHON_BIN data/generate.py --recipe budding_pool \
        --samples 1000000 \
        --output "$POOL_FILE" \
        --seed 42
else
    echo -e "\n[阶段 1/3] ✅ 检测到已有 100 万纯随机候选池: $POOL_FILE"
fi

# ==============================================================================
# 阶段 2: 检查并生成 Iteration 0 初始训练集 (MiniBatchKMeans 选拔 + 多进程 ODE 仿真)
# ==============================================================================
if [ ! -f "$INIT_DATASET" ]; then
    echo -e "\n[阶段 2/3] 未找到 Iteration 0 训练集，运行 MiniBatchKMeans (K=10000) 选拔并求解 ODE..."
    $PYTHON_BIN data/al_pool_manager.py \
        --action init_kmeans \
        --pool_path "$POOL_FILE" \
        --n_initial 10000 \
        --output_dataset "$INIT_DATASET" \
        --num_workers 40 \
        --seed 42
else
    echo -e "\n[阶段 2/3] ✅ 检测到已有 Iteration 0 训练集: $INIT_DATASET"
fi

# ==============================================================================
# 阶段 3: 并行启动 AL4PDE 主动学习任务
# ==============================================================================
echo -e "\n[阶段 3/3] 正在并行启动 ${#AL_JOBS[@]} 个 AL4PDE 算法实验..."

PIDS=()
LOG_FILES=()

for JOB in "${AL_JOBS[@]}"; do
    read -r GPU_ID STRATEGY <<< "$JOB"
    
    EXP_DIR="${EXP_ROOT}/${DATASET_TAG}/${MODEL}/${LOSS_TYPE}_${STRATEGY}/${HPARAM_SIG}/seed_${SEED}"
    CKPT_DIR="${EXP_DIR}/checkpoints"
    EVAL_DIR="${EXP_DIR}/eval"
    mkdir -p "$CKPT_DIR" "$EVAL_DIR"

    RUN_NAME="${MODEL}_${STRATEGY}_${LOSS_TYPE}_${DATASET_TAG}_modes${MODES}"
    if [ -n "$NORM_TYPE" ] && [ "$NORM_TYPE" != "log1p" ]; then
        RUN_NAME="${RUN_NAME}_${NORM_TYPE}"
    fi
    RUN_NAME="${RUN_NAME}_seed${SEED}"
    LOG_FILE="${EXP_DIR}/train.log"
    LOG_FILES+=("$LOG_FILE")

    echo "🚀 启动 [GPU $GPU_ID] -> 策略: ${STRATEGY} | 日志: $LOG_FILE"

    (
        echo "==========================================================="
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 AL4PDE 训练: $RUN_NAME (GPU $GPU_ID)"
        echo "==========================================================="
        
        CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN -u train.py \
            --model $MODEL \
            --wavelet haar \
            --loss_type $LOSS_TYPE \
            --dataset_name "$INIT_DATASET" \
            --modes $MODES \
            --width $WIDTH \
            --batch_size $BATCH_SIZE \
            --max_epochs $EPOCHS \
            --seed $SEED \
            --name "$RUN_NAME" \
            --lr $LR \
            --warmup_epochs $WARMUP \
            --num_workers 2 \
            --val_batch_size $BATCH_SIZE \
            --devices 1 \
            --norm_type $NORM_TYPE \
            --rollout_type one_time \
            --exp_dir "$EXP_DIR" \
            --dataset_tag "$DATASET_TAG" \
            --al_strategy "$STRATEGY" \
            --al_pool_dataset "$POOL_FILE" \
            --al_initial_train_size 10000 \
            --num_al_rounds 3 \
            --no_wandb

        TRAIN_EXIT=$?
        if [ $TRAIN_EXIT -eq 0 ]; then
            echo -e "\n==========================================================="
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎯 AL4PDE 3 轮训练与各轮次全面评测圆满完成！"
            echo "  |-- 各轮次 Checkpoint: $CKPT_DIR"
            echo "  |-- 各轮次及最终评测结果: $EVAL_DIR"
            echo "  |-- 主动学习性能演化曲线: ${EVAL_DIR}/al_performance_curve.csv"
            echo "==========================================================="
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 训练异常退出 (exit=$TRAIN_EXIT)"
        fi
    ) > "$LOG_FILE" 2>&1 &

    PID=$!
    PIDS+=("$PID")
    echo "  |-- PID: $PID (后台运行中)"
    sleep 3  # 错开启动瞬间的初始化冲击
done

echo -e "\n================================================================================"
echo " 🎯 ${#AL_JOBS[@]} 个主动学习任务已全部成功启动到后台！"
echo " 任务 PID 列表: ${PIDS[*]}"
echo " 监控各卡训练状态命令:"
for LF in "${LOG_FILES[@]}"; do
    echo "   tail -f $LF"
done
echo "================================================================================"
