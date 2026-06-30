#!/bin/bash
# ============================================================
# 过滤训练集实验脚本: 只用 MAE > 阈值的差母体 LHS 子样本训练
# ============================================================
#
# 【这个脚本做什么？】
#   验证真实 mutant 指标差是否与数据集分布有关。
#   从 per_mutant_metrics.csv 提取 MAE > 阈值的差母体,
#   只保留这些母体的 LHS 子样本作为训练集 (验证集/测试集不变),
#   用 best_params.json 的超参训练, 最后评估并对比前后 MAE。
#
# 【与 run_pipeline.sh 的区别】
#   - 训练集被过滤 (只含差母体的 LHS 子样本)
#   - 超参来自 best_params.json (Optuna 搜索结果)
#   - 输出文件夹带特殊 tag: filtered_poor_mae{threshold}
#   - 内置防过拟合措施 (EarlyStopping + 增大 weight_decay + 降 epochs + gap 监控)
#
# 【输出结果】
#   train_result/{run_name}/best-*.ckpt              —— 最佳模型
#   eval_result/{run_name}/global_metrics_denorm.json —— 全局评估指标
#   eval_result/{run_name}/per_mutant_metrics.csv     —— 每个突变体的 MAE
#   eval_result/{run_name}/poor_mutants_comparison.csv —— 差母体训练前后对比
#   eval_result/{run_name}/real_mutants/*.png         —— 轨迹对比图
#
# 【中断后恢复】
#   训练中断后重新运行即可 (ModelCheckpoint 会保存最佳模型)
#   若想从头开始, 删除 train_result/{run_name}/ 目录
# ============================================================

# 自动 cd 到脚本所在目录
cd "$(dirname "$0")" || exit

# ===== 环境变量（同 run_pipeline.sh）=====
export CUDA_VISIBLE_DEVICES=3                # 使用的 GPU 编号，按需修改 (0/1/2/.../7)
export NCCL_P2P_DISABLE=1
export PYTHONPATH=$(pwd)/..:$PYTHONPATH
# export WANDB_API_KEY="你的wandb_key"  # 默认离线模式，无需配置

# ===== Python 解释器 =====
PYTHON=/home/users/zsy/.conda/envs/neuralop/bin/python

# ============================================================
# 用户配置区（只需修改这里 ↓↓↓）
# ============================================================

# --- 输入文件路径 ---
CSV_PATH="./eval_result/hyper_fno_none_mse_only_210m_500s_modes32_seed42/per_mutant_metrics.csv"
BEST_PARAMS="./tune_result/best_params.json"
DATASET="/home/users/zsy/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels.npz"

# --- 实验参数 ---
MAE_THRESHOLD=1.8                     # MAE 阈值，大于此值的真实突变体母体将被保留
SEED=42                                # 随机种子
DEVICES=1                              # GPU 数量
NUM_WORKERS=4                          # DataLoader 进程数
VAL_BATCH_SIZE=512                     # 验证/测试 batch size

# --- 防过拟合参数 (针对小数据集 < 1万样本 重新配置) ---
EARLY_STOPPING_PATIENCE=15             # EarlyStopping patience (0=禁用)，小数据集容易过拟合，缩短耐心
WEIGHT_DECAY_OVERRIDE=1e-3             # 覆盖 weight_decay (原 1.72e-6 太小，增大约 600 倍)
MAX_EPOCHS_OVERRIDE=100                # 覆盖 max_epochs (原 200，小数据集不需要那么多轮)
BATCH_SIZE_OVERRIDE=32                 # 覆盖 batch_size (原 64，小数据集用小 batch 增加梯度更新次数)
LR_OVERRIDE=5e-4                       # 覆盖 lr (原 1.11e-3，降低学习率防止小数据集上震荡)
WARMUP_EPOCHS_OVERRIDE=5               # 覆盖 warmup_epochs (原 10，总 epoch 少了预热也要缩短)

# --- 选项开关 ---
NO_WANDB=true                          # true=离线模式(推荐), false=在线上传
TAG=""                                 # 实验 tag (留空则自动用 filtered_poor_mae{threshold})

# ============================================================
# 用户配置区结束（以下无需修改 ↑↑↑）
# ============================================================

# 把布尔值转换为命令行参数
WANDB_FLAG=""
if [ "$NO_WANDB" = "true" ]; then
    WANDB_FLAG="--no_wandb"
fi

# 构建 tag (若用户未指定)
if [ -z "$TAG" ]; then
    TAG="filtered_poor_mae${MAE_THRESHOLD}"
fi

# 推导 run_name (与 train_eval_filtered.py 内部逻辑一致)
# run_name = "{model}_{tag}_seed{seed}"
# model 从 best_params.json 读取 (这里先占位, 脚本会自动处理)
RUN_NAME_DISPLAY="hyper_fno_${TAG}_seed${SEED}"

# 打印实验概要
echo "============================================================"
echo "  过滤训练集实验: 只用差母体的 LHS 子样本训练"
echo "============================================================"
echo "  输入 CSV:        $CSV_PATH"
echo "  最优超参:        $BEST_PARAMS"
echo "  数据集:          $(basename $DATASET)"
echo "  MAE 阈值:        $MAE_THRESHOLD"
echo "  实验 tag:        $TAG"
echo "  预期 run_name:   $RUN_NAME_DISPLAY"
echo "  Seed:            $SEED"
echo "  GPU:             CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "------------------------------------------------------------"
echo "  防过拟合措施 (小数据集优化):"
echo "    EarlyStopping patience: $EARLY_STOPPING_PATIENCE"
echo "    batch_size:             $BATCH_SIZE_OVERRIDE (原 64)"
echo "    lr:                     $LR_OVERRIDE (原 1.11e-3)"
echo "    weight_decay:           $WEIGHT_DECAY_OVERRIDE (原 1.72e-6)"
echo "    warmup_epochs:          $WARMUP_EPOCHS_OVERRIDE (原 10)"
echo "    max_epochs:             $MAX_EPOCHS_OVERRIDE (原 200)"
echo "    train/val gap 监控:     启用"
echo "------------------------------------------------------------"
echo "  W&B 模式:        $([ "$NO_WANDB" = "true" ] && echo '离线' || echo '在线')"
echo "============================================================"
echo ""

# ===== 步骤 1: 检查输入文件是否存在 =====
echo "▶️ [阶段 0/3] 检查输入文件..."
MISSING=0
if [ ! -f "$CSV_PATH" ]; then
    echo "❌ 错误: CSV 文件不存在: $CSV_PATH"
    MISSING=1
fi
if [ ! -f "$BEST_PARAMS" ]; then
    echo "❌ 错误: best_params.json 不存在: $BEST_PARAMS"
    MISSING=1
fi
if [ ! -f "$DATASET" ]; then
    echo "❌ 错误: 数据集不存在: $DATASET"
    MISSING=1
fi
if [ "$MISSING" = "1" ]; then
    echo "请检查文件路径后重试。"
    exit 1
fi
echo "✅ 所有输入文件检查通过"
echo ""

# ===== 步骤 2: 运行训练+评估 =====
echo "▶️ [阶段 1/3] 开始训练 + 评估 (train_eval_filtered.py)..."
echo "============================================================"

$PYTHON train_eval_filtered.py \
    --csv_path "$CSV_PATH" \
    --mae_threshold $MAE_THRESHOLD \
    --best_params "$BEST_PARAMS" \
    --dataset_name "$DATASET" \
    --tag "$TAG" \
    --seed $SEED \
    --devices $DEVICES \
    --num_workers $NUM_WORKERS \
    --val_batch_size $VAL_BATCH_SIZE \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --weight_decay_override $WEIGHT_DECAY_OVERRIDE \
    --max_epochs_override $MAX_EPOCHS_OVERRIDE \
    --batch_size_override $BATCH_SIZE_OVERRIDE \
    --lr_override $LR_OVERRIDE \
    --warmup_epochs_override $WARMUP_EPOCHS_OVERRIDE \
    $WANDB_FLAG

# 检查运行结果
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "  ❌ 训练/评估出错！退出码: $EXIT_CODE"
    echo "  请检查上方错误信息。"
    echo "============================================================"
    exit $EXIT_CODE
fi

echo ""
echo "============================================================"
echo "  ✅ 训练 + 评估完成！"
echo "============================================================"

# ===== 步骤 3: 显示结果文件位置 =====
EVAL_OUT_DIR="./eval_result/${RUN_NAME_DISPLAY}"
TRAIN_OUT_DIR="./train_result/${RUN_NAME_DISPLAY}"

echo ""
echo "  结果文件位置:"
echo "    训练 checkpoint:   $TRAIN_OUT_DIR/"
echo "    评估结果:          $EVAL_OUT_DIR/"
echo ""
echo "  关键文件:"
echo "    $EVAL_OUT_DIR/global_metrics_denorm.json   —— 全局评估指标"
echo "    $EVAL_OUT_DIR/per_mutant_metrics.csv       —— 每个突变体的 MAE"
echo "    $EVAL_OUT_DIR/poor_mutants_comparison.csv  —— 差母体训练前后对比 (重点)"
echo "    $EVAL_OUT_DIR/real_mutants/*.png           —— 轨迹对比图"
echo ""
echo "  查看差母体对比结果:"
echo "    cat $EVAL_OUT_DIR/poor_mutants_comparison.csv"
echo ""
echo "  查看全局指标:"
echo "    cat $EVAL_OUT_DIR/global_metrics_denorm.json"
echo "============================================================"
