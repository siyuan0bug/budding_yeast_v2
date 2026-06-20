#!/bin/bash
# ============================================================
# Optuna 超参自动搜索脚本
# ============================================================
#
# 【这个脚本做什么？】
#   自动寻找让模型在 Real Mutants（真实突变体）上表现最好的超参数组合。
#   你只需在下方"用户配置区"填好模型名、数据集路径等，然后运行：
#       bash run_tune.sh
#   脚本会自动尝试上百组不同的超参组合，最终输出最优的一组。
#
# 【Optuna 是什么？】
#   一个超参自动搜索库。它不是"网格搜索"（穷举所有组合，太慢），
#   而是用"贝叶斯优化"算法：根据历史结果，智能猜测下一组该试什么参数。
#   实测比穷举快 5-10 倍。
#
# 【核心概念解释】
#   - trial（试验）：1 次 trial = 1 次完整训练 + 1 次评估。
#                   100 个 trial 意味着最多训练 100 次。
#   - 剪枝（Pruning）：训练中途发现这组参数表现太差，立即终止，
#                     跳到下一组。这样 100 个 trial 实际可能只完整跑 40-60 次，
#                     其余被提前终止，大幅节省算力。
#   - TPE 采样器：Optuna 的核心算法。前 10 次随机试，之后根据结果
#                 聚焦到"好参数"所在的区域采样。
#   - study（研究）：一次完整的搜索任务，所有 trial 历史都存在 SQLite 数据库，
#                   中断后可恢复续跑。
#
# 【搜索哪些超参？】（这些由 Optuna 自动尝试不同组合）
#   训练类: batch_size(256~4096), lr(1e-4~5e-2), weight_decay(1e-6~1e-2),
#           warmup_epochs(5~40), max_epochs(100~500)
#   模型类: modes(16~64), width(16~128), n_blocks(2~6)
#
# 【你固定什么？】（这些不参与搜索，由你在下方配置）
#   - 模型名（如 hyper_fno）
#   - 主动学习策略（如 none 或 rgs）
#   - 损失函数、ODE 参数等
#
# 【输出结果】
#   save_dir/best_params.json   —— 最优超参组合 + Real MAE + ckpt 路径
#   save_dir/all_trials.csv     —— 全部 trial 历史（可用 Excel 查看）
#   save_dir/optuna_study.db    —— 搜索数据库（用于恢复/可视化）
#
# 【中断后恢复】
#   搜索中途断了？没关系！再次运行本脚本（保持配置不变），
#   会自动从断点续跑，已完成的好 trial 不会丢失。搜索中途断了？把 RESUME=false 改成 RESUME=true ，再次 bash run_tune.sh ，自动从断点续跑。
#
# 【可视化结果】
#   pip install optuna-dashboard
#   optuna-dashboard sqlite:///./tune_result/optuna_study.db
#   然后浏览器打开 http://localhost:8080 看交互式图表
# ============================================================

# 自动 cd 到脚本所在目录（同 run_pipeline.sh）
cd "$(dirname "$0")" || exit

# ===== 环境变量（同 run_pipeline.sh）=====
export CUDA_VISIBLE_DEVICES=7          # 使用的 GPU 编号，按需修改
export NCCL_P2P_DISABLE=1
export PYTHONPATH=$(pwd)/..:$PYTHONPATH
# export WANDB_API_KEY="你的wandb_key"  # tune.py 默认关闭 wandb，无需配置

# ============================================================
# 用户配置区（只需修改这里 ↓↓↓）
# ============================================================

# --- 你要搜索的模型（固定，不参与搜索）---
MODEL="hyper_fno"                      # 可选: pure_fno, cross_fno, hyper_fno, hp_fno, ultimate_fno, wno, lno, neural_ode

# --- 主动学习策略（固定，不参与搜索）---
AL_STRATEGY="none"                     # 可选: none(不用AL), rgs(查询合成), random/us/is/wrs/vessal/hggs(标准池化AL)

# --- 损失函数（固定）---
LOSS_TYPE="mse_only"                   # 可选: mse_only, physics_informed, dilated_sobolev

# --- 数据集（固定）---
DATASET="/home/users/zsy/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels.npz"

# --- 搜索配置 ---
N_TRIALS=100                           # 总共尝试多少组超参组合（建议 100-200）
SAVE_DIR="./tune_result"               # 结果保存目录
SEED=42                                # 随机种子（保证可复现）
DEVICES=1                              # GPU 数量
NUM_WORKERS=4                          # DataLoader 进程数

# --- 主动学习超参（仅 AL_STRATEGY != none 时生效，固定不搜索）---
AL_TRIGGER_EPOCH=10                    # 每 N 个 epoch 触发一次 AL
AL_NUM_ADD=5000                        # 每轮 AL 新增样本数
AL_PERTURBATION=0.1                    # RGS 局部扰动幅度 (±10%)
AL_MAE_THRESHOLD=0.1                   # RGS 的 MAE 阈值

# --- 选项开关 ---
CLEANUP_CKPTS=true                     # true=评估后删除中间ckpt省磁盘(推荐), false=保留所有ckpt
RESUME=true                           # true=从上次中断处续跑, false=全新搜索
NO_PRUNE=false                         # true=关闭剪枝(不推荐,浪费算力), false=开启剪枝
PRUNE_WARMUP=10                        # 前 N 个 epoch 不剪枝，让模型先热身

# ============================================================
# 用户配置区结束（以下无需修改 ↑↑↑）
# ============================================================

# 把布尔值转换为命令行参数
CLEANUP_FLAG=""
if [ "$CLEANUP_CKPTS" = "true" ]; then CLEANUP_FLAG="--cleanup_ckpts"; fi

RESUME_FLAG=""
if [ "$RESUME" = "true" ]; then RESUME_FLAG="--resume"; fi

NO_PRUNE_FLAG=""
if [ "$NO_PRUNE" = "true" ]; then NO_PRUNE_FLAG="--no_prune"; fi

# 构建 AL 参数（仅当策略非 none 时传递）
AL_PARAMS=""
if [ "$AL_STRATEGY" != "none" ]; then
    AL_PARAMS="--al_trigger_epoch $AL_TRIGGER_EPOCH --al_num_add $AL_NUM_ADD --al_perturbation $AL_PERTURBATION --al_mae_threshold $AL_MAE_THRESHOLD"
fi

# 打印搜索概要
echo "============================================================"
echo "  Optuna 超参自动搜索"
echo "============================================================"
echo "  模型:         $MODEL"
echo "  AL 策略:      $AL_STRATEGY"
echo "  损失函数:     $LOSS_TYPE"
echo "  数据集:       $(basename $DATASET)"
echo "  总 trials:    $N_TRIALS"
echo "  结果目录:     $SAVE_DIR"
echo "  剪枝:         $([ "$NO_PRUNE" = "true" ] && echo '关闭' || echo "开启(热身${PRUNE_WARMUP}epoch)")"
echo "  续跑:         $([ "$RESUME" = "true" ] && echo '是' || echo '否')"
echo "  清理ckpt:     $([ "$CLEANUP_CKPTS" = "true" ] && echo '是' || echo '否')"
echo "============================================================"
echo ""
echo "  搜索空间（Optuna 自动尝试这些超参的不同组合）:"
echo "    batch_size:    256 / 512 / 1024 / 2048 / 4096"
echo "    lr:            1e-4 ~ 5e-2 (对数均匀采样)"
echo "    weight_decay:  1e-6 ~ 1e-2 (对数均匀采样)"
echo "    warmup_epochs: 5 ~ 40"
echo "    max_epochs:    100 ~ 500 (步长 50)"
echo "    modes:         16 / 24 / 32 / 48 / 64"
echo "    width:         16 / 32 / 48 / 64 / 96 / 128"
echo "    n_blocks:      2 ~ 6"
echo ""
echo "  优化目标: Real Mutants MAE (越小越好)"
echo "  每个 trial = 1次完整训练 + 1次Real评估，差的trial会被提前剪枝"
echo "============================================================"
echo ""

# 运行搜索
python tune.py \
    --model $MODEL \
    --al_strategy $AL_STRATEGY \
    --loss_type $LOSS_TYPE \
    --dataset_name "$DATASET" \
    --n_trials $N_TRIALS \
    --save_dir $SAVE_DIR \
    --seed $SEED \
    --devices $DEVICES \
    --num_workers $NUM_WORKERS \
    --prune_warmup_epochs $PRUNE_WARMUP \
    $CLEANUP_FLAG \
    $RESUME_FLAG \
    $NO_PRUNE_FLAG \
    $AL_PARAMS

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "  搜索完成！"
    echo "============================================================"
    echo ""
    echo "  结果文件:"
    echo "    $SAVE_DIR/best_params.json   —— 最优超参（重点看这个）"
    echo "    $SAVE_DIR/all_trials.csv     —— 全部 trial 历史"
    echo "    $SAVE_DIR/optuna_study.db    —— 搜索数据库"
    echo ""
    echo "  查看最优结果:"
    echo "    cat $SAVE_DIR/best_params.json"
    echo ""
    echo "  可视化（可选）:"
    echo "    pip install optuna-dashboard"
    echo "    optuna-dashboard sqlite:///$SAVE_DIR/optuna_study.db"
    echo "    # 然后浏览器打开 http://localhost:8080"
    echo ""
    echo "  用最优超参重新训练（拿到 best_params.json 后）:"
    echo "    # 查看 best_params.json 里的 params 字段，填入 train.py 参数"
    echo "    python train.py --model $MODEL --al_strategy $AL_STRATEGY \\"
    echo "        --dataset_name $DATASET --loss_type $LOSS_TYPE \\"
    echo "        --modes <最优modes> --width <最优width> --batch_size <最优bs> \\"
    echo "        --lr <最优lr> --warmup_epochs <最优warmup> --max_epochs <最优epochs>"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "  搜索出错！请检查上方错误信息。"
    echo "============================================================"
fi
