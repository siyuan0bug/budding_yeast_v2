#!/bin/bash
# 自动 cd 到脚本所在目录，无论终端 cwd 在哪都能正确运行
cd "$(dirname "$0")" || exit

export CUDA_VISIBLE_DEVICES=5
export NCCL_P2P_DISABLE=1
export PYTHONPATH=$(pwd)/..:$PYTHONPATH
#export WANDB_API_KEY="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"  #v1版本
export WANDB_API_KEY="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"  #active版本

# ==========================================
# 全局开关：设为 1 启用 W&B 离线模式（日志保存在本地 wandb/ 目录，不联网）
#           设为 0 恢复在线上传
# 离线日志恢复上传: wandb sync wandb/offline-run-<id>
# 批量上传所有离线日志: wandb sync --sync-all
# ==========================================
NO_WANDB=1

DATA_DIR="/home/users/zsy/mydata/budding_yeast"
DATASET_CHEAT="${DATA_DIR}/lhs_cheat_origin_210min_500steps_dual_labels.npz"
DATASET_BLIND="${DATA_DIR}/lhs_blind_210min_500steps_dual_labels.npz"

SEEDS=(42) #42,1024,3407          

# ==========================================
# 终极多任务实验清单
#
#
# 说明:
#   MODEL         : 模型名 (hyper_fno, cross_fno, pure_fno, patchtst, ...)
#   LOSS_TYPE     : 损失函数 (mse_only, physics_informed,pinn_residual)
#   USE_ADJ       : 是否用邻接矩阵 (true/false)
#   DATASET_PATH  : 数据集 npz 绝对路径
#   DATASET_TAG   : 数据集标签 (用于命名)
#   MODES         : Fourier modes 数 (Transformer 模型用作 d_model 参考)
#   WIDTH         : 隐藏层宽度
#   BATCH_SIZE    : 批大小
#   EPOCHS        : 最大训练轮数
#   AL_STRATEGY   : 主动学习策略 (none/random/us/is/wrs/vessal/hggs/rgs/pial)
#                   none=不用AL; random/us/is/wrs/vessal/hggs=标准池化AL; rgs=查询合成AL
#                   pial=Physics-Informed AL (ODE残差+轨迹多样性, 导师要求)
#                   所有策略统一使用K-Means初始训练集选择(默认5000样本)
#   LR            : 初始学习率
#   WARMUP        : Cosine+Warmup 预热轮数
#   AL_NUM_ADD    : 每轮 AL 新增样本数 (AL无关时填0)
#   AL_PERTURBATION: RGS局部扰动幅度 (非RGS时填0)
#   AL_MAE_THRESHOLD: RGS的MAE阈值 (非RGS时填0)
#   ROLLOUT_TYPE  : 预测方式 (one_time=一次性预测, autoregressive=自回归滚动)
# ==========================================

EXPERIMENTS=(
    # 格式 (16列，空格分隔):
#   MODEL LOSS_TYPE USE_ADJ DATASET_PATH DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE
    
    # 实验 1: RGS 主动学习 (减小新增样本数到500, 扰动幅度降到0.05)
    #"hyper_fno mse_only false ${DATASET_CHEAT} 32 32 64 200 rgs 6e-4 10 500 0.05 0.1 one_time"
    #"hyper_fno mse_only false ${DATASET_CHEAT} 32 32 64 200 us 6e-4 10 500 0 0 one_time"
    #"hyper_fno mse_only false ${DATASET_CHEAT} 32 32 64 200 is 6e-4 10 500 0 0. one_time"
    #"hyper_fno mse_only false ${DATASET_CHEAT} 32 32 64 200 wrs 6e-4 10 500 0 0. one_time"
    #"hyper_fno mse_only false ${DATASET_CHEAT} 32 32 64 200 vessal 6e-4 10 500 0 0. one_time"
    #"hyper_fno mse_only false ${DATASET_CHEAT} one_time 32 32 64 200 none 6e-4 10 0 0 0 one_time"

    # 实验: PatchTST (ICLR 2023) 时间序列预测
    # 注: autoregressive 模式需保留 499 步梯度图, 显存占用约 one_time 的 499 倍, 故默认用 one_time
    #"patchtst mse_only false ${DATASET_CHEAT} auto 32 128 64 200 none 6e-4 10 0 0 0 one_time"
    #"hyper_fno mse_only false ${DATASET_CHEAT} auto 32 32 32 200 none 6e-4 10 0 0 0 autoregressive"
    # 实验: Physics-Informed AL (ODE残差+轨迹多样性, 导师要求)
    #"hyper_fno pinn_residual false ${DATASET_CHEAT} one 32 32 64 200 none 6e-4 10 0 0 0 one_time"
    "hyper_fno mse_only false ${DATASET_CHEAT} one 32 32 64 200 pial 6e-4 10 500 0 0 one_time"
    


    #"cross_fno mse_only true ${DATASET_CHEAT} 32 16 96 200 rgs 9e-4 10 5000 0.1 0.1 one_time"
    # 实验 2: Baseline (无主动学习)
    #"hyper_fno mse_only false ${DATASET_CHEAT} 32 32 32 300 none 3e-4 10 0 0 0 one_time"

    # 实验 3: 随机基线
    #"cross_fno mse_only true ${DATASET_CHEAT} 32 16 32 120 random 3e-4 10 5000 0 0 one_time"

    # 实验 4: HGGS
    #"cross_fno mse_only true ${DATASET_CHEAT} hggs 3e-4 10 5000 0 0 one_time"
)

for SEED in "${SEEDS[@]}"; do
    for EXP in "${EXPERIMENTS[@]}"; do

        read -r MODEL LOSS_TYPE USE_ADJ DATASET DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE <<< "$EXP"

        # 兼容旧格式: 若未指定 ROLLOUT_TYPE 则默认 one_time
        if [ -z "$ROLLOUT_TYPE" ]; then
            ROLLOUT_TYPE="one_time"
        fi

        if [ "$USE_ADJ" = "true" ]; then
            ADJ_PARAM="--adj_matrix_path mydata/yeast_signed_adjacency_matrix.npy"
        else
            ADJ_PARAM=""
        fi

        # 构建 AL 专属参数 (只有策略非 none 时才传递)
        AL_PARAMS=""
        if [ "$AL_STRATEGY" != "none" ]; then
            # PIAL 策略推迟到 epoch 50 触发: 早期模型预测差, ODE 残差无意义
            # 其他策略保持 epoch 10 触发
            if [ "$AL_STRATEGY" = "pial" ]; then
                TRIGGER_EPOCH=50
            else
                TRIGGER_EPOCH=10
            fi
            AL_PARAMS="--al_strategy $AL_STRATEGY --al_trigger_epoch $TRIGGER_EPOCH --al_num_add $AL_NUM_ADD --al_perturbation $AL_PERTURBATION --al_mae_threshold $AL_MAE_THRESHOLD"
        else
            AL_PARAMS="--al_strategy none"
        fi

        echo "==========================================================="
        echo "🔥 正在处理 -> Seed: $SEED | 策略: $AL_STRATEGY | LR: $LR | Warmup: $WARMUP | Rollout: $ROLLOUT_TYPE"
        echo "==========================================================="

        RUN_NAME="${MODEL}_${AL_STRATEGY}_${LOSS_TYPE}_${DATASET_TAG}_modes${MODES}_seed${SEED}"
        EVAL_OUT_DIR="./eval_result/${RUN_NAME}"

        # 步骤 A: 训练
        # 根据 LOSS_TYPE 构建 PINN 专属参数
        PINN_PARAMS=""
        if [ "$LOSS_TYPE" = "pinn_residual" ]; then
            # v2: 物理约束作弱正则, Curriculum learning (warmup 30 轮纯 MSE)
            # v1 问题: lambda_phys=1.0 → val_loss=0.930 (baseline=0.037, 差 25 倍)
            # v1 问题: lambda_penalty=10.0 → 负值惩罚压制 MSE
            # v1 问题: event 掩码基于预测 → 早期掩码错误导致梯度爆炸
            # v2 修复: 掩码改用真值, 负值惩罚降到 1.0, warmup 30 轮
            PINN_PARAMS="--lambda_phys 0.01 --lambda_jump 0.001 --lambda_penalty 1.0 --residual_subsample 10 --event_weight 0.1"
        fi

        echo "▶️ [阶段 1/2] 训练阶段"
        WANDB_FLAG=""
        if [ "$NO_WANDB" = "1" ]; then
            WANDB_FLAG="--no_wandb"
        fi
        python train.py \
            --model $MODEL \
            --wavelet haar \
            --loss_type $LOSS_TYPE \
            --dataset_name "$DATASET" \
            --modes $MODES \
            --width $WIDTH \
            --batch_size $BATCH_SIZE \
            --max_epochs $EPOCHS \
            --seed $SEED \
            --name $RUN_NAME \
            --lr $LR \
            --warmup_epochs $WARMUP \
            --num_workers 4 \
            --val_batch_size 512 \
            --devices 1 \
            $AL_PARAMS \
            $ADJ_PARAM \
            $PINN_PARAMS \
            --rollout_type $ROLLOUT_TYPE \
            $WANDB_FLAG

        # 步骤 B: 寻找 Checkpoint
        CKPT_DIR="./train_result/${RUN_NAME}"
        CKPT_PATH=$(ls ${CKPT_DIR}/*.ckpt 2>/dev/null | head -n 1)

        if [ -z "$CKPT_PATH" ]; then
            echo "❌ 错误: 在 ${CKPT_DIR} 没找到 Checkpoint 文件，跳过 Evaluate。"
            continue
        fi

        echo "🎯 成功找到 Checkpoint: $CKPT_PATH"

        # 步骤 C: 评估（独立大 batch，充分利用显存加速）
        echo "▶️ [阶段 2/2] 开始评估与生成突变体图..."
        python eval.py \
            --ckpt "$CKPT_PATH" \
            --dataset_name "$DATASET" \
            --save_dir "$EVAL_OUT_DIR" \
            --batch_size 512

        echo "✅ 实验 $RUN_NAME 评估完成！结果已保存至: $EVAL_OUT_DIR"
        echo "-----------------------------------------------------------"
    done
done
