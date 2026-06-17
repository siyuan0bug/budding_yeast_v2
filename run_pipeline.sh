#!/bin/bash
# 自动 cd 到脚本所在目录，无论终端 cwd 在哪都能正确运行
cd "$(dirname "$0")" || exit

export CUDA_VISIBLE_DEVICES=7
export NCCL_P2P_DISABLE=1
export PYTHONPATH=$(pwd)/..:$PYTHONPATH
#export WANDB_API_KEY="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"  #v1版本
export WANDB_API_KEY="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"  #active版本

SEEDS=(42) #42,1024,3407          

# ==========================================
# 终极多任务实验清单
#
#
# 说明:
#   MODEL         : 模型名 (hyper_fno, cross_fno, pure_fno, ...)
#   LOSS_TYPE     : 损失函数 (mse_only, physics_informed)
#   USE_ADJ       : 是否用邻接矩阵 (true/false)
#   DATASET_PATH  : 数据集 npz 绝对路径
#   DATASET_TAG   : 数据集标签 (用于命名)
#   MODES         : Fourier modes 数
#   WIDTH         : 隐藏层宽度
#   BATCH_SIZE    : 批大小
#   EPOCHS        : 最大训练轮数
#   AL_STRATEGY   : 主动学习策略 (none/random/us/is/wrs/vessal/hggs/rgs)
#   LR            : 初始学习率
#   WARMUP        : Cosine+Warmup 预热轮数
#   AL_NUM_ADD    : 每轮 AL 新增样本数 (AL无关时填0)
#   AL_PERTURBATION: RGS局部扰动幅度 (非RGS时填0)
#   AL_MAE_THRESHOLD: RGS的MAE阈值 (非RGS时填0)
# ==========================================

EXPERIMENTS=(
    # 格式 (15列，空格分隔):
#   MODEL LOSS_TYPE USE_ADJ DATASET_PATH DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD
    
    # 实验 1: RGS 主动学习
    #"hyper_fno mse_only false /home/users/zsy/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels.npz 210m_500s 32 32 64 200 rgs 6e-4 10 5000 0.1 0.1"
    "hyper_fno mse_only false /home/users/zsy/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels.npz 210m_500s 32 32 64 200 none 6e-4 10 0 0.1 0.1"
    #"cross_fno mse_only true /home/users/zsy/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels.npz 210m_500s 32 16 96 200 rgs 9e-4 10 5000 0.1 0.1"
    # 实验 2: Baseline (无主动学习)
    #"hyper_fno mse_only false /home/users/zsy/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels.npz 210m_500s 32 32 32 300 none 3e-4 10 0 0 0"

    # 实验 3: 随机基线
    #"cross_fno mse_only true /home/users/zsy/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels.npz 210m_500s 32 16 32 120 random 3e-4 10 5000 0 0"

    # 实验 4: HGGS
    #"cross_fno mse_only true /home/users/zsy/mydata/budding_yeast/lhs_cheat_origin_210min_500steps_dual_labels.npz 210m_500s 32 16 32 120 hggs 3e-4 10 5000 0 0"
)

for SEED in "${SEEDS[@]}"; do
    for EXP in "${EXPERIMENTS[@]}"; do

        read -r MODEL LOSS_TYPE USE_ADJ DATASET DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD <<< "$EXP"

        if [ "$USE_ADJ" = "true" ]; then
            ADJ_PARAM="--adj_matrix_path mydata/yeast_signed_adjacency_matrix.npy"
        else
            ADJ_PARAM=""
        fi

        # 构建 AL 专属参数 (只有策略非 none 时才传递)
        AL_PARAMS=""
        if [ "$AL_STRATEGY" != "none" ]; then
            AL_PARAMS="--al_strategy $AL_STRATEGY --al_trigger_epoch 10 --al_num_add $AL_NUM_ADD --al_perturbation $AL_PERTURBATION --al_mae_threshold $AL_MAE_THRESHOLD"
        else
            AL_PARAMS="--al_strategy none"
        fi

        echo "==========================================================="
        echo "🔥 正在处理 -> Seed: $SEED | 策略: $AL_STRATEGY | LR: $LR | Warmup: $WARMUP"
        echo "==========================================================="

        RUN_NAME="${MODEL}_${AL_STRATEGY}_${LOSS_TYPE}_${DATASET_TAG}_modes${MODES}_seed${SEED}"
        EVAL_OUT_DIR="./eval_result/${RUN_NAME}"

        # 步骤 A: 训练
        echo "▶️ [阶段 1/2] 训练阶段"
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
            $ADJ_PARAM

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
