#!/bin/bash
# 自动 cd 到脚本所在目录，无论终端 cwd 在哪都能正确运行
cd "$(dirname "$0")" || exit

# 注意: CUDA_VISIBLE_DEVICES 不再全局设置, 改为每个实验在配置中单独指定 GPU_ID
export NCCL_P2P_DISABLE=1
export PYTHONPATH=$(pwd)/..:$PYTHONPATH
# 强制 Python stdout/stderr 无缓冲, 让 tqdm 进度条和 print 摘要行立即写入日志文件
export PYTHONUNBUFFERED=1
#export WANDB_API_KEY="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"  #v1版本
export WANDB_API_KEY="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"  #active版本

# 创建并行实验日志目录
mkdir -p logs

# ==========================================
# 全局开关：设为 1 启用 W&B 离线模式（日志保存在本地 wandb/ 目录，不联网）
#           设为 0 恢复在线上传
# 离线日志恢复上传: wandb sync wandb/offline-run-<id>
# 批量上传所有离线日志: wandb sync --sync-all
# ==========================================
NO_WANDB=0

DATA_DIR="/home/users/zsy/mydata/budding_yeast"
DATASET_CHEAT="${DATA_DIR}/lhs_cheat_origin_210min_500steps_dual_labels.npz"
DATASET_BLIND="${DATA_DIR}/lhs_blind_210min_500steps_dual_labels.npz"
# v2 数据集 (Sim-to-Real Gap 缓解: 三大家族定向扩展 + N_LHS 补足)
DATASET_V2="${DATA_DIR}/lhs_cheat_origin_v2_210min_500steps_dual_labels.npz"
# v3 数据集 (V2 基础 + 残留瓶颈定向扩展: 26 个 MAE>0.3 突变体, 5 类家族配置)
DATASET_V3="${DATA_DIR}/lhs_cheat_origin_v3_210min_500steps_dual_labels.npz"
# v4 数据集 (V3 基础 + 12 类扩展家族: 67 突变体, ~37.6K 定向样本)
#   1. KO_family_fixed (3 个, density 模式修复 V3 boundary 邻域污染)
#   2. tem1_ts_family (4 个, density 模式, 新增)
#   3. cdc14_ts_family (6 个, density 模式, 新增)
#   4. swi5_cdh1_family (4 个, density 模式, 新增)
#   5. cln1_cln2_KO_family (9 个, density 模式, 新增)
#   6. sic1_KO_family (3 个, density 模式, 新增)
#   7. GAL_CLB2_cdh1_family (3 个, density 模式, 新增)
#   8. cln_triple_extended (4→11 个, density 模式, 扩展)
#   9. CLB1_clb2_extended (2→4 个, boundary 模式 pert 0.2→0.3, 扩展)
#  10. CLB2_db_enhanced (7→10 个, density 模式 pert 0.15→0.20, 增强)
#  11. cdh1_contamination_fix (4 个, density 模式, 保留 V3)
#  12. contamination_fix (9 个, density 模式, 保留 V3)
DATASET_V4="${DATA_DIR}/lhs_cheat_origin_v4_210min_500steps_dual_labels.npz"
# v5 数据集 (V4 基础 + 7 项定向修复: 17 类家族, 88 突变体, ~54K 定向样本)
#   P0 修复: CLB1_clb2_extended (boundary→density, pert 0.30→0.50, num 500→1000)
#   P1 新增: cdc6_family (5 个, 600 样本/突变体)
#   P2 增强: tem1_ts_family (num 800→1200, pert 0.08→0.12)
#   P3 新增: CLB2_db_clb5_Raff_family (1 个, 400 样本)
#   P4 新增: APC_A_GAL_CLB2_family (1 个, 400 样本)
#   P5 新增: cdc15_net1_ts_family (2 个, 400 样本/突变体)
#   P6 新增: sic1_cdh1_family (2 个, 400 样本/突变体)
DATASET_V5="${DATA_DIR}/lhs_cheat_origin_v5_210min_500steps_dual_labels.npz"
# v5 无 target 副本 (去除 _LHS_T 定向采样, 只保留 Real Mutant + 普通 LHS, 用于对照实验)
DATASET_V5_NO_TARGET="${DATA_DIR}/lhs_cheat_origin_v5_no_target_210min_500steps_dual_labels.npz"
# v6 数据集 (V5 基础 + P0-P5 定向修复: 修复 tem1_ts 干扰及 CLB2_db_Gal 盲区等)
DATASET_V6="${DATA_DIR}/lhs_cheat_origin_v6_210min_500steps_dual_labels.npz"
# nofilter 数据集 (去除所有生成和清洗拦截，强制样本数一致)
DATASET_nofilter="${DATA_DIR}/lhs_cheat_origin_no_filter_210min_500steps_dual_labels.npz"
# guaranteed 数据集 (保留过滤拦截, 但补采保证每个 mutant 过滤后恰好 1000 个有效样本)
DATASET_guaranteed="${DATA_DIR}/lhs_cheat_origin_guaranteed_210min_500steps_dual_labels.npz"
# mutant_aware 策略专用: atlas CSV (含 lhs_MAE_min, lhs_n_samples 列)
ATLAS_CSV="./eval_result/old_hyper_fno_none_mse_only_210m_500s_modes32_seed42/mutant_atlas_merged.csv"

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
#   AL_STRATEGY   : 主动学习策略 (none/random/us/is/wrs/vessal/hggs/rgs/pial/mutant_aware)
#                   none=不用AL; random/us/is/wrs/vessal/hggs=标准池化AL; rgs=查询合成AL
#                   pial=Physics-Informed AL (ODE残差+轨迹多样性, 导师要求)
#                   mutant_aware=Mutant-Aware AL (基于atlas优先级加权采样, 需配合 ATLAS_CSV)
#                       利用 lhs_MAE_min + lhs_n_samples + 当前MAE 三维加权计算 priority,
#                       聚焦采样到 LHS 覆盖不足 + 模型表现差的突变体邻域; 若 atlas 不可用自动降级为 rgs
#                   所有策略统一使用K-Means初始训练集选择(默认5000样本)
#   LR            : 初始学习率
#   WARMUP        : Cosine+Warmup 预热轮数
#   AL_NUM_ADD    : 每轮 AL 新增样本数 (AL无关时填0)
#   AL_PERTURBATION: RGS局部扰动幅度 (非RGS时填0)
#   AL_MAE_THRESHOLD: RGS的MAE阈值 (非RGS时填0)
#   ROLLOUT_TYPE  : 预测方式 (one_time=一次性预测, autoregressive=自回归滚动)
#   GPU_ID        : 使用的 GPU 编号 (0-7, 每个实验独占一张卡, 并行执行)
# ==========================================

EXPERIMENTS=(
    # 格式 (17列，空格分隔):
#   MODEL LOSS_TYPE USE_ADJ DATASET_PATH DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE GPU_ID
    
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
    #"hyper_fno mse_only false ${DATASET_CHEAT} one 32 32 64 200 pial 6e-4 10 500 0 0 one_time"

    # [已注释] Mutant-Aware AL + v5 数据集 (已被 al4pde 框架替代)
    # "hyper_fno mse_only false ${DATASET_V5} v5 32 32 64 200 mutant_aware 6e-4 10 5000 0.05 0.1 one_time"

    # ====== al4pde 框架实验 v2: 初始 15000 (v1 K-Means) + AL 50000 (v5 候选池) ======
    # 4 种策略对比: Random / Uncertainty / Diversity / MAAL
    # 数据集: v1 (DATASET_CHEAT) 作为初始训练集, v5 (DATASET_V5) 作为 AL 候选池 (排除 real mutants)
    # AL 预算: 每 4 epoch 触发一次, 每次选 1000 样本, 共 50 轮 = 50,000 样本
    # 初始训练集: 15000 (K-Means 从 v1 LHS 常规样本中选取)
    # 候选池: v5 全部 LHS 样本 (~162,008, 含 target + LHS 常规, 排除 126 Real Mutant)
    # 内存优化: 候选池只加载 P_norm + Y_IC 到内存, 完整 Y 按需从磁盘加载
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 200 al4pde_random 6e-4 10 1000 0 0 one_time 0"
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 200 al4pde_uncertainty 6e-4 10 1000 0 0 one_time 1"
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 200 al4pde_diversity 6e-4 10 1000 0 0 one_time 2"
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 200 maal 6e-4 10 1000 0 0 one_time 3"

    # 对照实验: v5 无 target 数据集 (验证 target 定向采样对模型的贡献, 区分 target 贡献 vs 普通 LHS 贡献)
    #"hyper_fno mse_only false ${DATASET_V5_NO_TARGET} v5_no_target 32 32 64 200 none 6e-4 10 0 0 0 one_time 3"

    # 对照实验: 无任何拦截的数据集实验 (验证强制去重和不丢弃数据对性能的影响)
    #"hyper_fno mse_only false ${DATASET_nofilter} nofilter 32 32 64 200 none 6e-4 10 0 0 0 one_time 3"

    # 对照实验: guaranteed 数据集 (保留过滤拦截, 但补采保证每个 mutant 过滤后恰好 1000 个有效样本)
    #"hyper_fno mse_only false ${DATASET_guaranteed} guaranteed 32 32 64 200 none 6e-4 10 0 0 0 one_time 0"

    # V6 数据集实验 (V5 基础 + P0-P5 定向修复: tem1_ts 分离 + CLB2_db_Gal + clb5 等)
    # mutant_aware AL: 基于 atlas 优先级加权采样, epoch 100 触发, 每5轮 AL
    #"hyper_fno mse_only false ${DATASET_V6} v6 32 32 64 200 mutant_aware 6e-4 10 5000 0.05 0.1 one_time 0"
    # none baseline: 纯 v6 数据集无 AL, 验证 v6 数据集本身的扩展修复效果
    #"hyper_fno mse_only false ${DATASET_V6} v6 32 32 64 200 none 6e-4 10 0 0 0 one_time 3"

    # 对照实验: v5 数据集 + 无 AL (验证 v5 数据集本身的扩展修复效果, 区分数据贡献 vs AL 贡献)
    "hyper_fno mse_only false ${DATASET_V5} v5 32 32 64 200 none 6e-4 10 0 0 0 one_time 0"

    # [已完成] V4 实验 (保留供参考, 已有 eval 结果)
    #"hyper_fno mse_only false ${DATASET_V4} v4 32 32 64 200 mutant_aware 6e-4 10 5000 0.05 0.1 one_time"
    #"hyper_fno mse_only false ${DATASET_V4} v4 32 32 64 200 none 6e-4 10 0 0 0 one_time"

    # 对照实验: v3 数据集 + mutant_aware AL (验证 v4 数据集相对 v3 的提升)
    #"hyper_fno mse_only false ${DATASET_V3} v3 32 32 64 200 mutant_aware 6e-4 10 5000 0.05 0.1 one_time"

    # 对照实验: v3 数据集 + 无 AL (V3 baseline, 已完成, 仅供参考)
    #"hyper_fno mse_only false ${DATASET_V3} v3 32 32 64 200 none 6e-4 10 0 0 0 one_time"

    # 对照实验: v2 数据集 + 无 AL (V2 baseline, 已完成, 仅供参考)
    #"hyper_fno mse_only false ${DATASET_V2} v2 32 32 64 200 none 6e-4 10 0 0 0 one_time"
    


    #"cross_fno mse_only true ${DATASET_CHEAT} 32 16 96 200 rgs 9e-4 10 5000 0.1 0.1 one_time"
    # 实验 2: Baseline (无主动学习)
    #"hyper_fno mse_only false ${DATASET_CHEAT} 32 32 32 300 none 3e-4 10 0 0 0 one_time"

    # 实验 3: 随机基线
    #"cross_fno mse_only true ${DATASET_CHEAT} 32 16 32 120 random 3e-4 10 5000 0 0 one_time"

    # 实验 4: HGGS
    #"cross_fno mse_only true ${DATASET_CHEAT} hggs 3e-4 10 5000 0 0 one_time"
)

# ====== 并行执行: 每个实验在指定 GPU 上同时运行 ======
# 变量在子 shell 中自动继承快照, 不会被下次循环迭代覆盖
PIDS=()

for SEED in "${SEEDS[@]}"; do
    for EXP in "${EXPERIMENTS[@]}"; do

        read -r MODEL LOSS_TYPE USE_ADJ DATASET DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE GPU_ID <<< "$EXP"

        # 兼容旧格式: 若未指定 GPU_ID 则默认 0
        if [ -z "$GPU_ID" ]; then
            GPU_ID=0
        fi

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
            # mutant_aware 策略推迟到 epoch 100 触发: 需模型基本收敛后再做 atlas 优先级评估, 否则 priority 基于错误的模型状态
            # al4pde/maal 策略每 4 epoch 触发: 200 epochs / 4 = 50 轮 AL (总 50,000 样本)
            # 其他策略保持 epoch 10 触发
            if [ "$AL_STRATEGY" = "pial" ]; then
                TRIGGER_EPOCH=50
            elif [ "$AL_STRATEGY" = "mutant_aware" ]; then
                TRIGGER_EPOCH=100
            elif [[ "$AL_STRATEGY" == al4pde_* || "$AL_STRATEGY" = "maal" ]]; then
                TRIGGER_EPOCH=4
            else
                TRIGGER_EPOCH=10
            fi
            AL_PARAMS="--al_strategy $AL_STRATEGY --al_trigger_epoch $TRIGGER_EPOCH --al_num_add $AL_NUM_ADD --al_perturbation $AL_PERTURBATION --al_mae_threshold $AL_MAE_THRESHOLD"

            # mutant_aware 策略专属参数:
            # 必须传入 atlas CSV 路径 (含 lhs_MAE_min, lhs_n_samples 列)
            # 若文件不存在, al_callback 会自动降级为 rgs 策略
            # 注: al_initial_train_size 参数对 mutant_aware 无效 (al_callback.py 中查询合成策略跳过 K-Means, 直接用全量数据)
            if [ "$AL_STRATEGY" = "mutant_aware" ]; then
                AL_PARAMS="$AL_PARAMS --atlas_csv $ATLAS_CSV"
            fi

            # al4pde/maal 策略专属参数:
            # --al_pool_dataset: v5 数据集路径 (含 target 样本 _LHS_T 后缀)
            # --al_initial_train_size 15000: K-Means 从 v1 LHS 常规样本中选 15000 个初始训练集
            if [[ "$AL_STRATEGY" == al4pde_* || "$AL_STRATEGY" = "maal" ]]; then
                AL_PARAMS="$AL_PARAMS --al_pool_dataset $DATASET_V5 --al_initial_train_size 15000"
            fi
        else
            AL_PARAMS="--al_strategy none"
        fi

        RUN_NAME="${MODEL}_${AL_STRATEGY}_${LOSS_TYPE}_${DATASET_TAG}_modes${MODES}_seed${SEED}"
        EVAL_OUT_DIR="./eval_result/${RUN_NAME}"
        LOG_FILE="logs/${RUN_NAME}.log"

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

        WANDB_FLAG=""
        if [ "$NO_WANDB" = "1" ]; then
            WANDB_FLAG="--no_wandb"
        fi

        echo "🚀 启动实验 -> Seed: $SEED | 策略: $AL_STRATEGY | GPU: $GPU_ID | 日志: $LOG_FILE"

        # 后台并行执行 train + eval (子 shell 继承当前变量快照)
        {
            echo "===========================================================" > "$LOG_FILE" 2>&1
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始训练: $RUN_NAME (GPU=$GPU_ID)" >> "$LOG_FILE" 2>&1
            echo "===========================================================" >> "$LOG_FILE" 2>&1

            # 步骤 A: 训练
            CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
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
                $WANDB_FLAG >> "$LOG_FILE" 2>&1

            TRAIN_EXIT=$?
            if [ $TRAIN_EXIT -ne 0 ]; then
                echo "❌ 训练失败 (exit=$TRAIN_EXIT), 跳过评估" >> "$LOG_FILE" 2>&1
                exit 1
            fi

            # 步骤 B: 寻找 Checkpoint
            CKPT_DIR="./train_result/${RUN_NAME}"
            CKPT_PATH=$(ls ${CKPT_DIR}/*.ckpt 2>/dev/null | head -n 1)

            if [ -z "$CKPT_PATH" ]; then
                echo "❌ 错误: 在 ${CKPT_DIR} 没找到 Checkpoint 文件，跳过 Evaluate。" >> "$LOG_FILE" 2>&1
                exit 1
            fi

            echo "🎯 成功找到 Checkpoint: $CKPT_PATH" >> "$LOG_FILE" 2>&1

            # 步骤 C: 评估（独立大 batch，充分利用显存加速）
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始评估..." >> "$LOG_FILE" 2>&1
            CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
                --ckpt "$CKPT_PATH" \
                --dataset_name "$DATASET" \
                --save_dir "$EVAL_OUT_DIR" \
                --batch_size 512 >> "$LOG_FILE" 2>&1

            if [ $? -eq 0 ]; then
                echo "✅ [$RUN_NAME] 评估完成！结果已保存至: $EVAL_OUT_DIR" >> "$LOG_FILE" 2>&1
            else
                echo "❌ [$RUN_NAME] 评估失败" >> "$LOG_FILE" 2>&1
            fi
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实验结束: $RUN_NAME" >> "$LOG_FILE" 2>&1
        } &
        PIDS+=($!)

    done
done

# 等待所有后台实验完成
echo ""
echo "⏳ 等待 ${#PIDS[@]} 个并行实验完成..."
echo "   实时查看日志: tail -f logs/*.log"
echo "   查看某个实验: tail -f logs/<实验名>.log"
echo ""

FAIL_COUNT=0
for pid in "${PIDS[@]}"; do
    wait $pid
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "⚠️ PID $pid 异常退出 (code=$STATUS)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 所有 ${#PIDS[@]} 个实验全部成功完成!"
else
    echo "⚠️ ${#PIDS[@]} 个实验中有 $FAIL_COUNT 个失败, 请检查 logs/ 目录下的日志"
fi
echo "📊 结果目录: ./eval_result/"
echo "📋 日志目录: ./logs/"
