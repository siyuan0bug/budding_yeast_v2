#!/bin/bash
# 自动 cd 到脚本所在目录，无论终端 cwd 在哪都能正确运行
cd "$(dirname "$0")" || exit

# 注意: CUDA_VISIBLE_DEVICES 不再全局设置, 改为每个实验在配置中单独指定 GPU_ID
export NCCL_P2P_DISABLE=1
export PYTHONPATH=$(pwd)/..:$PYTHONPATH
# 强制 Python stdout/stderr 无缓冲, 让 tqdm 进度条和 print 摘要行立即写入日志文件
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN="/home/users/zsy/.conda/envs/neuralop/bin/python"
export WANDB_API_KEY="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"  

# 创建并行实验日志目录
mkdir -p logs_v3 logs

# ==========================================
# 全局开关：设为 1 启用 W&B 离线模式（日志保存在本地 wandb/ 目录，不联网）
#           设为 0 恢复在线上传
# 离线日志恢复上传: wandb sync wandb/offline-run-<id>
# 批量上传所有离线日志: wandb sync --sync-all
# ==========================================
NO_WANDB=1

DATA_DIR="/home/users/zsy/mydata/budding_yeast"

# MASS 数据集 (mass_dataset_generator.py 生成: 126 Real Mutants + MASS 合成样本, 无 LHS 家族结构)
#   样本命名: real mutants 用原名 (如 1_WT_Glc), 合成样本以 'MASS_' 开头 (如 MASS_00001)
#   yeast_datamodule.py 会自动识别 (无 '_LHS_' + 有 'MASS_' 前缀) 并跳过家族分层, 仅按 Pattern 宏观划分
DATASET_MASS="${DATA_DIR}/mass_dataset_210min_500steps_dual_labels.npz"
# MASS v4.5 / v2 数据集 (全新升级: 结构化生物救援基序扩充 + 高阶多突变占比提升至75%)
DATASET_MASS_V2="${DATA_DIR}/mass_v2_dataset_210min_500steps_dual_labels.npz"
# MASS 100,000 数据集 (含生物救援基序 Motif, 10万样本全量高精数据集)
DATASET_MASS_100000="${DATA_DIR}/mass_100000_dataset_210min_500steps_dual_labels.npz"
# MASS 100,000 Random 数据集 (无 Motif 纯随机基线, 10万样本对照数据集)
DATASET_MASS_100000_RANDOM="${DATA_DIR}/mass_100000_random_dataset_210min_500steps_dual_labels.npz"
# MASS 50,000 数据集 (5万样本规模对比数据集)
DATASET_MASS_50000="${DATA_DIR}/mass_50000_dataset_210min_500steps_dual_labels.npz"
# 旧版 MASS v3 数据集 (已注释归档)
#DATASET_MASS_V3="${DATA_DIR}/massv3_dataset_210min_500steps_dual_labels.npz"
# mutant_aware 策略专用: atlas CSV (含 lhs_MAE_min, lhs_n_samples 列)
ATLAS_CSV="./eval_result/old_hyper_fno_none_mse_only_210m_500s_modes32_seed42/mutant_atlas_merged.csv"

SEEDS=(42) #42,1024,3407          

# ==========================================
# 终极多任务实验清单
#
# 说明:
#   MODEL         : 模型名 (biom_refiner_plus, biom_refiner_curvature, fno1d, unet1d, galerkin_transformer1d, cno1d, mwt1d, transolver1d, deeponet, ...)
#   LOSS_TYPE     : 损失函数 (hybrid_mse_rel, triple_dynamic, mse_only, phase_derivative, soft_dtw, pinn_residual, pino)
#   USE_ADJ       : 是否用邻接矩阵 (true/false)
#   DATASET_PATH  : 数据集 npz 绝对路径
#   DATASET_TAG   : 数据集标签 (用于命名)
#   MODES         : Fourier modes 数 (Transformer 模型用作 d_model 参考)
#   WIDTH         : 隐藏层宽度
#   BATCH_SIZE    : 批大小
#   EPOCHS        : 最大训练轮数
#   AL_STRATEGY   : 主动学习策略 (none/random/us/is/wrs/vessal/hggs/rgs/pial/mutant_aware/al4pde_novelty/al4pde_random/al4pde_diversity/maal/al4pde_mass)
#   LR            : 初始学习率
#   WARMUP        : Cosine+Warmup 预热轮数
#   AL_NUM_ADD    : 每轮 AL 新增样本数 (AL无关时填0)
#   AL_PERTURBATION: RGS局部扰动幅度 (非RGS时填0)
#   AL_MAE_THRESHOLD: RGS的MAE阈值 (非RGS时填0)
#   ROLLOUT_TYPE  : 预测方式 (one_time=一次性预测, autoregressive=自回归滚动)
#   GPU_ID        : 使用的 GPU 编号 (0-7, 同卡自动串行排队, 跨卡并行执行)
# ==========================================

EXPERIMENTS=(
    # 格式 (17列，空格分隔):
#   MODEL LOSS_TYPE USE_ADJ DATASET_PATH DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE GPU_ID

    # # ====== 👑👑👑 1. 旗舰最强机制模型在 MASS 100,000 (含 Motif) 数据集上的决战配置 👑👑👑 ======
    # # GPU 0: 👑 全项目最高精三元动力学终极大融合旗舰 biom_refiner_plus + Triple-Dynamic 复合损失
    # "biom_refiner_plus triple_dynamic false ${DATASET_MASS_100000} mass100k_biom_refiner_plus_triple_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    # # GPU 1: 👑 超高精破百纪录保持者 biom_refiner_curvature + Triple-Dynamic 复合损失 (二阶曲率加速度模型)
    # "biom_refiner_curvature triple_dynamic false ${DATASET_MASS_100000} mass100k_biom_refiner_curv_triple_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    # # GPU 2: 👑 生物化学计量反应图拓扑注意力模型 biom_refiner_graph + Hybrid 损失
    # "biom_refiner_graph hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_biom_refiner_graph_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    # # GPU 3: 👑 经典双阶粗精细化基准模型 BioRefiner + Hybrid 损失
    # "biom_refiner hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_biom_refiner_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 3"

    # # ====== 🌟🌟🌟 2. RealPDEBench & 经典基线神经算子全家桶 (MASS 100,000 含 Motif 数据集) 🌟🌟🌟 ======
    # # GPU 4: RealPDEBench 标准 1D FNO (通道级频域交互 + FiLM 参数调制)
    # "fno1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_fno1d_standard_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    # # GPU 5: RealPDEBench / PDEBench 标准 1D U-Net 算子 (多尺度残差跳跃连接)
    # "unet1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_unet1d_hybrid_w64 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    # # GPU 6: RealPDEBench 连续希尔伯特函数空间线性注意力 Galerkin Transformer
    # "galerkin_transformer1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_galerkin_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 6"
    # # GPU 7: Lu Lu et al. 经典 DeepONet (Branch-Trunk 深度算子网络)
    # "deeponet hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_deeponet_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 7"
    # # GPU 4: ICLR 2023 RealPDEBench 连续带限抗混叠卷积神经算子 (CNO1d)
    # "cno1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_cno1d_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    # # GPU 5: NeurIPS 2021 RealPDEBench 多小波连续神经算子 (MWT1d)
    # "mwt1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_mwt1d_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    # # GPU 6: RealPDEBench 物理切片注意力算子 (Transolver1d)
    # "transolver1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_transolver1d_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 6"
    # # GPU 1: ICLR/NeurIPS 2024 RealPDEBench 双路径算子 Transformer (DPOT1d / AFNO)
    # "dpot1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_dpot1d_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    # # GPU 7: 1D 空洞残差网络 (ResNetFiLM + 通道注意力)
    # "resnet_film hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_resnet_film_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 7"
    # # GPU 2: 超网络调制傅里叶神经算子 (HyperFNO)
    # "hyper_fno hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_hyper_fno_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    # # GPU 3: ICLR 2021 原版单场 FNO (PureFNO, 单变量展开 BS=256 安全防 OOM)
    # "pure_fno hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_pure_fno_hybrid_w64 32 64 256 100 none 1.7e-3 10 0 0 0 one_time 3"
    # # GPU 0: CMAME 2022 原版小波算子 (PureWNO, BS=256 安全防 OOM)
    # "pure_wno hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_pure_wno_hybrid_w64 32 64 256 100 none 1.7e-3 10 0 0 0 one_time 0"

    # # ====== 🎲🎲🎲 3. 全模型在 MASS 100,000 Random (无 Motif 纯随机基线) 数据集上的消融对照 🎲🎲🎲 ======
    # # GPU 0: 👑 大融合旗舰 biom_refiner_plus (Random 100k 对照)
    # "biom_refiner_plus triple_dynamic false ${DATASET_MASS_100000_RANDOM} random100k_biom_refiner_plus_triple_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    # # GPU 1: 👑 二阶曲率模型 biom_refiner_curvature (Random 100k 对照)
    # "biom_refiner_curvature triple_dynamic false ${DATASET_MASS_100000_RANDOM} random100k_biom_refiner_curv_triple_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    # # GPU 2: 👑 生物化学计量图拓扑注意力 biom_refiner_graph (Random 100k 对照)
    # "biom_refiner_graph hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_biom_refiner_graph_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    # # GPU 3: 👑 双阶粗精细化基准 BioRefiner (Random 100k 对照)
    # "biom_refiner hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_biom_refiner_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    # GPU 4: RealPDEBench 标准 1D FNO (Random 100k 对照)
    "fno1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_fno1d_standard_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    # GPU 5: RealPDEBench 1D U-Net (Random 100k 对照)
    "unet1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_unet1d_hybrid_w64 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    # GPU 6: RealPDEBench Galerkin Transformer (Random 100k 对照)
    "galerkin_transformer1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_galerkin_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    # GPU 7: Lu Lu 经典 DeepONet (Random 100k 对照)
    "deeponet hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_deeponet_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    # GPU 4: RealPDEBench CNO1d (Random 100k 对照)
    "cno1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_cno1d_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    # GPU 5: RealPDEBench MWT1d (Random 100k 对照)
    "mwt1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_mwt1d_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    # GPU 6: RealPDEBench Transolver1d (Random 100k 对照)
    "transolver1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_transolver1d_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    # GPU 1: RealPDEBench DPOT1d (Random 100k 对照)
    "dpot1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_dpot1d_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    # GPU 7: ResNetFiLM (Random 100k 对照)
    "resnet_film hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_resnet_film_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    # GPU 2: HyperFNO (Random 100k 对照)
    "hyper_fno hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_hyper_fno_hybrid_w128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    # # GPU 3: PureFNO (Random 100k 对照, BS=256)
    # "pure_fno hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_pure_fno_hybrid_w64 32 64 256 100 none 1.7e-3 10 0 0 0 one_time 3"
    # # GPU 0: PureWNO (Random 100k 对照, BS=256)
    # "pure_wno hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_pure_wno_hybrid_w64 32 64 256 100 none 1.7e-3 10 0 0 0 one_time 0"
)

# ==========================================
# 实验管理模式选择:
#   USE_HIERARCHICAL=1 (推荐, 启用新版分层语义化目录 experiments/dataset/model/...)
#   USE_HIERARCHICAL=0 (兼容旧版扁平目录 train_results_v3/ / eval_results_v3/)
# ==========================================
USE_HIERARCHICAL=1
EXP_ROOT="./experiments"
mkdir -p "$EXP_ROOT" logs_v3 logs

# =========================================================================
# 1. 任务解析与多 GPU 队列分发 (GPU Worker Queue Dispatcher)
#    • 每张 GPU 维护独立的串行任务队列 (同卡按顺序跑完一个再跑下一个，彻底杜绝 OOM 冲突)
#    • 不同 GPU 之间完全并发并行运行
# =========================================================================
declare -A GPU_TASK_LIST
declare -A GPU_TASK_COUNT
ALL_GPUS=()
TOTAL_TASKS=0

for SEED in "${SEEDS[@]}"; do
    for EXP in "${EXPERIMENTS[@]}"; do
        # 忽略空行与纯注释行
        EXP_TRIM=$(echo "$EXP" | sed -e 's/^[[:space:]]*//')
        if [ -z "$EXP_TRIM" ] || [[ "$EXP_TRIM" == \#* ]]; then
            continue
        fi

        read -r MODEL LOSS_TYPE USE_ADJ DATASET DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE GPU_ID <<< "$EXP_TRIM"
        GPU_ID=${GPU_ID:-0}

        if [ -z "${GPU_TASK_COUNT[$GPU_ID]}" ]; then
            GPU_TASK_COUNT[$GPU_ID]=0
            GPU_TASK_LIST[$GPU_ID]=""
            ALL_GPUS+=("$GPU_ID")
        fi

        GPU_TASK_COUNT[$GPU_ID]=$((GPU_TASK_COUNT[$GPU_ID] + 1))
        TOTAL_TASKS=$((TOTAL_TASKS + 1))
        GPU_TASK_LIST[$GPU_ID]+="${SEED}|${EXP_TRIM}"$'\n'
    done
done

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "⚠️ 未找到任何待执行的有效实验配置 (请检查 EXPERIMENTS 数组是否有未注释的实验)"
    exit 0
fi

echo "==========================================================="
echo "📋 GPU 任务排队与分发摘要 (总计 $TOTAL_TASKS 个任务，分布在 ${#ALL_GPUS[@]} 张 GPU):"
for gpu in "${ALL_GPUS[@]}"; do
    echo "   • GPU $gpu: 队列共 ${GPU_TASK_COUNT[$gpu]} 个任务 (同卡串行依序执行，跨卡并发执行)"
done
echo "==========================================================="
echo ""

# ====== 启动 GPU Worker 进程 (每张 GPU 独立启动一个后台 Worker 串行消费队列) ======
PIDS=()

for gpu in "${ALL_GPUS[@]}"; do
    {
        task_idx=0
        total_for_gpu=${GPU_TASK_COUNT[$gpu]}

        while IFS='|' read -r SEED EXP; do
            [ -z "$SEED" ] && continue
            task_idx=$((task_idx + 1))

            read -r MODEL LOSS_TYPE USE_ADJ DATASET DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE GPU_ID <<< "$EXP"
            GPU_ID=${GPU_ID:-$gpu}
            ROLLOUT_TYPE=${ROLLOUT_TYPE:-one_time}

            if [ "$USE_ADJ" = "true" ]; then
                ADJ_PARAM="--adj_matrix_path mydata/yeast_signed_adjacency_matrix.npy"
            else
                ADJ_PARAM=""
            fi

            # 构建 AL 专属参数
            AL_PARAMS=""
            if [ "$AL_STRATEGY" != "none" ]; then
            # PIAL 策略推迟到 epoch 50 触发: 早期模型预测差, ODE 残差无意义
            # mutant_aware 策略推迟到 epoch 100 触发: 需模型基本收敛后再做 atlas 优先级评估, 否则 priority 基于错误的模型状态
            # al4pde/maal 策略: 命令行 TRIGGER_EPOCH 被 al_callback.py 硬编码覆盖 (trigger=4ep, num_add=1000 固定)
            #   实际每 4 epoch 触发一次, 200/4=50 轮 +1 safety = 51 轮 AL, 每轮固定新增 1000 样本 (非指数增长)
            # 其他策略保持 epoch 10 触发
                if [ "$AL_STRATEGY" = "pial" ]; then
                    TRIGGER_EPOCH=50
                elif [ "$AL_STRATEGY" = "mutant_aware" ]; then
                    TRIGGER_EPOCH=100
                elif [[ "$AL_STRATEGY" == al4pde_* || "$AL_STRATEGY" = "maal" ]]; then
                    TRIGGER_EPOCH=30
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
            # --al_pool_dataset: 遗留参数 (MASS 候选池由 sample_mass_mutant 动态生成, 不读取 npz)
            # --al_initial_train_size 2000: 命令行传入但被 al_callback.py 硬编码覆盖为 5000
            #   K-Means 从 MASS 合成训练样本中选 5000 个作为初始训练集
            # 候选池 (MASS 版, al_callback._generate_param_only_pool):
            #   sample_mass_mutant 分层采样 80000 个 (L1/L2/L3/L4 = 25%/35%/25%/15%),
            #   D_real 隔离 (126 real mutants) + 池内 8% 相对阈值去重
            #   仅存参数 P_norm + Y_IC + ovr, 选中后才 mass 版 ODE 仿真生成完整轨迹
                if [[ "$AL_STRATEGY" == al4pde_* || "$AL_STRATEGY" = "maal" ]]; then
                    AL_PARAMS="$AL_PARAMS --al_pool_dataset $DATASET_MASS_V3 --al_initial_train_size 2000"
                fi
            else
                AL_PARAMS="--al_strategy none"
            fi

            # 格式化学习率字符串
            if (( $(echo "$LR >= 0.001" | bc -l 2>/dev/null || echo 0) )); then
                LR_STR=$(printf "%.4g" "$LR")
            else
                LR_STR=$(printf "%.1e" "$LR" | sed 's/-0/-/g')
            fi

        # 直接使用数据集文件名称 (去掉 .npz 后缀) 作为 Level 1 目录名
            DATASET_NAME=$(basename "$DATASET" .npz)

        # 根据模式构建路径与目录
            RUN_NAME="${MODEL}_${AL_STRATEGY}_${LOSS_TYPE}_${DATASET_TAG}_modes${MODES}_seed${SEED}"

            if [ "$USE_HIERARCHICAL" = "1" ]; then
                if [[ "$MODEL" == *"wno"* ]]; then
                    HPARAM_SIG="w${WIDTH}_lev3_haar_b${BATCH_SIZE}_ep${EPOCHS}_lr${LR_STR}"
                else
                    HPARAM_SIG="w${WIDTH}_m${MODES}_b${BATCH_SIZE}_ep${EPOCHS}_lr${LR_STR}"
                fi

                if [ "$USE_ADJ" = "true" ]; then
                    HPARAM_SIG="${HPARAM_SIG}_adj"
                fi
                if [ "$ROLLOUT_TYPE" = "autoregressive" ]; then
                    HPARAM_SIG="${HPARAM_SIG}_ar"
                fi
                if [ -n "$WARMUP" ] && [ "$WARMUP" -ne 10 ] && [ "$WARMUP" -gt 0 ]; then
                    HPARAM_SIG="${HPARAM_SIG}_wm${WARMUP}"
                fi

                EXP_DIR="${EXP_ROOT}/${DATASET_NAME}/${MODEL}/${LOSS_TYPE}_${AL_STRATEGY}/${HPARAM_SIG}/seed_${SEED}"
                CKPT_DIR="${EXP_DIR}/checkpoints"
                EVAL_OUT_DIR="${EXP_DIR}/eval"
                LOG_FILE="${EXP_DIR}/train.log"
                mkdir -p "$CKPT_DIR" "$EVAL_OUT_DIR"
            # 建立软链接到 logs_v3 方便统一 tail
                ln -sf "$(pwd)/${LOG_FILE}" "logs_v3/${RUN_NAME}.log" 2>/dev/null
                DIR_PARAMS="--exp_dir $EXP_DIR --dataset_tag $DATASET_NAME"
            else
                EXP_DIR="./train_results_v3/${RUN_NAME}"
                CKPT_DIR="./train_results_v3/${RUN_NAME}"
                EVAL_OUT_DIR="./eval_results_v3/${RUN_NAME}"
                LOG_FILE="logs_v3/${RUN_NAME}.log"
                mkdir -p "$EXP_DIR" "$EVAL_OUT_DIR"
                DIR_PARAMS="--save_dir ./train_results_v3"
            fi

        # 根据 LOSS_TYPE 构建损失函数专属参数
            PINN_PARAMS=""
            if [ "$LOSS_TYPE" = "pinn_residual" ] || [ "$LOSS_TYPE" = "pino" ]; then
            # v2: 物理约束作弱正则, Curriculum learning (warmup 30 轮纯 MSE)
            # v1 问题: lambda_phys=1.0 → val_loss=0.930 (baseline=0.037, 差 25 倍)
            # v1 问题: lambda_penalty=10.0 → 负值惩罚压制 MSE
            # v1 问题: event 掩码基于预测 → 早期掩码错误导致梯度爆炸
            # v2 修复: 掩码改用真值, 负值惩罚降到 1.0, warmup 30 轮
            # PINO 复用同一组物理约束参数 (继承 PINNResidualLoss)
                PINN_PARAMS="--lambda_phys 0.01 --lambda_jump 0.001 --lambda_penalty 1.0 --residual_subsample 10 --event_weight 0.1"
            fi
        # PINO 专属: H1 频域梯度损失参数 (warmup 30 轮后线性增长到 lambda_h1)
            if [ "$LOSS_TYPE" = "pino" ]; then
                PINN_PARAMS="$PINN_PARAMS --lambda_h1 0.1 --h1_beta 1.0"
            fi

            WANDB_FLAG=""
            if [ "$NO_WANDB" = "1" ]; then
                WANDB_FLAG="--no_wandb"
            fi

            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 [GPU $gpu | 任务 $task_idx/$total_for_gpu] 启动: $RUN_NAME -> 目录: $EXP_DIR"

            # 执行 Train + Eval (同步串行执行)
            {
                echo "===========================================================" > "$LOG_FILE" 2>&1
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始训练 (GPU=$gpu | 任务 $task_idx/$total_for_gpu): $RUN_NAME" >> "$LOG_FILE" 2>&1
                echo "===========================================================" >> "$LOG_FILE" 2>&1

            # ===== 临时开关: SKIP_TRAIN=1 跳过训练只做eval, =0 恢复正常训练 =====
                SKIP_TRAIN=0
            # ===== 临时开关: SKIP_EVAL=1 跳过评估只做训练(OOM/冒烟测试用), =0 恢复正常评估 =====
                SKIP_EVAL=0

                # 步骤 A: 训练
                if [ "$SKIP_TRAIN" = "0" ]; then
                # 训练前清理旧 checkpoint, 防止 步骤B 的 `ls *.ckpt|head -1` 命中上一次残留权重。
                # 尤其 v2.3 al4pde 保存 final-last-epoch.ckpt: 若目录残留旧 best-*.ckpt,
                # 按字母序 best- 会排在 final- 前被误选, 导致评估到过期模型。
                # 仅在 SKIP_TRAIN=0 (确实要重新训练) 时清理; 范围严格限定为本 run 目录下的 *.ckpt。
                    if [ -d "$CKPT_DIR" ]; then
                        rm -f "${CKPT_DIR}"/*.ckpt
                        echo "🧹 训练前已清理旧 checkpoint: ${CKPT_DIR}/*.ckpt" >> "$LOG_FILE" 2>&1
                    fi
                    CUDA_VISIBLE_DEVICES=$gpu $PYTHON_BIN train.py \
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
                        --val_batch_size $BATCH_SIZE \
                        --devices 1 \
                        $AL_PARAMS \
                        $ADJ_PARAM \
                        $PINN_PARAMS \
                        --rollout_type $ROLLOUT_TYPE \
                        $DIR_PARAMS \
                        $WANDB_FLAG >> "$LOG_FILE" 2>&1

                    TRAIN_EXIT=$?
                    if [ $TRAIN_EXIT -ne 0 ]; then
                        echo "❌ [GPU $gpu | 任务 $task_idx/$total_for_gpu] 训练异常退出 (exit=$TRAIN_EXIT), 自动跳过评估, 准备执行下一个任务" >> "$LOG_FILE" 2>&1
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ [GPU $gpu | 任务 $task_idx/$total_for_gpu] 训练失败: $RUN_NAME (exit=$TRAIN_EXIT, 详细报错见 $LOG_FILE)"
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🛡️ [GPU $gpu] 正在进行显存安全重置并自动无缝拉起下一项实验..."
                        # 冷却 2 秒确保 Linux 内核与 CUDA 驱动完全回收显存和 DataLoader 子进程
                        sleep 2
                        continue
                    fi
                fi

                # 步骤 B+C: 寻找 Checkpoint 并评估
                if [ "$SKIP_EVAL" = "0" ]; then
                    CKPT_PATH=$(ls ${CKPT_DIR}/*.ckpt 2>/dev/null | head -n 1)

                    if [ -z "$CKPT_PATH" ]; then
                        echo "❌ 错误: 在 ${CKPT_DIR} 没找到 Checkpoint 文件，跳过 Evaluate。" >> "$LOG_FILE" 2>&1
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ [GPU $gpu | 任务 $task_idx/$total_for_gpu] 找不到 Checkpoint: $RUN_NAME"
                    else
                        echo "🎯 成功找到 Checkpoint: $CKPT_PATH" >> "$LOG_FILE" 2>&1
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始评估..." >> "$LOG_FILE" 2>&1
                        CUDA_VISIBLE_DEVICES=$gpu $PYTHON_BIN eval.py \
                            --ckpt "$CKPT_PATH" \
                            --dataset_name "$DATASET" \
                            --save_dir "$EVAL_OUT_DIR" \
                            --batch_size $BATCH_SIZE >> "$LOG_FILE" 2>&1

                        if [ $? -eq 0 ]; then
                            echo "✅ [$RUN_NAME] 评估完成！结果已保存至: $EVAL_OUT_DIR" >> "$LOG_FILE" 2>&1
                            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ [GPU $gpu | 任务 $task_idx/$total_for_gpu] 评估完成: $RUN_NAME"
                        else
                            echo "❌ [$RUN_NAME] 评估失败" >> "$LOG_FILE" 2>&1
                            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ [GPU $gpu | 任务 $task_idx/$total_for_gpu] 评估失败: $RUN_NAME"
                        fi
                    fi
                fi
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实验结束: $RUN_NAME" >> "$LOG_FILE" 2>&1
            }

            # 任务间预留 10 秒显存与 CUDA 驱动上下文彻底释放缓冲
            sleep 10

        done <<< "${GPU_TASK_LIST[$gpu]}"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🏁 [GPU $gpu] 队列中的全部 $total_for_gpu 个任务已顺利完成！"
    } &
    PIDS+=($!)
done

# 等待所有后台 GPU Worker 完成
echo ""
echo "⏳ 等待所有 GPU Worker 完成 (共 ${#ALL_GPUS[@]} 个并行卡，总计 $TOTAL_TASKS 个任务)..."
echo "   实时查看日志: tail -f logs_v3/*.log"
echo "   分层实验目录: ${EXP_ROOT}/"
echo ""

FAIL_COUNT=0
for pid in "${PIDS[@]}"; do
    wait $pid
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "⚠️ Worker PID $pid 异常退出 (code=$STATUS)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 所有 ${#ALL_GPUS[@]} 张 GPU 的全部 $TOTAL_TASKS 个任务调度已全部执行完毕!"
else
    echo "⚠️ 有 $FAIL_COUNT 个 GPU Worker 报告异常, 请检查对应日志"
fi

# 自动生成多 Seed 聚合报告与排行榜
if [ "$USE_HIERARCHICAL" = "1" ] && [ -d "$EXP_ROOT" ]; then
    echo ""
    echo "📊 正在自动汇总多 Seed 统计指标并生成排行榜..."
    $PYTHON_BIN analyze_experiments.py --exp_root "$EXP_ROOT" --sort_by real_mae
fi
