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
NO_WANDB=0

DATA_DIR="/home/users/zsy/mydata/budding_yeast"

# MASS 数据集 (mass_dataset_generator.py 生成: 126 Real Mutants + MASS 合成样本, 无 LHS 家族结构)
#   样本命名: real mutants 用原名 (如 1_WT_Glc), 合成样本以 'MASS_' 开头 (如 MASS_00001)
#   yeast_datamodule.py 会自动识别 (无 '_LHS_' + 有 'MASS_' 前缀) 并跳过家族分层, 仅按 Pattern 宏观划分
DATASET_MASS="${DATA_DIR}/mass_dataset_210min_500steps_dual_labels.npz"
# MASS v4.5 / v2 数据集 (全新升级: 结构化生物救援基序扩充 + 高阶多突变占比提升至75%)
DATASET_MASS_V2="${DATA_DIR}/mass_v2_dataset_210min_500steps_dual_labels.npz"
# MASS v3 数据集 (全新微初值与显式微拓扑重平衡数据集)
DATASET_MASS_V3="${DATA_DIR}/massv3_dataset_210min_500steps_dual_labels.npz"
# MASS Random 数据集 (无 Motif 纯随机基线数据集)
DATASET_MASS_RANDOM="${DATA_DIR}/mass_random_dataset_210min_500steps_dual_labels.npz"
DATASET_MASS_50000="${DATA_DIR}/mass_50000_dataset_210min_500steps_dual_labels.npz"
# mutant_aware 策略专用: atlas CSV (含 lhs_MAE_min, lhs_n_samples 列)
ATLAS_CSV="./eval_result/old_hyper_fno_none_mse_only_210m_500s_modes32_seed42/mutant_atlas_merged.csv"

SEEDS=(42) #42,1024,3407          

# ==========================================
# 终极多任务实验清单
#
#
# 说明:
#   MODEL         : 模型名 (hyper_fno, cross_fno, pure_fno, patchtst, ...)
#   LOSS_TYPE     : 损失函数 (mse_only, physics_informed, pinn_residual, h1, pino, hybrid_mse_rel, triple_dynamic, phase_derivative, soft_dtw)
#                   h1=H1 Sobolev 损失 (时域L2 + 频域梯度L2, 来自 HyperNOs H1relLoss_1D)
#                   pino=PINO 损失 (MSE + ODE残差 + H1频域梯度, 来自 PINO 论文)
#   USE_ADJ       : 是否用邻接矩阵 (true/false)
#   DATASET_PATH  : 数据集 npz 绝对路径
#   DATASET_TAG   : 数据集标签 (用于命名)
#   MODES         : Fourier modes 数 (Transformer 模型用作 d_model 参考)
#   WIDTH         : 隐藏层宽度
#   BATCH_SIZE    : 批大小
#   EPOCHS        : 最大训练轮数
#   AL_STRATEGY   : 主动学习策略 (none/random/us/is/wrs/vessal/hggs/rgs/pial/mutant_aware/al4pde_novelty/al4pde_random/al4pde_diversity/maal/al4pde_mass)
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

    # ====== 🔥🔥🔥 MASS v3 (全新微初值与拓扑重平衡数据集) 7 卡终极大决战 (已完成归档) 🔥🔥🔥 ======
    # GPU 0: 👑 全项目总冠军 biom_refiner_graph + Hybrid 损失 (最强拓扑图注意力算子冲击更高峰)
    #"biom_refiner_graph hybrid_mse_rel false ${DATASET_MASS_V3} massv3_biom_refiner_graph_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    # GPU 1: 👑 终极大融合旗舰 biom_refiner_plus + 三元动力学复合损失 (全项目最低 MAE 0.0427 冲击巅峰)
    #"biom_refiner_plus triple_dynamic false ${DATASET_MASS_V3} massv3_biom_refiner_plus_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    # GPU 2: 👑 终极大融合旗舰 biom_refiner_plus + Hybrid 损失
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_MASS_V3} massv3_biom_refiner_plus_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    # GPU 3: 👑 超高精破百纪录保持者 biom_refiner_curvature + 三元动力学复合损失 (101个超高精)
    #"biom_refiner_curvature triple_dynamic false ${DATASET_MASS_V3} massv3_biom_refiner_curv_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    # GPU 4: 二阶曲率消融模型 biom_refiner_curvature + Hybrid 损失
    #"biom_refiner_curvature hybrid_mse_rel false ${DATASET_MASS_V3} massv3_biom_refiner_curv_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    # GPU 5: 经典双阶粗精细化算子 BioRefiner + Hybrid 损失 (Core 9 最佳基准)
    #"biom_refiner hybrid_mse_rel false ${DATASET_MASS_V3} massv3_biom_refiner_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    # GPU 6: 三大机制复合全模型 biom_full + Hybrid 损失 (单阶段机制算子最强代表)
    #"biom_full hybrid_mse_rel false ${DATASET_MASS_V3} massv3_biom_full_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 6"
    # GPU 7: 保留空闲 (留给他人使用)

    # ====== 🎲🎲🎲 MASS Random (无 Motif 纯随机基线数据集) 对比实验 🎲🎲🎲 ======
    # GPU 0: 👑 最强模型 biom_refiner_curvature + 三元动力学复合损失 (Random 数据集消融基线)
    #"biom_refiner_curvature triple_dynamic false ${DATASET_MASS_RANDOM} random_biom_refiner_curv_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    
    # MASS 50000
    "biom_refiner_curvature triple_dynamic false ${DATASET_MASS_50000} biom_refiner_curv_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    
    # ====== 🌟🌟🌟 MASS v3 基线模型横向对比评测 (PureFNO, DeepONet, ResNetFiLM, PureWNO, HyperFNO) 🌟🌟🌟 ======
    # GPU 0: PureFNO (ICLR 2021 原版) + Hybrid 损失 (width=64, batch_size=512, epochs=100)
    #"pure_fno hybrid_mse_rel false ${DATASET_MASS_V3} massv3_pure_fno_hybrid_width64 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    # GPU 1: DeepONet (已完成评估: Real MAE=0.1567, Corr=0.9641)
    #"deeponet hybrid_mse_rel false ${DATASET_MASS_V3} massv3_deeponet_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    # GPU 2: ResNetFiLM (已完成评估: Real MAE=0.0495, Corr=0.9961)
    #"resnet_film hybrid_mse_rel false ${DATASET_MASS_V3} massv3_resnet_film_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    # GPU 3: PureWNO (CMAME 2022 原版) + Hybrid 损失 (width=64, batch_size=128/256 防OOM, epochs=100)
    #"pure_wno hybrid_mse_rel false ${DATASET_MASS_V3} massv3_pure_wno_hybrid_width64 32 64 256 100 none 1.7e-3 10 0 0 0 one_time 3"
    # GPU 4: HyperFNO (已完成评估: Real MAE=0.0798, Corr=0.9924)
    #"hyper_fno hybrid_mse_rel false ${DATASET_MASS_V3} massv3_hyper_fno_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    # GPU 5: (备选/按需开启) WNO-Cross (Cross-Attention 增强版小波算子) + Hybrid 损失
    #"wno hybrid_mse_rel false ${DATASET_MASS_V3} massv3_wno_cross_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 5"

    # ====== 🌟 al4pde 框架实验 v2.4 (MASS 版): 外层重训式 AL + DATASET_MASS_V3 初始训练集 🌟 ======
    # 数据集生成算法已从 v5 LHS 家族采样切换为 MASS 分层采样 (mass_dataset_generator.py),
    # al4pde 各策略已全面适配 MASS (al_callback.py):
    #   - 候选池 (v2.6 全策略统一): sample_mass_random 纯随机生成 100000 个
    #     (无 rescue motif 定向采样), Level L1/L2/L3/L4 = 15%/30%/30%/25%
    #     (v2.7 多基因加强: L3+L4 上位性组合载体 40%→55%, 数据集原生 25/35/25/15),
    #     D_real 隔离 (126 real mutants 参数去重) + 池内 1% 相对阈值去重 (v2.8:
    #     8% 会使 L1 容量仅 ~5.9K < 配额 15000, 指针卡死烧尽尝试预算, 实测池仅
    #     39387/100000; 1% 使 L1 容量 ~45K, 平均间距 ~3-4%, L2-L4 不受影响)
    #   - 初始训练集 (v2.6 全策略统一): 从随机候选池完全随机抽取 5000 + ODE 仿真
    #     (不用 npz K-Means 子集, 其含 L3/L4 50% 概率 rescue motif 定向样本);
    #     随机碰撞产生的 motif 形组合属允许范围 (非定向构造)
    #   - 选中后仿真: mass 版 ODE 引擎 (完整 ovr 字典, 支持 init_*/MAD2-CSL/BUB2-CKO 特殊逻辑 + SIGALRM 超时)
    #   - maal: 家族 = _mass_sample_genes 提取全部突变基因 (v2.7 多基因聚合,
    #     每个样本的信号归属其全部基因家族, 纯碳源归 CARBON 伪家族)
    #   - al4pde_mass: 直接查 BINDING_GROUPS 真实 (gene, mode) 元数据打分 (通路频率 × 机制频率)
    #   4 轮采集各追加 19000 (来自随机候选池) → 总 81000, 与 v1 al4pde 实验规模对齐
    # 配置 (v2.6 换强模型): biom_refiner_curvature + triple_dynamic + width=128 + batch=512
    #   + 每轮 100 epoch + lr=1.7e-3 (对齐当前最强模型 L221 超高精破百纪录配置, 用户指定)
    # 注: --al_pool_dataset/--al_initial_train_size 为遗留参数 (候选池动态生成, 不读取 npz)
    # 注: al4pde_boundary 已删除 (用户判定无意义: 仅是"参数越接近0越优先选", 无判别价值)
    #"biom_refiner_curvature triple_dynamic false ${DATASET_MASS_V3} massv3_al4pde 32 128 512 100 al4pde_random 1.7e-3 10 1000 0 0 one_time 0"
    #"biom_refiner_curvature triple_dynamic false ${DATASET_MASS_V3} massv3_al4pde 32 128 512 100 al4pde_diversity 1.7e-3 10 1000 0 0 one_time 1"
    #"biom_refiner_curvature triple_dynamic false ${DATASET_MASS_V3} massv3_al4pde 32 128 512 100 maal 1.7e-3 10 1000 0 0 one_time 2"
    #"biom_refiner_curvature triple_dynamic false ${DATASET_MASS_V3} massv3_al4pde 32 128 512 100 al4pde_mass 1.7e-3 10 1000 0 0 one_time 5"

    # ====== 🌟 al4pde_novelty (v2.5): 组合新颖性主动选择 🌟 ======
    # v2.6 起 5 个 al4pde 实验的候选池与初始集完全一致 (均为 sample_mass_random
    # 纯随机池 + 随机初始 5000), 唯一差异是每轮的 acquisition 选样策略 → 策略间公平对比
    # 设计目标: 选样不得泄露 126 个 real mutants; 验证集/测试集与其他实验完全一致 (massv3 npz 同一划分)。
    # 机制 (al_callback.py):
    #   - 选样: acquisition = 0.6*novelty(突变单元全子集组合覆盖稀缺度, 主导信号;
    #     单元=(基因,突变类型), 评估单元/对/三元/.../完整组合, 尺寸加权 w_s=2^-|s-2|)
    #     + 0.2*family_unc(训练集家族不确定性) + 0.2*diversity(ODE 轨迹形状 min-dist),
    #     三信号仅来自训练集+候选池元数据, 对 real mutants 零泄露
    #   - 权重可调: --al_novelty_weight/--al_family_weight (diversity = 1 - 两者, 自动归一化)
    # 模型配置 (v2.6): 与 4 个对照实验统一, 均为当前最强模型 biom_refiner_curvature
    #   + triple_dynamic + width=128 + batch=512 + lr=1.7e-3 (对齐 L221 配置)
    #"biom_refiner_curvature triple_dynamic false ${DATASET_MASS_V3} massv3_al4pde 32 128 512 100 al4pde_novelty 1.7e-3 10 1000 0 0 one_time 4"

    # ====== 🌟🌟🌟 MASS v2 经典历史参考配置 (代表性模型归档) 🌟🌟🌟 ======
    # 1. biom_refiner_plus + Triple-Dynamic (MASS v2 巅峰模型)
    #"biom_refiner_plus triple_dynamic false ${DATASET_MASS_V2} massv2_biom_refiner_plus_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    # 2. biom_refiner_curvature + Triple-Dynamic (MASS v2 二阶曲率模型)
    #"biom_refiner_curvature triple_dynamic false ${DATASET_MASS_V2} massv2_biom_refiner_curv_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 6"
    # 3. biom_refiner + Hybrid 损失 (MASS v2 粗精细化双阶基线)
    #"biom_refiner hybrid_mse_rel false ${DATASET_MASS_V2} massv2_biom_refiner_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    # 4. resnet_film + Hybrid 损失 (MASS v2 深度残差基线)
    #"resnet_film hybrid_mse_rel false ${DATASET_MASS_V2} massv2_resnet_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    # 5. dyrfo + Triple-Dynamic (MASS v2 动态重构算子)
    #"dyrfo triple_dynamic false ${DATASET_MASS_V2} massv2_dyrfo_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
)

# ==========================================
# 实验管理模式选择:
#   USE_HIERARCHICAL=1 (推荐, 启用新版分层语义化目录 experiments/dataset/model/...)
#   USE_HIERARCHICAL=0 (兼容旧版扁平目录 train_results_v3/ / eval_results_v3/)
# ==========================================
USE_HIERARCHICAL=1
EXP_ROOT="./experiments"
mkdir -p "$EXP_ROOT" logs_v3 logs

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

            # 非默认标记智能追加
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

        echo "🚀 启动实验 -> Seed: $SEED | 模型: $MODEL | 策略: $AL_STRATEGY | GPU: $GPU_ID | 目录: $EXP_DIR"

        # 后台并行执行 train + eval (子 shell 继承当前变量快照)
        {
            echo "===========================================================" > "$LOG_FILE" 2>&1
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始训练: $RUN_NAME (GPU=$GPU_ID)" >> "$LOG_FILE" 2>&1
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
                CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN train.py \
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
                    echo "❌ 训练失败 (exit=$TRAIN_EXIT), 跳过评估" >> "$LOG_FILE" 2>&1
                    exit 1
                fi
            else
                echo "⏭️ SKIP_TRAIN=1, 跳过训练, 直接评估已有 checkpoint" >> "$LOG_FILE" 2>&1
            fi

            # 步骤 B+C: 寻找 Checkpoint 并评估 (SKIP_EVAL=1 时整体跳过, 用于 batch/OOM 冒烟测试)
            if [ "$SKIP_EVAL" = "0" ]; then
                # 步骤 B: 寻找 Checkpoint (优先选 final 或 best)
                CKPT_PATH=$(ls ${CKPT_DIR}/*.ckpt 2>/dev/null | head -n 1)

                if [ -z "$CKPT_PATH" ]; then
                    echo "❌ 错误: 在 ${CKPT_DIR} 没找到 Checkpoint 文件，跳过 Evaluate。" >> "$LOG_FILE" 2>&1
                    exit 1
                fi

                echo "🎯 成功找到 Checkpoint: $CKPT_PATH" >> "$LOG_FILE" 2>&1

                # 步骤 C: 评估（与模型 batch_size 对齐以确保显存安全）
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始评估..." >> "$LOG_FILE" 2>&1
                CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN eval.py \
                    --ckpt "$CKPT_PATH" \
                    --dataset_name "$DATASET" \
                    --save_dir "$EVAL_OUT_DIR" \
                    --batch_size $BATCH_SIZE >> "$LOG_FILE" 2>&1

                if [ $? -eq 0 ]; then
                    echo "✅ [$RUN_NAME] 评估完成！结果已保存至: $EVAL_OUT_DIR" >> "$LOG_FILE" 2>&1
                else
                    echo "❌ [$RUN_NAME] 评估失败" >> "$LOG_FILE" 2>&1
                fi
            else
                echo "⏭️ SKIP_EVAL=1, 跳过评估 (仅训练/OOM 冒烟测试)" >> "$LOG_FILE" 2>&1
            fi
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实验结束: $RUN_NAME" >> "$LOG_FILE" 2>&1
        } &
        PIDS+=($!)

    done
done

# 等待所有后台实验完成
echo ""
echo "⏳ 等待 ${#PIDS[@]} 个并行实验完成..."
echo "   实时查看日志: tail -f logs_v3/*.log"
echo "   分层实验目录: ${EXP_ROOT}/"
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
    echo "⚠️ ${#PIDS[@]} 个实验中有 $FAIL_COUNT 个失败, 请检查日志"
fi

# 自动生成多 Seed 聚合报告与排行榜
if [ "$USE_HIERARCHICAL" = "1" ] && [ -d "$EXP_ROOT" ]; then
    echo ""
    echo "📊 正在自动汇总多 Seed 统计指标并生成排行榜..."
    $PYTHON_BIN analyze_experiments.py --exp_root "$EXP_ROOT" --sort_by real_mae
fi
