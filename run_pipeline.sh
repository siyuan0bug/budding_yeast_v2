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
NORM_TYPE="log1p"

DATA_DIR="/home/users/zsy/mydata/budding_yeast"

# MASS 数据集 (mass_dataset_generator.py 生成: 126 Real Mutants + MASS 合成样本, 无 LHS 家族结构)
#   样本命名: real mutants 用原名 (如 1_WT_Glc), 合成样本以 'MASS_' 开头 (如 MASS_00001)
#   yeast_datamodule.py 会自动识别 (无 '_LHS_' + 有 'MASS_' 前缀) 并跳过家族分层, 仅按 Pattern 宏观划分
DATASET_MASS="${DATA_DIR}/mass_dataset_210min_500steps_dual_labels.npz"
# MASS v4.5 / v2 数据集 (全新升级: 结构化生物救援基序扩充 + 高阶多突变占比提升至75%)
DATASET_MASS_V2="${DATA_DIR}/mass_v2_dataset_210min_500steps_dual_labels.npz"
# MASS Mutant-Aware 100,000 数据集 (mutant_aware.py v5.0 生成: 零泄露分层采样 + 126 Real Mutants, 10万样本全量高精数据集)
DATASET_MUTANT_AWARE_100000="${DATA_DIR}/mutant_aware_100000_210m_500s.npz"
# MASS Mutant-Aware 100,000 Role-mode 数据集 (mutant_aware.py: Tier 2/3 affinity 角色模式采样 + 126 Real Mutants)
DATASET_MUTANT_AWARE_100000_ROLE="${DATA_DIR}/mutant_aware_100000_210m_500s_rolemode.npz"
# MASS Mutant-Aware 100,000 Leak-Level 数据集 (mutant_aware.py: 40/30/20/10 level 配额采样 + 126 Real Mutants)
DATASET_MUTANT_AWARE_100000_LEAKLEVEL="${DATA_DIR}/mutant_aware_100000_leaklevel_210m_500s.npz"
# MASS k-DPP Gated 100,000 V2 数据集 (generate_kdpp_gated_dataset.py: 图拉普拉斯扩散 k-DPP + 吸引子门控均衡 V2版, 75.1分最优配比 20/40/25/15)
DATASET_KDPP_GATED_V2_100000="${DATA_DIR}/kdpp_gated_100000_210m_500s.npz"
# MASS k-DPP Curv Gated 100,000 数据集 (generate_kdpp_curv_gated_dataset.py: 符号有向图拉普拉斯 k-DPP + 相空间曲率门控融合版)
DATASET_KDPP_CURV_GATED_100000="${DATA_DIR}/kdpp_curv_gated_100000_210m_500s.npz"
# MASS BAMS 100,000 数据集 (bams_dataset_generator.py: 基尔霍夫符号流守恒 + 闭环数值分岔二分逼近, 93.6分顶级无泄露数据集)
DATASET_BAMS_100000="${DATA_DIR}/bams_100000_210m_500s.npz"
# MASS 100,000 数据集 (含生物救援基序 Motif, 10万样本全量高精数据集)
DATASET_MASS_100000="${DATA_DIR}/mass_100000_dataset_210min_500steps_dual_labels.npz"
# MASS 100,000 Random 数据集 (无 Motif 纯随机基线, 10万样本对照数据集)
DATASET_MASS_100000_RANDOM="${DATA_DIR}/mass_100000_random_dataset_210min_500steps_dual_labels.npz"
# MASS Random + Mutant-Aware Parameter Distribution 100,000 数据集 (mass_random_mutant_dist.py 生成: mass_random_sampler 突变选取逻辑 + mutant_aware 连续物理参数分布采样消融实验)
DATASET_MASS_RANDOM_DIST_100000="${DATA_DIR}/mass_random_dist_100000_210m_500s.npz"

DATASET_MASS_100000_RANDOM_LEVEL="${DATA_DIR}/mass_100000_random_l_dataset_210min_500steps_dual_labels.npz"
# MASS 50,000 数据集 (5万样本规模对比数据集)
DATASET_MASS_50000="${DATA_DIR}/mass_50000_dataset_210min_500steps_dual_labels.npz"

# ====== BAMS 50,000 三分支靶向消融实验数据集 (50k样本, 50 Epochs, biom_refiner_plus 对比) ======
# 0. 对照组 (50k 纯 BAMS 0 靶向基线):
DATASET_BAMS_50000="${DATA_DIR}/bams_50000_210m_500s.npz"
# 1. 消融组 A (40k BAMS 底座 + 10k 靶向骨架 Hard Zeros 0.0):
DATASET_BAMS_ABLATION_HARD_50000="${DATA_DIR}/bams_ablation_hard_50000_210m_500s.npz"
# 2. 消融组 B (40k BAMS 底座 + 10k 靶向骨架 Continuous Hypomorphic ts 衰减):
DATASET_BAMS_ABLATION_HYPO_50000="${DATA_DIR}/bams_ablation_hypo_50000_210m_500s.npz"
# 3. 方案 A 增强优化版: 100,000 样本 (网络拓扑基序 + 并联检查点配对 + 主控开关 5 阶深探 + 硬零点 KO):
DATASET_BAMS_MOTIF_TOPOLOGY_100000="${DATA_DIR}/bams_motif_topology_100000_210m_500s.npz"

# 4. mutant_aware 纯随机对照基准: 100,000 样本 (零拓扑结构, 27基因均匀随机, 模式均匀随机, 25/25/25/25 平权)
DATASET_MUTANT_AWARE_PURE_RANDOM_100000="${DATA_DIR}/mutant_aware_pure_random_100000_210m_500s.npz"
# 5. mutant_aware 纯随机 + 碰撞拯救 (Collision Rescue) 增强版: 100,000 样本 (消除查重黑洞, 70% DAmP/ts/pLOF 连续弱化 + 30% 碳源漂移)
DATASET_MUTANT_AWARE_PURE_RANDOM_RESCUE_100000="${DATA_DIR}/mutant_aware_pure_random_rescue_100000_210m_500s.npz"
# 6. mutant_aware 纯随机 + KO 邻域微扰区间 [0, 1e-4] 采样版: 100,000 样本 (无 Rescue, 仅 KO 微扰采样, 消除 Dirac 硬零点)
DATASET_MUTANT_AWARE_PURE_RANDOM_KO_INTERVAL_100000="${DATA_DIR}/mutant_aware_pure_random_ko_interval_100000_210m_500s.npz"
# 7. Gibbs-Boltzmann 图连续热力学能量采样版 (无泄露直舍拦截, 100,000 样本 + 126 Real Mutants):
DATASET_GIBBS_BOLTZMANN_100000="${DATA_DIR}/gibbs_boltzmann_100000_210m_500s.npz"
# 8. Gibbs-Sobol 低差异准随机加扰采样版 (Scrambled Sobol + Gibbs-Boltzmann, 100,000 样本 + 126 Real Mutants):
DATASET_GIBBS_SOBOL_100000="${DATA_DIR}/gibbs_sobol_100000_210m_500s.npz"

# 9. 参数分布消融孪生实验数据集 (严格控制单变量对比: 仅 ts, db, APC-A, TAB6-1, CA, CSL, CKO, scc1-RR, PM 参数采样分布不同):
#   - Twin B: 完整体系 B 连续物理包络与边界缓冲采样 (mutant_aware.py v5.2)
#   - Twin A: 体系 B 骨架 + 体系 A 参数采样分布与赋值规则 (mass_dataset_generator.py)
DATASET_DIST_TWIN_B="${DATA_DIR}/distribution_twin_b_continuous_100000_210m_500s.npz"
DATASET_DIST_TWIN_A="${DATA_DIR}/distribution_twin_a_direct_100000_210m_500s.npz"

# 10. 纯随机无拓扑参数分布消融孪生实验数据集 (零拓扑、零 Motif、27基因完全均匀随机、突变类型完全均匀随机):
#   - Twin B: 纯随机骨架 + 完整体系 B 连续物理包络与边界缓冲采样
#   - Twin A: 纯随机骨架 + 体系 A 参数采样分布与赋值规则
DATASET_PURE_RANDOM_DIST_TWIN_B="${DATA_DIR}/pure_random_distribution_twin_b_continuous_100000_210m_500s.npz"
DATASET_PURE_RANDOM_DIST_TWIN_A="${DATA_DIR}/pure_random_distribution_twin_a_direct_100000_210m_500s.npz"

DATASET_MASS_100000_BLIND="${DATA_DIR}/wb_mass_100000_dataset_210min_500steps_dual_labels.npz"

DATASET_MASS_100000_BLIND_S5="${DATA_DIR}/wb_mass_100000_dataset_210min_500steps_dual_labels_s5mode.npz"
# 旧版 MASS v3 数据集 (已注释归档)
#DATASET_MASS_V3="${DATA_DIR}/massv3_dataset_210min_500steps_dual_labels.npz"
# mutant_aware 策略专用: atlas CSV (含 lhs_MAE_min, lhs_n_samples 列)
ATLAS_CSV="./eval_result/old_hyper_fno_none_mse_only_210m_500s_modes32_seed42/mutant_atlas_merged.csv"

SEEDS=(42) #42,1024,3407          

# ==========================================
# 终极多任务实验清单
#
# 说明:
#   MODEL         : 模型名 (biom_refiner_plus, biom_refiner_bifurcate, biom_refiner_graph_flux, biom_refiner_mechanistic, ...)
#   LOSS_TYPE     : 损失函数 (hybrid_mse_rel, hybrid_bifurcate, hybrid_pino_colloc, hybrid_all, triple_dynamic, ...)
#   USE_ADJ       : 是否用邻接矩阵 (true/false)
#   DATASET_PATH  : 数据集 npz 绝对路径
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
#   GPU_ID        : 使用的 GPU 编号 (严格仅限 0-3, 绝不使用 4-7; 同卡串行排队, 跨卡并行)
# ==========================================

EXPERIMENTS=(
    # 格式 (16列，空格分隔，彻底精简冗余的 DATASET_TAG):
#   MODEL LOSS_TYPE USE_ADJ DATASET_PATH MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE GPU_ID

    # ====== 🌟🌟🌟 新归一化方案 (asinh_z: 参数统一 asinh(Z) + 变量稳健 log1p) 三大基准对比实验 (已归档) 🌟🌟🌟 ======
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"biom_refiner_plus triple_dynamic false ${DATASET_MASS_100000} 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_GIBBS_BOLTZMANN_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"

    # ==============================================================================
    # 🌟🌟🌟 全量 39 种经典 Baseline 神经算子与深度序列模型全家桶 (10万纯随机数据集对照) 🌟🌟🌟
    # 数据集: mutant_aware_pure_random_100000_210m_500s.npz
    # 实验配置参考: biom_refiner_plus 经典基准 (hybrid_mse_rel, m24, w192, b256, ep100, lr8.8e-4, wm8)
    # 硬件分配 (5 卡并行，每卡智能单进程串行排队):
    #   - GPU 0 (8个): pure_fno, fno1d, pure_wno, cno1d, mwt1d, galerkin_transformer1d, transolver1d, transolver3_1d
    #   - GPU 1 (8个): dpot1d, gcgnet, deeponet, neural_ode, unet1d, resnet_film, gnot, latent_lno
    #   - GPU 2 (8个): lamo, koopman_dmd, wdno, icon, bampno, gfno, noble, itransformer
    #   - GPU 3 (8个): dlinear, nlinear, linear, autoformer, informer, iinformer, flowformer, iflowformer
    #   - GPU 7 (7个): reformer, ireformer, flashformer, iflashformer, vanilla_transformer, fedformer, pyraformer (独占卡串行)
    # ==============================================================================

    # ==============================================================================
    # 🌟🌟🌟 参数分布消融孪生对照实验 (Strict Distribution Ablation Twins) 🌟🌟🌟
    # 严格控制单变量: 突变骨架、模式、离散 MC 拷贝、GAL-OE、硬零 KO 100% 相同
    # 唯一变量: 仅对 ts, db, APC-A, TAB6-1, CA, CSL, CKO, scc1-RR, PM 参数采样分布进行消融对比
    # 配置完全对齐基准: biom_refiner_plus, hybrid_mse_rel, false, m24, w192, b256, ep100, lr8.8e-4, wm8
    # ==============================================================================
    # [Twin B | GPU 0]: 完整体系 B 连续物理包络与边界缓冲采样 (mutant_aware.py v5.2 - 已归档注释)
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_DIST_TWIN_B} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"

    # [Twin A | GPU 1]: 体系 B 骨架 + 体系 A 参数采样分布与赋值规则 (mass_dataset_generator.py - 已归档注释)
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_DIST_TWIN_A} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"

    # ====== 🌟🌟🌟 纯随机无拓扑参数分布消融孪生对照实验 (100% 零拓扑、零 Motif、完全均匀随机) 🌟🌟🌟 ======
    # [Pure Random Twin B | GPU 0]: 纯随机骨架 + 完整体系 B 连续物理包络与边界缓冲采样
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_PURE_RANDOM_DIST_TWIN_B} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"

    # [Pure Random Twin A | GPU 1]: 纯随机骨架 + 体系 A 参数采样分布与赋值规则
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_PURE_RANDOM_DIST_TWIN_A} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"

    # ==============================================================================
    # 🌟🌟🌟 4 个未完成修复模型 + 8 个全新引入 Baseline 神经算子 (10万纯随机数据集对照) 🌟🌟🌟
    # 修复未完成: wdno, autoformer, gfno, noble (修复 BF16 rfft/wavedec 精度下推问题，排除 pure_fno)
    # 全新引入: sc_fno (ICLR 2025), pno1d, mpno1d, konode, bregman_fno (ICML 2025), bregman_wno, ncflow (ICLR 2025), iganet_transformer
    # 数据集: mutant_aware_pure_random_100000_210m_500s.npz
    # 硬件分配 (5 卡并行，每卡智能单进程串行排队):
    #   - GPU 0: autoformer (修复重跑), sc_fno (全新)
    #   - GPU 1: noble (修复重跑), pno1d (全新)
    #   - GPU 2: wdno (修复重跑), mpno1d (全新), konode (全新)
    #   - GPU 3: gfno (修复重跑), bregman_fno (全新), bregman_wno (全新)
    #   - GPU 7: ncflow (全新), iganet_transformer (全新)
    # ==============================================================================
    # ==============================================================================
    # 🌟🌟🌟 新一代双吸引子解耦与因果可达性神经算子 (Mixture-of-Attractors Suite) 🌟🌟🌟
    # 彻底解决非线性分岔拓扑奇点与残差穿透缺陷 (攻克 Mutant 15, Mutant 26, 合成 1472/7317 badcases)
    # 损失函数: hybrid_bifurcate (联合时序拟合 + Pattern A/B/C 分岔分类 + 阻滞平坦稳态惩罚)
    # 数据集: mutant_aware_pure_random_100000_210m_500s.npz
    # 配置: hybrid_bifurcate, m24, w192, b256, ep100, lr8.8e-4, wm8
    # 硬件分配:
    #   - GPU 0: biom_refiner_attractor (双吸引子乘法熔断算子)
    #   - GPU 1: biom_refiner_attractor_plus (双吸引子 + 因果调控图可达性旗舰算子)
    #   - GPU 2: biom_refiner_causal_reach (生物调控图因果可达性算子)
    # ==============================================================================
    "biom_refiner_attractor hybrid_bifurcate false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    "biom_refiner_attractor_plus hybrid_bifurcate false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    "biom_refiner_causal_reach hybrid_bifurcate false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"

    # [已归档注释]
    #"biom_refiner_bifurcate hybrid_bifurcate false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"biom_refiner_mechanistic hybrid_bifurcate false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"

    # [已归档注释]
    #"biom_mass_action_refiner hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"

    # ==============================================================================
    # 🌟🌟🌟 4 个未完成修复模型 + 8 个全新引入 Baseline 神经算子 (已归档注释) 🌟🌟🌟
    # ==============================================================================
    # [GPU 0]
    #"autoformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"sc_fno hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"

    # [GPU 1]
    #"noble hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    #"pno1d hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"

    # [GPU 2]
    #"wdno hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    #"mpno1d hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    #"konode hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"

    # [GPU 3]
    #"gfno hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    #"bregman_fno hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    #"bregman_wno hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"

    # [GPU 7]
    #"ncflow hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 7"
    #"iganet_transformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 7"

    # [GPU 0: 连续神经算子组 (8 模型) - 已归档注释]
    #"pure_fno hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"fno1d hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"pure_wno hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 64 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"cno1d hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"mwt1d hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"galerkin_transformer1d hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"transolver1d hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"transolver3_1d hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"

    # [GPU 1: 物理感知与深度算子组 (8 模型) - 已归档注释]
    #"dpot1d hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    #"gcgnet hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    #"deeponet hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    #"neural_ode hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    #"unet1d hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 64 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    #"resnet_film hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    #"gnot hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    #"latent_lno hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"

    # [GPU 2: 几何图结构与隐空间算子组 (8 模型) - 已归档注释]
    #"lamo hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    #"koopman_dmd hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    #"wdno hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    #"icon hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    #"bampno hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    #"gfno hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    #"noble hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    #"itransformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"

    # [GPU 3: 线性反转与经典时间序列组 (8 模型) - 已归档注释]
    #"dlinear hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    #"nlinear hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    #"linear hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    #"autoformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    #"informer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    #"iinformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    #"flowformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    #"iflowformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"

    # [GPU 7: 稀疏分解与高效注意力时序组 (7 模型, 独占卡严格单任务串行) - 已归档注释]
    #"reformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 7"
    #"ireformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 7"
    #"flashformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 7"
    #"iflashformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 7"
    #"vanilla_transformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 7"
    #"fedformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 7"
    #"pyraformer hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 7"

    # ====== ⚡⚡⚡ 39 通道自适应梯度平衡 GradNormHybrid 对照实验 (GPU 0-1) ⚡⚡⚡ ======
    # [实验 GN-1 | GPU 0]: 新 Gibbs-Sobol 数据集 × 全事件混合模型 + gradnorm_hybrid (39 通道微观自适应平衡)
    #   对标基线: experiments/gibbs_sobol_100000_210m_500s/biom_refiner_event_hybrid/hybrid_mse_rel_none/
    #"biom_refiner_event_hybrid gradnorm_hybrid false ${DATASET_GIBBS_SOBOL_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"

    # [实验 GN-2 | GPU 1]: 旧 Gibbs-Boltzmann 数据集 × 经典旗舰模型 + gradnorm_hybrid (跨代基准对比)
    #   对标基线: experiments/gibbs_boltzmann_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/
    #"biom_refiner_plus gradnorm_hybrid false ${DATASET_GIBBS_BOLTZMANN_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"

    # ====== 🏆🏆🏆 终极全事件驱动混成神经算子 BioEventHybridRefiner (已评测完成归档) 🏆🏆🏆 ======
    # [实验 1 | GPU 0]: 新 Gibbs-Sobol 数据集 × 全事件混合模型 + triple_dynamic
    #"biom_refiner_event_hybrid triple_dynamic false ${DATASET_GIBBS_SOBOL_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    # [实验 2 | GPU 1]: 旧 Gibbs-Boltzmann 数据集 × 全事件混合模型 + triple_dynamic
    #"biom_refiner_event_hybrid triple_dynamic false ${DATASET_GIBBS_BOLTZMANN_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # [实验 3 | GPU 2]: 新 Gibbs-Sobol 数据集 × 全事件混合模型 + hybrid_mse_rel
    #"biom_refiner_event_hybrid hybrid_mse_rel false ${DATASET_GIBBS_SOBOL_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    # [实验 4 | GPU 3]: 纯随机数据集 × 全事件混合模型 + triple_dynamic
    #"biom_refiner_event_hybrid triple_dynamic false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"

    # ====== 📦📦📦 前代实验归档 (已全部评测完成，保留配置注释避免重复覆盖) 📦📦📦 ======
    # [前代 DynamicPlus 1 | GPU 0]: 旧 Gibbs 数据集 × BioRefinerDynamicPlus
    #"biom_refiner_dynamic_plus hybrid_mse_rel false ${DATASET_GIBBS_BOLTZMANN_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    # [前代 DynamicPlus 2 | GPU 1]: 新 Gibbs-Sobol 数据集 × BioRefinerDynamicPlus
    #"biom_refiner_dynamic_plus hybrid_mse_rel false ${DATASET_GIBBS_SOBOL_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # [前代 DynamicPlus 3 | GPU 2]: 旧 Gibbs 数据集 × BioRefinerDynamicPlus + triple_dynamic
    #"biom_refiner_dynamic_plus triple_dynamic false ${DATASET_GIBBS_BOLTZMANN_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    # [前代 DynamicPlus 4 | GPU 3]: 新 Gibbs-Sobol 数据集 × BioRefinerDynamicPlus + triple_dynamic
    #"biom_refiner_dynamic_plus triple_dynamic false ${DATASET_GIBBS_SOBOL_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"

    # ====== 📦📦📦 前代实验归档 (已全部评测完成，保留配置注释避免重复覆盖) 📦📦📦 ======
    # [前代 1 | GPU 0]: 旧 Gibbs 数据集 × 旧模型 (biom_refiner_plus)
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_GIBBS_BOLTZMANN_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    # [前代 2 | GPU 1]: 旧 Gibbs 数据集 × 因果自锁模型 (biom_refiner_latch_causal)
    #"biom_refiner_latch_causal hybrid_mse_rel false ${DATASET_GIBBS_BOLTZMANN_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # [前代 3 | GPU 2]: 新 Gibbs-Sobol 数据集 × 旧模型 (biom_refiner_plus)
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_GIBBS_SOBOL_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    # [前代 4 | GPU 3]: 新 Gibbs-Sobol 数据集 × 因果自锁模型 (biom_refiner_latch_causal)
    #"biom_refiner_latch_causal hybrid_mse_rel false ${DATASET_GIBBS_SOBOL_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"

    # ====== 🌌🌌🌌 Gibbs-Boltzmann 图连续热力学能量采样版 (备选归档) 🌌🌌🌌 ======
    # GPU 0: biom_refiner_plus + hybrid_mse_rel
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_GIBBS_BOLTZMANN_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    # GPU 2: biom_refiner_plus + triple_dynamic
    #"biom_refiner_plus triple_dynamic false ${DATASET_GIBBS_BOLTZMANN_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"

    # ====== 🔬🔬🔬 纯随机 + KO 邻域微扰区间 [0, 1e-4] 采样版: 100,000 样本 (已完成评测归档) 🔬🔬🔬 ======
    # GPU 0: biom_refiner_plus + hybrid_mse_rel
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_KO_INTERVAL_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    # GPU 2: biom_refiner_plus + triple_dynamic
    #"biom_refiner_plus triple_dynamic false ${DATASET_MUTANT_AWARE_PURE_RANDOM_KO_INTERVAL_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"

    # ====== 🚑🚑🚑 纯随机 + 碰撞拯救 (Collision Rescue) 增强版: 100,000 样本 (已完成评测归档) 🚑🚑🚑 ======
    # GPU 0: biom_refiner_plus + hybrid_mse_rel
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_RESCUE_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    # GPU 2: biom_refiner_plus + triple_dynamic
    #"biom_refiner_plus triple_dynamic false ${DATASET_MUTANT_AWARE_PURE_RANDOM_RESCUE_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"

    # ====== 🎲🎲🎲 纯随机对照基准: 100,000 样本 (零拓扑结构, 已完成评测归档) 🎲🎲🎲 ======
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"biom_refiner_plus triple_dynamic false ${DATASET_MUTANT_AWARE_PURE_RANDOM_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"

    # ====== 🚀🚀🚀 方案 A 增强优化版: 100,000 样本网络拓扑流形数据集 (已完成评测归档) 🚀🚀🚀 ======
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_BAMS_MOTIF_TOPOLOGY_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"biom_refiner_plus triple_dynamic false ${DATASET_BAMS_MOTIF_TOPOLOGY_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    # ====== 🧪🧪🧪 BAMS 50,000 三分支靶向消融实验 (已完成评测归档) 🧪🧪🧪 ======
    # 实验 0 (对照组): 50k 纯 BAMS 0 靶向基线
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_BAMS_50000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # 实验 1 (消融组 A): 50k 靶向骨架 + 硬零点 Hard Zeros 0.0
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_BAMS_ABLATION_HARD_50000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    # 实验 2 (消融组 B): 50k 靶向骨架 + 连续弱化流形 Continuous Hypomorphic ts 衰减
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_BAMS_ABLATION_HYPO_50000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"

    # ====== 🧬🧬🧬 时变动态因果与相变解耦消融实验矩阵 (已完成归档) 🧬🧬🧬 ======
    # EXP-C1: biom_refiner_causal_plus + triple_dynamic
    #"biom_refiner_causal_plus triple_dynamic false ${DATASET_MUTANT_AWARE_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    # EXP-C2: biom_refiner_dynamic_causal_plus + triple_dynamic
    #"biom_refiner_dynamic_causal_plus triple_dynamic false ${DATASET_MUTANT_AWARE_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # EXP-C3: biom_refiner_dynamic_causal_plus + hybrid_mse_rel
    #"biom_refiner_dynamic_causal_plus hybrid_mse_rel false ${DATASET_MUTANT_AWARE_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"

    # (已完成的历史消融实验，已归档供对比)
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_MUTANT_AWARE_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    #"biom_refiner_bifurcate hybrid_bifurcate false ${DATASET_MUTANT_AWARE_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    #"biom_refiner_graph_flux hybrid_mse_rel false ${DATASET_MUTANT_AWARE_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 4"
    #"biom_refiner_plus hybrid_pino_colloc false ${DATASET_MUTANT_AWARE_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    #"biom_refiner_mechanistic hybrid_bifurcate false ${DATASET_MUTANT_AWARE_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    #"biom_refiner_mechanistic hybrid_all false ${DATASET_MUTANT_AWARE_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"

    # ====== 🏆🏆🏆 当前活跃任务: BAMS 100,000 (16列格式) 🏆🏆🏆 ======
    # GPU 0: biom_refiner_plus 在 BAMS 100k 数据集上的实验
    # "biom_refiner_plus hybrid_mse_rel false ${DATASET_BAMS_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"

    # # GPU 1: biom_refiner_causal_plus 在 BAMS 100k 数据集上的实验
    # "biom_refiner_causal_plus hybrid_mse_rel false ${DATASET_BAMS_100000} 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"

    # "unet1d hybrid_mse_rel false ${DATASET_MASS_100000_BLIND} random100k_blind_unet1d_hybrid_w64 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 2"

    # ====== 🆕 HANO_Dual: 双路径门控融合 (性能主推, 修复 HANO v1 性能瓶颈) ======
    # HANO v1 纯物理路径是开环系统 (通量误差随积分累积) + 98 个数量级各异的速率难回归,
    # mass100k 上 relL2=0.266 vs unet1d 0.168; 且 bf16 下 epoch 75+ NaN (已修: 物理路径强制 float32)。
    # HANO_Dual: out = g·物理路径(通量积分+事件跳变) + (1-g)·直接路径(归一化残差回归),
    # 下限 unet 水平, 上限物理增强; 物理路径有独立辅助损失 (lambda_phys=0.3)。
    # GPU 0: 主战场 mass100k 对标 unet1d (hybrid_mse_rel 与 unet 同损失, 公平对比)
    #"hano_dual hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} mass100k_hano_dual_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    # GPU 1: 损失消融 (triple_dynamic)
    #"hano_dual triple_dynamic false ${DATASET_MASS_100000_RANDOM} mass100k_hano_dual_triple_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # GPU 2: WT-Blind 100k (对标 unet1d / hano_v1, 验证 NaN 修复)
    # "hano_dual hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM_LEVEL} blind100k_hano_dual_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"

    # "biom_refiner_plus hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM_LEVEL} blind100k_biom_refiner_plus_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    # "unet1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM_LEVEL} blind100k_unet1d_hybrid_w64 24 64 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    # "unet1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} mass100k_unet1d_hybrid_w64 24 64 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # ====== (已完成) HANO v1 纯物理路径实验 ======
    # "hano triple_dynamic false ${DATASET_MASS_100000_BLIND} blind100k_hano_triple_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    # "unet1d hybrid_mse_rel false ${DATASET_MASS_100000_BLIND} mass100k_unet1d_hybrid_w64 24 64 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    # "biom_refiner_plus triple_dynamic false ${DATASET_MASS_100000_BLIND} blind100k_biom_refiner_triple_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # "hano triple_dynamic false ${DATASET_MASS_100000} blind100k_hano_triple_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    # ---- HANO 消融实验 (需要时取消注释) ----
    # "hano_no_event triple_dynamic false ${DATASET_MASS_100000_BLIND} blind100k_hano_no_event_triple_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    # "hano_no_flux triple_dynamic false ${DATASET_MASS_100000_BLIND} blind100k_hano_no_flux_triple_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    # "hano_no_graph triple_dynamic false ${DATASET_MASS_100000_BLIND} blind100k_hano_no_graph_triple_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 4"
    # "hano_refine triple_dynamic false ${DATASET_MASS_100000_BLIND} blind100k_hano_refine_triple_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 4"
    # # ====== 👑👑👑 1. 旗舰最强机制模型在 MASS 100,000 (含 Motif) 数据集上的决战配置 👑👑👑 ======
    # # GPU 0: 👑 全项目最高精三元动力学终极大融合旗舰 biom_refiner_plus + Triple-Dynamic 复合损失
    # "biom_refiner_plus triple_dynamic false ${DATASET_MASS_100000} mass100k_biom_refiner_plus_triple_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    # # GPU 1: 👑 超高精破百纪录保持者 biom_refiner_curvature + Triple-Dynamic 复合损失 (二阶曲率加速度模型)
    # "biom_refiner_curvature triple_dynamic false ${DATASET_MASS_100000} mass100k_biom_refiner_curv_triple_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # # GPU 2: 👑 生物化学计量反应图拓扑注意力模型 biom_refiner_graph + Hybrid 损失
    # "biom_refiner_graph hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_biom_refiner_graph_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    # # GPU 3: 👑 经典双阶粗精细化基准模型 BioRefiner + Hybrid 损失
    # "biom_refiner hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_biom_refiner_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"

    # # ====== 🌟🌟🌟 2. RealPDEBench & 经典基线神经算子全家桶 (MASS 100,000 含 Motif 数据集) 🌟🌟🌟 ======
    # # GPU 4: RealPDEBench 标准 1D FNO (通道级频域交互 + FiLM 参数调制)
    # "fno1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_fno1d_standard_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 4"
    # # GPU 5: RealPDEBench / PDEBench 标准 1D U-Net 算子 (多尺度残差跳跃连接, width=64 保持轻量)
    # "unet1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_unet1d_hybrid_w64 24 64 256 100 none 8.8e-4 8 0 0 0 one_time 5"
    # # GPU 6: RealPDEBench 连续希尔伯特函数空间线性注意力 Galerkin Transformer
    # "galerkin_transformer1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_galerkin_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 6"
    # # GPU 7: Lu Lu et al. 经典 DeepONet (Branch-Trunk 深度算子网络)
    # "deeponet hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_deeponet_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 7"
    # # GPU 4: ICLR 2023 RealPDEBench 连续带限抗混叠卷积神经算子 (CNO1d)
    # "cno1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_cno1d_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 4"
    # # GPU 5: NeurIPS 2021 RealPDEBench 多小波连续神经算子 (MWT1d)
    # "mwt1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_mwt1d_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 5"
    # # GPU 6: RealPDEBench 物理切片注意力算子 (Transolver1d)
    # "transolver1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_transolver1d_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 6"
    # # GPU 1: ICLR/NeurIPS 2024 RealPDEBench 双路径算子 Transformer (DPOT1d / AFNO)
    # "dpot1d hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_dpot1d_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # # GPU 7: 1D 空洞残差网络 (ResNetFiLM + 通道注意力)
    # "resnet_film hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_resnet_film_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 7"
    # # GPU 2: 超网络调制傅里叶神经算子 (HyperFNO)
    # "hyper_fno hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_hyper_fno_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    # # GPU 3: ICLR 2021 原版单场 FNO (PureFNO)
    # "pure_fno hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_pure_fno_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    # # GPU 0: CMAME 2022 原版小波算子 (PureWNO, width=64 防小波树 OOM)
    # "pure_wno hybrid_mse_rel false ${DATASET_MASS_100000} mass100k_pure_wno_hybrid_w64 24 64 256 100 none 8.8e-4 8 0 0 0 one_time 0"

    # # ====== 🎲🎲🎲 3. 全模型在 MASS 100,000 Random (无 Motif 纯随机基线) 数据集上的消融对照 🎲🎲🎲 ======
    # # GPU 0: 👑 大融合旗舰 biom_refiner_plus (Random 100k 对照)
    # "biom_refiner_plus triple_dynamic false ${DATASET_MASS_100000_RANDOM} random100k_biom_refiner_plus_triple_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 0"
    # # GPU 1: 👑 二阶曲率模型 biom_refiner_curvature (Random 100k 对照)
    # "biom_refiner_curvature triple_dynamic false ${DATASET_MASS_100000_RANDOM} random100k_biom_refiner_curv_triple_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # # GPU 2: 👑 生物化学计量图拓扑注意力 biom_refiner_graph (Random 100k 对照)
    # "biom_refiner_graph hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_biom_refiner_graph_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    # # GPU 3: 👑 双阶粗精细化基准 BioRefiner (Random 100k 对照)
    # "biom_refiner hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_biom_refiner_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # # GPU 4: RealPDEBench 标准 1D FNO (Random 100k 对照)
    # "fno1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_fno1d_standard_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # # GPU 5: RealPDEBench 1D U-Net (Random 100k 对照)
    # "unet1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_unet1d_hybrid_w64 24 64 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    # # GPU 6: RealPDEBench Galerkin Transformer (Random 100k 对照)
    # "galerkin_transformer1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_galerkin_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    # # GPU 7: Lu Lu 经典 DeepONet (Random 100k 对照)
    # "deeponet hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_deeponet_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # # GPU 4: RealPDEBench CNO1d (Random 100k 对照)
    # "cno1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_cno1d_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    # # GPU 5: RealPDEBench MWT1d (Random 100k 对照)
    # "mwt1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_mwt1d_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    # # GPU 6: RealPDEBench Transolver1d (Random 100k 对照)
    # "transolver1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_transolver1d_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 2"
    # # GPU 1: RealPDEBench DPOT1d (Random 100k 对照)
    # "dpot1d hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_dpot1d_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 1"
    # # GPU 7: ResNetFiLM (Random 100k 对照)
    # "resnet_film hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_resnet_film_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    # # GPU 2: HyperFNO (Random 100k 对照)
    # "hyper_fno hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_hyper_fno_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    # # GPU 3: PureFNO (Random 100k 对照)
    # "pure_fno hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_pure_fno_hybrid_w192 24 192 256 100 none 8.8e-4 8 0 0 0 one_time 3"
    # # GPU 0: PureWNO (Random 100k 对照, width=64 防 OOM)
    # "pure_wno hybrid_mse_rel false ${DATASET_MASS_100000_RANDOM} random100k_pure_wno_hybrid_w64 24 64 256 100 none 8.8e-4 8 0 0 0 one_time 0"
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
# 1. 动态自适应显存装载与资源配置 (零单任务限额, 纯实测动态驱动)
# =========================================================================
MEM_SAFETY_MARGIN_MB=2048   # 仅预留 2GB 防 CUDA 内存碎片与波动的安全缓冲 (绝不限制单任务显存)
STABILIZE_WAIT_SEC=35       # 任务启动后等待显存分配稳定的秒数 (让 PyTorch 完成显存建立)
MAX_CONCURRENT_PER_GPU=1    # 智能串行调度: 每卡严格执行单任务串行排队 (前序实验完成并评测后再执行下一个实验)
MIN_LAUNCH_VRAM_MB=12000    # 启动新任务所需的最低剩余可用显存阈值 (12GB)

get_gpu_used_vram() {
    local g=$1
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' '
}

get_gpu_free_vram() {
    local g=$1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' '
}

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

        read -r MODEL LOSS_TYPE USE_ADJ DATASET MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE GPU_ID <<< "$EXP_TRIM"
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
echo "📋 GPU 任务排队与自适应分发摘要 (总计 $TOTAL_TASKS 个任务，分布在 ${#ALL_GPUS[@]} 张 GPU):"
for gpu in "${ALL_GPUS[@]}"; do
    echo "   • GPU $gpu: 队列共 ${GPU_TASK_COUNT[$gpu]} 个任务 (自适应混合调度: 显存充裕时自动并发，不足时安全串行)"
done
echo "==========================================================="
echo ""

# ====== 进程追踪与优雅终止信号拦截 (Ctrl+C / SIGTERM) ======
PIDS=()

cleanup_all() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🛑 捕获中断信号 (Ctrl+C / SIGTERM)，正在强制终止所有任务与子进程..."
    trap '' SIGINT SIGTERM  # 避免重复触发
    
    # 终止记录在案的所有 Worker 及其子进程树
    for pid in "${PIDS[@]}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in "${PIDS[@]}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            pkill -9 -P "$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    # 清理当前 shell 派生的所有直接/间接子进程 (如正在执行的 train.py/eval.py)
    pkill -P $$ 2>/dev/null || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🛑 所有子进程已彻底清理，安全退出。"
    exit 130
}
trap cleanup_all SIGINT SIGTERM

# ====== 单任务执行体 (Train + Eval) ======
run_single_task() {
    local SEED=$1
    local EXP=$2
    local gpu=$3
    local task_idx=$4
    local total_for_gpu=$5

    local MODEL LOSS_TYPE USE_ADJ DATASET MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE GPU_ID
    read -r MODEL LOSS_TYPE USE_ADJ DATASET MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE GPU_ID <<< "$EXP"
    GPU_ID=${GPU_ID:-$gpu}
    ROLLOUT_TYPE=${ROLLOUT_TYPE:-one_time}

    local ADJ_PARAM=""
    if [ "$USE_ADJ" = "true" ]; then
        ADJ_PARAM="--adj_matrix_path mydata/yeast_signed_adjacency_matrix.npy"
    fi

    # 构建 AL 专属参数
    local AL_PARAMS=""
    if [ "$AL_STRATEGY" != "none" ]; then
        local TRIGGER_EPOCH=10
        if [ "$AL_STRATEGY" = "pial" ]; then
            TRIGGER_EPOCH=50
        elif [ "$AL_STRATEGY" = "mutant_aware" ]; then
            TRIGGER_EPOCH=100
        elif [[ "$AL_STRATEGY" == al4pde_* || "$AL_STRATEGY" = "maal" ]]; then
            TRIGGER_EPOCH=30
        fi
        AL_PARAMS="--al_strategy $AL_STRATEGY --al_trigger_epoch $TRIGGER_EPOCH --al_num_add $AL_NUM_ADD --al_perturbation $AL_PERTURBATION --al_mae_threshold $AL_MAE_THRESHOLD"

        if [ "$AL_STRATEGY" = "mutant_aware" ]; then
            AL_PARAMS="$AL_PARAMS --atlas_csv $ATLAS_CSV"
        fi

        if [[ "$AL_STRATEGY" == al4pde_* || "$AL_STRATEGY" = "maal" ]]; then
            AL_PARAMS="$AL_PARAMS --al_pool_dataset $DATASET_MASS_V3 --al_initial_train_size 2000"
        fi
    else
        AL_PARAMS="--al_strategy none"
    fi

    # 格式化学习率字符串
    local LR_STR
    if (( $(echo "$LR >= 0.001" | bc -l 2>/dev/null || echo 0) )); then
        LR_STR=$(printf "%.4g" "$LR")
    else
        LR_STR=$(printf "%.1e" "$LR" | sed 's/-0/-/g')
    fi

    local DATASET_NAME
    DATASET_NAME=$(basename "$DATASET" .npz)
    local RUN_NAME="${MODEL}_${AL_STRATEGY}_${LOSS_TYPE}_${DATASET_NAME}_modes${MODES}"
    if [ -n "$NORM_TYPE" ] && [ "$NORM_TYPE" != "log1p" ]; then
        RUN_NAME="${RUN_NAME}_${NORM_TYPE}"
    fi
    RUN_NAME="${RUN_NAME}_seed${SEED}"

    local EXP_DIR CKPT_DIR EVAL_OUT_DIR LOG_FILE DIR_PARAMS
    if [ "$USE_HIERARCHICAL" = "1" ]; then
        local HPARAM_SIG
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
        if [ -n "$NORM_TYPE" ] && [ "$NORM_TYPE" != "log1p" ]; then
            HPARAM_SIG="${HPARAM_SIG}_${NORM_TYPE}"
        fi

        EXP_DIR="${EXP_ROOT}/${DATASET_NAME}/${MODEL}/${LOSS_TYPE}_${AL_STRATEGY}/${HPARAM_SIG}/seed_${SEED}"
        CKPT_DIR="${EXP_DIR}/checkpoints"
        EVAL_OUT_DIR="${EXP_DIR}/eval"
        LOG_FILE="${EXP_DIR}/train.log"
        mkdir -p "$CKPT_DIR" "$EVAL_OUT_DIR"
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

    local PINN_PARAMS=""
    if [ "$LOSS_TYPE" = "pinn_residual" ] || [ "$LOSS_TYPE" = "pino" ]; then
        PINN_PARAMS="--lambda_phys 0.01 --lambda_jump 0.001 --lambda_penalty 1.0 --residual_subsample 10 --event_weight 0.1"
    fi
    if [ "$LOSS_TYPE" = "pino" ]; then
        PINN_PARAMS="$PINN_PARAMS --lambda_h1 0.1 --h1_beta 1.0"
    fi
    if [ "$LOSS_TYPE" = "hybrid_pino_colloc" ] || [ "$LOSS_TYPE" = "hybrid_all" ]; then
        PINN_PARAMS="--lambda_phys 0.01 --lambda_jump 0.001 --lambda_penalty 1.0 --residual_subsample 10 --event_weight 0.1 --pino_event_strategy decoupled --huber_delta 1.0"
    fi
    if [ "$LOSS_TYPE" = "hybrid_bifurcate" ] || [ "$LOSS_TYPE" = "hybrid_all" ]; then
        PINN_PARAMS="$PINN_PARAMS --lambda_pattern 0.1 --lambda_arrest 0.05"
    fi

    local WANDB_FLAG=""
    if [ "$NO_WANDB" = "1" ]; then
        WANDB_FLAG="--no_wandb"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 [GPU $gpu | 任务 $task_idx/$total_for_gpu] 启动: $RUN_NAME -> 目录: $EXP_DIR"

    echo "===========================================================" > "$LOG_FILE" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始训练 (GPU=$gpu | 任务 $task_idx/$total_for_gpu): $RUN_NAME" >> "$LOG_FILE" 2>&1
    echo "===========================================================" >> "$LOG_FILE" 2>&1

    local SKIP_TRAIN=0
    local SKIP_EVAL=0

    # 步骤 A: 训练
    if [ "$SKIP_TRAIN" = "0" ]; then
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
            --norm_type $NORM_TYPE \
            $DIR_PARAMS \
            $WANDB_FLAG >> "$LOG_FILE" 2>&1

        local TRAIN_EXIT=$?
        # 自适应 OOM 容错重试: 若遇到显存不足 (CUDA OOM 或 SIGKILL=137)，自动将 BatchSize 减半重试一次
        if [ $TRAIN_EXIT -ne 0 ]; then
            if grep -qi "out of memory" "$LOG_FILE" 2>/dev/null || [ $TRAIN_EXIT -eq 137 ]; then
                local FALLBACK_BS=$((BATCH_SIZE / 2))
                echo "⚠️ [GPU $gpu | 任务 $task_idx/$total_for_gpu] 检测到显存溢出 (OOM)，自动自适应降级 BatchSize: $BATCH_SIZE -> $FALLBACK_BS 重试训练..." >> "$LOG_FILE" 2>&1
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 [GPU $gpu | 任务 $task_idx/$total_for_gpu] OOM 自适应降级重试: $RUN_NAME (BS: $BATCH_SIZE -> $FALLBACK_BS)"
                if [ -d "$CKPT_DIR" ]; then
                    rm -f "${CKPT_DIR}"/*.ckpt
                fi
                CUDA_VISIBLE_DEVICES=$gpu $PYTHON_BIN train.py \
                    --model $MODEL \
                    --wavelet haar \
                    --loss_type $LOSS_TYPE \
                    --dataset_name "$DATASET" \
                    --modes $MODES \
                    --width $WIDTH \
                    --batch_size $FALLBACK_BS \
                    --max_epochs $EPOCHS \
                    --seed $SEED \
                    --name $RUN_NAME \
                    --lr $LR \
                    --warmup_epochs $WARMUP \
                    --num_workers 4 \
                    --val_batch_size $FALLBACK_BS \
                    --devices 1 \
                    $AL_PARAMS \
                    $ADJ_PARAM \
                    $PINN_PARAMS \
                    --rollout_type $ROLLOUT_TYPE \
                    --norm_type $NORM_TYPE \
                    $DIR_PARAMS \
                    $WANDB_FLAG >> "$LOG_FILE" 2>&1
                TRAIN_EXIT=$?
            fi
        fi

        if [ $TRAIN_EXIT -ne 0 ]; then
            echo "❌ [GPU $gpu | 任务 $task_idx/$total_for_gpu] 训练异常退出 (exit=$TRAIN_EXIT), 自动跳过评估并保留日志，继续执行下一任务" >> "$LOG_FILE" 2>&1
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ [GPU $gpu | 任务 $task_idx/$total_for_gpu] 训练失败: $RUN_NAME (exit=$TRAIN_EXIT, 详细见 $LOG_FILE)，自动调度下一个任务"
            return 0
        fi
    fi

    # 步骤 B+C: 寻找 Checkpoint 并评估
    if [ "$SKIP_EVAL" = "0" ]; then
        local CKPT_PATH
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
                --norm_type $NORM_TYPE \
                --save_dir "$EVAL_OUT_DIR" \
                --seed $SEED \
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
    return 0
}

# ====== 启动 GPU Worker 进程 (每张 GPU 独立启动一个自适应排队 Worker) ======
for gpu in "${ALL_GPUS[@]}"; do
    {
        worker_pids=()
        cleanup_worker() {
            trap '' SIGINT SIGTERM
            for p in "${worker_pids[@]}"; do
                kill -TERM "$p" 2>/dev/null || true
            done
            sleep 0.5
            for p in "${worker_pids[@]}"; do
                kill -9 "$p" 2>/dev/null || true
            done
            exit 130
        }
        trap cleanup_worker SIGINT SIGTERM

        # 解析该卡的所有任务列表为数组
        mapfile -t TASKS_FOR_GPU <<< "${GPU_TASK_LIST[$gpu]}"
        VALID_TASKS=()
        for item in "${TASKS_FOR_GPU[@]}"; do
            [ -n "$item" ] && VALID_TASKS+=("$item")
        done

        total_for_gpu=${#VALID_TASKS[@]}
        [ "$total_for_gpu" -eq 0 ] && exit 0

        ACTIVE_PIDS=()
        LAST_TASK_VRAM_DELTA=0

        reap_active_pids() {
            local still_running=()
            for p in "${ACTIVE_PIDS[@]}"; do
                if kill -0 "$p" 2>/dev/null; then
                    still_running+=("$p")
                else
                    wait "$p" 2>/dev/null
                    local status=$?
                    if [ $status -eq 130 ] || [ $status -eq 143 ]; then
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🛑 [GPU $gpu] 任务被用户主动中断 (code=$status)，终止该卡后续队列！"
                        cleanup_worker
                    fi
                fi
            done
            ACTIVE_PIDS=("${still_running[@]}")
            worker_pids=("${still_running[@]}")
        }

        # 独占模式检测 (如 GPU 7 的 Exclusive_Process)
        gpu_mode=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' ')
        is_exclusive=0
        if [ "$gpu_mode" = "Exclusive_Process" ] || [ "$gpu" -eq 7 ]; then
            is_exclusive=1
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔒 [GPU $gpu] 检测为独占计算模式 (Exclusive_Process)，强制启用严格单进程串行排队机制！"
        fi

        for ((idx=0; idx<total_for_gpu; idx++)); do
            task_line="${VALID_TASKS[$idx]}"
            task_idx=$((idx + 1))
            IFS='|' read -r SEED EXP <<< "$task_line"

            # ----------------------------------------------------
            # 步骤 1: 检查资源与队列，决定是否需要等待
            # ----------------------------------------------------
            reap_active_pids
            while true; do
                # 独占卡规则: 如果已有本 worker 任务在跑，必须等待其结束
                if [ "$is_exclusive" -eq 1 ] && [ "${#ACTIVE_PIDS[@]}" -gt 0 ]; then
                    sleep 5
                    reap_active_pids
                    continue
                fi

                # 单卡并发数上限规则 (若设了 MAX_CONCURRENT_PER_GPU > 0，或独占卡强制为 1)
                local max_c=$MAX_CONCURRENT_PER_GPU
                [ "$is_exclusive" -eq 1 ] && max_c=1
                if [ "$max_c" -gt 0 ] && [ "${#ACTIVE_PIDS[@]}" -ge "$max_c" ]; then
                    sleep 5
                    reap_active_pids
                    continue
                fi

                # 独占卡额外保护: 检查是否有外部非本 worker 进程占用该独占 GPU
                if [ "$is_exclusive" -eq 1 ]; then
                    ext_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' ')
                    if [ -n "$ext_procs" ]; then
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏳ [GPU $gpu] 独占显卡当前有外部进程 (PID: $ext_procs) 占用，等待其释放..."
                        sleep 10
                        continue
                    fi
                fi

                # 显存阈值检查: 启动任何新任务前，确保当前卡剩余显存充足
                curr_free=$(get_gpu_free_vram "$gpu")
                required_vram=${MIN_LAUNCH_VRAM_MB:-12000}
                if [ "${#ACTIVE_PIDS[@]}" -gt 0 ] && [ "$LAST_TASK_VRAM_DELTA" -gt 0 ]; then
                    required_vram=$((LAST_TASK_VRAM_DELTA + MEM_SAFETY_MARGIN_MB))
                fi

                # 若当前卡剩余显存不足预估量，则转入安全等待
                if [ "$curr_free" -lt "$required_vram" ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏳ [GPU $gpu] 剩余显存 (${curr_free}MiB) 低于启动安全阈值 (${required_vram}MiB)，等待显存释放..."
                    sleep 10
                    reap_active_pids
                    continue
                fi

                # 显存充裕或卡处于空闲状态，跳出等待
                break
            done

            # ----------------------------------------------------
            # 步骤 2: 启动新任务 (完全自由分配显存，不设单任务限额)
            # ----------------------------------------------------
            mem_before=$(get_gpu_used_vram "$gpu")
            run_single_task "$SEED" "$EXP" "$gpu" "$task_idx" "$total_for_gpu" &
            task_pid=$!
            ACTIVE_PIDS+=("$task_pid")
            worker_pids+=("$task_pid")

            # ----------------------------------------------------
            # 步骤 3: 若为串行模式/独占卡则同步等待完成；否则动态并发实测显存增量
            # ----------------------------------------------------
            if [ "$is_exclusive" -eq 1 ] || [ "$MAX_CONCURRENT_PER_GPU" -eq 1 ]; then
                wait "$task_pid" 2>/dev/null || true
                reap_active_pids
            else
                if [ "$idx" -lt "$((total_for_gpu - 1))" ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏱️ [GPU $gpu] 等待 ${STABILIZE_WAIT_SEC}s 待任务显存分配稳定，以动态评估下一个任务能否并发加入..."
                    for ((w=0; w<STABILIZE_WAIT_SEC; w+=5)); do
                        sleep 5
                        if ! kill -0 "$task_pid" 2>/dev/null; then
                            break
                        fi
                    done

                    mem_after=$(get_gpu_used_vram "$gpu")
                    delta=$((mem_after - mem_before))
                    if [ "$delta" -gt 500 ]; then
                        LAST_TASK_VRAM_DELTA=$delta
                        curr_free=$(get_gpu_free_vram "$gpu")
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📊 [GPU $gpu] 实测任务实际消耗显存: ${LAST_TASK_VRAM_DELTA}MiB (~$((LAST_TASK_VRAM_DELTA / 1024))GB)，当前剩余显存: ${curr_free}MiB (~$((curr_free / 1024))GB)"
                    fi
                fi
            fi
        done

        # ----------------------------------------------------
        # 步骤 4: 等待该卡所有派发的并行/串行任务全部完成
        # ----------------------------------------------------
        while [ "${#ACTIVE_PIDS[@]}" -gt 0 ]; do
            reap_active_pids
            sleep 5
        done

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
