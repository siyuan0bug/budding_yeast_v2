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
#export WANDB_API_KEY="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"  #v1版本
export WANDB_API_KEY="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"  #active版本

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
# v5_boundary 数据集 (V5 扩展家族配置不变, target 采样统一改为 boundary 0 值邻域模式)
#   差异仅在 mode 字段: V5 部分 density / 部分 boundary, V5_boundary 全部统一为 boundary
#   连续参数采样范围 [0, center_val*(1+pert)], 强制覆盖 0 值邻域 (完全敲除边界)
#   KO 突变体用默认值作为上界参考; CDH1/BUB2 成对参数 30% 概率显式选 (0,0)
DATASET_V5_BOUNDARY="${DATA_DIR}/lhs_cheat_origin_v5_boundary_210min_500steps_dual_labels.npz"
# v5 无 target 副本 (去除 _LHS_T 定向采样, 只保留 Real Mutant + 普通 LHS, 用于对照实验)
DATASET_V5_NO_TARGET="${DATA_DIR}/lhs_cheat_origin_v5_no_target_210min_500steps_dual_labels.npz"
# v6 数据集 (V5 基础 + P0-P5 定向修复: 修复 tem1_ts 干扰及 CLB2_db_Gal 盲区等)
DATASET_V6="${DATA_DIR}/lhs_cheat_origin_v6_210min_500steps_dual_labels.npz"
# nofilter 数据集 (去除所有生成和清洗拦截，强制样本数一致)
DATASET_nofilter="${DATA_DIR}/lhs_cheat_origin_no_filter_210min_500steps_dual_labels.npz"
# guaranteed 数据集 (保留过滤拦截, 但补采保证每个 mutant 过滤后恰好 1000 个有效样本)
DATASET_guaranteed="${DATA_DIR}/lhs_cheat_origin_guaranteed_210min_500steps_dual_labels.npz"
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
# mutant_aware 策略专用: atlas CSV (含 lhs_MAE_min, lhs_n_samples 列)
ATLAS_CSV="./eval_result/old_hyper_fno_none_mse_only_210m_500s_modes32_seed42/mutant_atlas_merged.csv"

SEEDS=(42) #42,1024,3407          

# ==========================================
# 终极多任务实验清单
#
#
# 说明:
#   MODEL         : 模型名 (hyper_fno, cross_fno, pure_fno, patchtst, ...)
#   LOSS_TYPE     : 损失函数 (mse_only, physics_informed, pinn_residual, h1, pino)
#                   h1=H1 Sobolev 损失 (时域L2 + 频域梯度L2, 来自 HyperNOs H1relLoss_1D)
#                   pino=PINO 损失 (MSE + ODE残差 + H1频域梯度, 来自 PINO 论文)
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
    
    # ====== [临时] batch size / lr OOM 边界探测 (激进档, √缩放 lr, GPU 0-3) ======
    # 基准: v5 none baseline (batch=64, lr=6e-4, warmup=10) 已验证健康收敛
    # 激进倍增阶梯 512/1024/2048/4096 (相对 bs64 = 8×/16×/32×/64×), 宽跨度一次性 bracket 出 OOM 天花板
    # (仅 4 张卡 GPU 0-3, 每档须独占一卡避免显存争抢, 故用大跨度倍增替代细粒度阶梯)
    # √缩放: lr(bs) = 6e-4 × √(bs/64) → bs512=1.7e-3 / bs1024=2.4e-3 / bs2048=3.39e-3 / bs4096=4.8e-3
    # 短跑冒烟: EPOCHS=5 + WARMUP=3; 且脚本内 SKIP_EVAL=1 只训练不评估 (OOM 探测无需 eval)
    # 独立 tag v5_probe_bs{N}: 与现有 v5 baseline 完全隔离, 各占一张卡 (GPU 0-3), 4 档并行
    # 判读: 启动即崩 CUDA OOM → 该 batch 超上界(取能跑通的最大档并回退留 10~15% 余量);
    #       OOM 在首个 step 的显存分配即触发, 早于 nan, 故激进 lr 不影响 OOM 判定;
    #       未 OOM 时再看 ep1-2 是否 train_loss=nan (nan → 降 lr 或加大 warmup)
    # 提示: 保持 precision=bf16-mixed + gradient_clip_val=1.0 (train.py 已固定, 是防 nan 主防线)
    #"hyper_fno mse_only false ${DATASET_V5} v5_probe_bs512 32 32 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    #"hyper_fno mse_only false ${DATASET_V5} v5_probe_bs1024 32 32 1024 100 none 2.4e-3 10 0 0 0 one_time 1"
    #"hyper_fno mse_only false ${DATASET_V5} v5_probe_bs2048 32 32 2048 100 none 3.39e-3 10 0 0 0 one_time 2"
    #"hyper_fno mse_only false ${DATASET_V5} v5_probe_bs4096 32 32 4096 100 none 4.8e-3 10 0 0 0 one_time 3"
    
    # ====== MASS 数据集实验 (mass_dataset_generator.py 生成) ======
    # 与 v5_probe_bs1024 完全相同的模型/训练配置, 仅替换数据集为 DATASET_MASS、标签为 mass。
    # 数据集无 LHS 家族结构 (real mutants + MASS_* 合成样本), datamodule 会自动跳过家族分层、
    # 仅按 Pattern 做宏观 train/val/test 划分 (real mutants 全部作为 test_real)。
    # ====== 8卡全并行旗舰神经算子全面评测 (8 GPUs 0~7 Parallel Benchmark) ======
    #"hyper_fno hybrid_mse_rel false ${DATASET_MASS} mass 32 32 512 100 none 1.7e-3 10 0 0 0 one_time 6"
    #"cross_fno hybrid_mse_rel false ${DATASET_MASS} mass 32 32 128 100 none 3e-4 10 0 0 0 one_time 0"
    #"multiscale_diffgraph hybrid_mse_rel false ${DATASET_MASS} mass_multiscale_diffgraph 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    #"dualpath_spectralgraph hybrid_mse_rel false ${DATASET_MASS} mass_dualpath_spectralgraph 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    #"bio_statespace_graph hybrid_mse_rel false ${DATASET_MASS} mass_bio_statespace 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    #"gated_graph_resnet_film hybrid_mse_rel false ${DATASET_MASS} mass_gated_graph_hybrid 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    #"graph_resnet_film hybrid_mse_rel false ${DATASET_MASS} mass_graph_resnet_hybrid 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    #"resnet_film hybrid_mse_rel false ${DATASET_MASS} mass_resnet_hybrid 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    #"channel_hyper_fno hybrid_mse_rel false ${DATASET_MASS} mass_channel_hybrid 32 32 512 100 none 1.7e-3 10 0 0 0 one_time 6"
    #"deeponet hybrid_mse_rel false ${DATASET_MASS} mass_deeponet_hybrid 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 7"

    #"resnet_film hybrid_mse_rel false ${DATASET_MASS} mass_resnet_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"

    # ====== 新一代生化多关系图神经算子 (Bio-Hetero Multirelational Graph Operators) ======
    #"bio_hetero_causal_graph hybrid_mse_rel false ${DATASET_MASS} mass_bio_hetero 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    #"hetero_graph_resnet_film hybrid_mse_rel false ${DATASET_MASS} mass_hetero_resnet 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    #"hetero_graph_dualpath hybrid_mse_rel false ${DATASET_MASS} mass_hetero_dualpath 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 7"
    #"hetero_graph_channel_hyper_fno hybrid_mse_rel false ${DATASET_MASS} mass_hetero_channel 32 32 512 100 none 1.7e-3 10 0 0 0 one_time 6"

    # ====== 三大原创学术级旗舰神经算子 (Flagship Novel Neural Operators: DyRFO / ST-HyperNO / PC-BioNO) ======
    #"dyrfo hybrid_mse_rel false ${DATASET_MASS} mass_dyrfo_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    #"st_hyperno hybrid_mse_rel false ${DATASET_MASS} mass_st_hyperno_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    #"pc_bio_operator hybrid_mse_rel false ${DATASET_MASS} mass_pc_bio_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 3"

    # ====== 终极物理生化机制算子 (BioMechanisticOperator / BioMechNet - 线性基线归一化已完成) ======
    #"biom_full hybrid_mse_rel false ${DATASET_MASS} mass_biom_full_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 7"
    #"biom_titration hybrid_mse_rel false ${DATASET_MASS} mass_biom_titration_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 6"
    #"biom_event hybrid_mse_rel false ${DATASET_MASS} mass_biom_event_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    #"biom_pathway hybrid_mse_rel false ${DATASET_MASS} mass_biom_pathway_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"

    # ====== 🌟 新一代 Log1p 平滑对数归一化基线实验 (已完成评测归档) ======
    #"resnet_film hybrid_mse_rel false ${DATASET_MASS} mass_log1p_resnet_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    #"st_hyperno hybrid_mse_rel false ${DATASET_MASS} mass_log1p_st_hyperno_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    #"biom_pathway hybrid_mse_rel false ${DATASET_MASS} mass_log1p_biom_pathway_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    #"biom_full hybrid_mse_rel false ${DATASET_MASS} mass_log1p_biom_full_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    #"dyrfo hybrid_mse_rel false ${DATASET_MASS} mass_log1p_dyrfo_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 6"
    #"biom_titration hybrid_mse_rel false ${DATASET_MASS} mass_log1p_biom_titration_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 7"

    # ====== 🚀 动力学感知创新损失全旗舰实验 (Phase Space & Soft-DTW Loss Suite - MASS v1 已完成归档) ======
    #"biom_pathway phase_derivative false ${DATASET_MASS} mass_log1p_phase_biom_pathway_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    #"resnet_film phase_derivative false ${DATASET_MASS} mass_log1p_phase_resnet_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    #"st_hyperno phase_derivative false ${DATASET_MASS} mass_log1p_phase_st_hyperno_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    #"dyrfo phase_derivative false ${DATASET_MASS} mass_log1p_phase_dyrfo_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 6"
    #"biom_pathway soft_dtw false ${DATASET_MASS} mass_log1p_dtw_biom_pathway_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    #"resnet_film soft_dtw false ${DATASET_MASS} mass_log1p_dtw_resnet_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 5"

    # ====== 🌟🌟🌟 新一代 MASS v2 / v4.5 高阶救援与拓扑重平衡数据集全旗舰实验 (8 卡第一轮 - 已完成归档) ======
    #"st_hyperno phase_derivative false ${DATASET_MASS_V2} massv2_st_hyperno_phase_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    #"biom_pathway soft_dtw false ${DATASET_MASS_V2} massv2_biom_pathway_dtw_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    #"biom_pathway hybrid_mse_rel false ${DATASET_MASS_V2} massv2_biom_pathway_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    #"biom_pathway phase_derivative false ${DATASET_MASS_V2} massv2_biom_pathway_phase_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    #"resnet_film hybrid_mse_rel false ${DATASET_MASS_V2} massv2_resnet_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    #"resnet_film soft_dtw false ${DATASET_MASS_V2} massv2_resnet_dtw_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    #"dyrfo hybrid_mse_rel false ${DATASET_MASS_V2} massv2_dyrfo_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 6"
    #"biom_full hybrid_mse_rel false ${DATASET_MASS_V2} massv2_biom_full_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 7"

    # ====== 🌟🌟🌟 MASS v2 第二轮：三元动力学复合损失 (Triple-Dynamic) 与 双阶粗精残差细化算子 (BioRefiner) 7 卡大决战 (已完成归档) ======
    #"biom_pathway triple_dynamic false ${DATASET_MASS_V2} massv2_biom_pathway_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    #"biom_full triple_dynamic false ${DATASET_MASS_V2} massv2_biom_full_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    #"dyrfo triple_dynamic false ${DATASET_MASS_V2} massv2_dyrfo_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    #"st_hyperno triple_dynamic false ${DATASET_MASS_V2} massv2_st_hyperno_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    #"biom_refiner hybrid_mse_rel false ${DATASET_MASS_V2} massv2_biom_refiner_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    #"biom_refiner phase_derivative false ${DATASET_MASS_V2} massv2_biom_refiner_phase_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    #"biom_refiner triple_dynamic false ${DATASET_MASS_V2} massv2_biom_refiner_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 6"

    # ====== 🌟🌟🌟 MASS v2 第三轮：BioRefiner-Plus 终极大升级与 3 大独立消融子模型 7 卡决战 (已完成归档) ======
    #"biom_refiner_curvature hybrid_mse_rel false ${DATASET_MASS_V2} massv2_biom_refiner_curv_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    #"biom_refiner_pyramid hybrid_mse_rel false ${DATASET_MASS_V2} massv2_biom_refiner_pyr_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    #"biom_refiner_graph hybrid_mse_rel false ${DATASET_MASS_V2} massv2_biom_refiner_graph_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    #"biom_refiner_plus hybrid_mse_rel false ${DATASET_MASS_V2} massv2_biom_refiner_plus_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    #"biom_refiner_plus phase_derivative false ${DATASET_MASS_V2} massv2_biom_refiner_plus_phase_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    #"biom_refiner_plus triple_dynamic false ${DATASET_MASS_V2} massv2_biom_refiner_plus_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    #"biom_refiner_curvature triple_dynamic false ${DATASET_MASS_V2} massv2_biom_refiner_curv_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 6"

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
    "biom_refiner_curvature triple_dynamic false ${DATASET_MASS_RANDOM} random_biom_refiner_curv_triple_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"

    # ====== 🌟🌟🌟 MASS v3 基线模型横向对比评测 (PureFNO, DeepONet, ResNetFiLM, PureWNO, HyperFNO) 🌟🌟🌟 ======
    # GPU 0: PureFNO (ICLR 2021 原版) + Hybrid 损失 (width=64, batch_size=512, epochs=100)
    #"pure_fno hybrid_mse_rel false ${DATASET_MASS_V3} massv3_pure_fno_hybrid_width64 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    # GPU 1: DeepONet (已完成评估: Real MAE=0.1567, Corr=0.9641)
    #"deeponet hybrid_mse_rel false ${DATASET_MASS_V3} massv3_deeponet_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    # GPU 2: ResNetFiLM (已完成评估: Real MAE=0.0495, Corr=0.9961)
    #"resnet_film hybrid_mse_rel false ${DATASET_MASS_V3} massv3_resnet_film_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    # GPU 3: PureWNO (CMAME 2022 原版) + Hybrid 损失 (width=64, batch_size=512, epochs=100)
    #"pure_wno hybrid_mse_rel false ${DATASET_MASS_V3} massv3_pure_wno_hybrid_width64 32 64 256 100 none 1.7e-3 10 0 0 0 one_time 3"
    # GPU 4: HyperFNO (已完成评估: Real MAE=0.0798, Corr=0.9924)
    #"hyper_fno hybrid_mse_rel false ${DATASET_MASS_V3} massv3_hyper_fno_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
    # GPU 5: (备选/按需开启) WNO-Cross (Cross-Attention 增强版小波算子) + Hybrid 损失
    #"wno hybrid_mse_rel false ${DATASET_MASS_V3} massv3_wno_cross_hybrid_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    # GPU 5: CrossFNO (已注释跳过)
    #"cross_fno hybrid_mse_rel false ${DATASET_MASS_V3} mass 32 32 128 100 none 3e-4 10 0 0 0 one_time 5"
    # GPU 6~7: 保留空闲 (留给他人使用)

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

    # ====== al4pde 框架实验 v2.3: 外层重训式 AL (方案 A, 向原版 al4pde 靠拢) ======
    # 5 种策略对比: Random / Uncertainty / Diversity / MAAL / Boundary
    # 数据集: v1 (DATASET_CHEAT) 作为初始训练集, 候选池参数由 lhs_v1_origin_v5.py + lhs_v1_origin_v5_boundary.py 动态生成
    # v2.3 外层重训式 (al_callback.py + train.py 硬编码):
    #   - 5 轮 AL, 每轮全新模型(冷启动) + 全新余弦周期在完整累积数据上跑满 100 epoch, 取最后一轮最后 epoch
    #   - 初始训练集 5000 (沿用 v2.2 基线), 中间 4 次采集, 每次追加 19000, 总追加 76000 → 总 81000
    #   - 无早停、无 val-best 选择: 依赖余弦退火到近零, 最后 epoch ≈ 收敛点 (对齐原版 al4pde 取最后 epoch)
    #   - 选样与训练解耦: 每轮训练后用当前模型调用 al4pde_select_and_append (在 train.py 外层循环)
    # v2.2 (已作废): 增量式单轮 + 分段 trigger + 递增 num_add + 早停/best 依据 test_loss_real (数据泄露)
    # 候选池: 排除 3 个 WT 家族, 每 mutant 3 类采样 (cdh1 家族 density=500, 其他 300, boundary 全 300)
    # diversity/maal/boundary: 初始化时 ODE 仿真候选池提取 332 维形状特征 + 极端值过滤 + 方案A权重 + MaxDist
    # al4pde_boundary: acquisition = (1-w)*diversity + w*boundary_proximity (0 值邻域接近度)
    # 注: EPOCHS 列=100 (每轮), AL_NUM_ADD 列=1000 (仅占位, 实际每次追加由 al_callback 硬编码为 19000)
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 100 al4pde_random 6e-4 10 1000 0 0 one_time 0"
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 100 al4pde_uncertainty 6e-4 10 1000 0 0 one_time 1"
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 100 al4pde_diversity 6e-4 10 1000 0 0 one_time 2"
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 100 maal 6e-4 10 1000 0 0 one_time 3"
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 100 al4pde_boundary 6e-4 10 1000 0 0 one_time 0"

    # al4pde_mass (新增, 平行于 random/diversity/maal/boundary): 借鉴 mass_dataset_generator.py 分层采样
    #   acquisition = (1-mass_weight)*diversity + mass_weight*mass_priority
    #   mass_priority = 通路频率(high/med/low=6/3/1) × 机制频率(KO/GAL-OE/ts, Chen 2004 统计), 详见 al_callback._compute_mass_priority_scores
    #   mass_weight 默认 0.5 (train.py --al_mass_weight); 与其他 al4pde 策略同配置 (batch=64/EPOCHS=100/lr=6e-4) 以便对比; 独占 GPU 0
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 1024 al4pde_mass 2.4e-3 10 1000 0 0 one_time 1"

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

    # [已注释] H1/PINO 损失实验 (已完成初步验证, 待修复后重跑)
    #"hyper_fno h1 false ${DATASET_CHEAT} v1_h1 32 32 64 200 none 3e-4 10 0 0 0 one_time 1"
    #"hyper_fno pino false ${DATASET_CHEAT} v1_pino 32 32 64 200 none 3e-4 10 0 0 0 one_time 2"

    # 对照实验: v5 无 target 数据集 (验证 target 定向采样对模型的贡献, 区分 target 贡献 vs 普通 LHS 贡献)
    #"hyper_fno mse_only false ${DATASET_V5_NO_TARGET} v5_no_target 32 32 64 200 none 6e-4 10 0 0 0 one_time 3"

    # 对照实验: v5_boundary 数据集 (target 采样统一为 boundary 0 值邻域模式, 验证 boundary 采样对模型的贡献)
    # 参数与 v5_no_target 对照实验保持一致 (mse_only + none + 同超参), 仅数据集不同
    #"hyper_fno mse_only false ${DATASET_V5_BOUNDARY} v5_boundary 32 32 64 200 none 6e-4 10 0 0 0 one_time 1"

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
    #"hyper_fno mse_only false ${DATASET_V5} v5 32 32 64 200 none 6e-4 10 0 0 0 one_time 0"

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
    #"hyper_fno mse_only false ${DATASET_CHEAT} 210m_500s 32 32 64 200 none 6e-4 10 0 0 0 one_time 1"

    # 实验 3: 随机基线
    #"cross_fno mse_only true ${DATASET_CHEAT} 32 16 32 120 random 3e-4 10 5000 0 0 one_time"

    # 实验 4: HGGS
    #"cross_fno mse_only true ${DATASET_CHEAT} hggs 3e-4 10 5000 0 0 one_time"
)

# ==========================================
# 实验管理模式选择:
#   USE_HIERARCHICAL=1 (推荐, 启用新版分层语义化目录 experiments/dataset/model/...)
#   USE_HIERARCHICAL=0 (兼容旧版扁平目录 train_results_v3/ / eval_results_v3/)
# ==========================================
USE_HIERARCHICAL=1
EXP_ROOT="./experiments"
mkdir -p "$EXP_ROOT" logs_v3

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

            if [ "$AL_STRATEGY" = "mutant_aware" ]; then
                AL_PARAMS="$AL_PARAMS --atlas_csv $ATLAS_CSV"
            fi

            if [[ "$AL_STRATEGY" == al4pde_* || "$AL_STRATEGY" = "maal" ]]; then
                AL_PARAMS="$AL_PARAMS --al_pool_dataset $DATASET_V5 --al_initial_train_size 2000"
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
            PINN_PARAMS="--lambda_phys 0.01 --lambda_jump 0.001 --lambda_penalty 1.0 --residual_subsample 10 --event_weight 0.1"
        fi
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

            SKIP_TRAIN=0
            SKIP_EVAL=0

            # 步骤 A: 训练
            if [ "$SKIP_TRAIN" = "0" ]; then
                # 训练前清理旧 checkpoint
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

            # 步骤 B+C: 寻找 Checkpoint 并评估
            if [ "$SKIP_EVAL" = "0" ]; then
                # 步骤 B: 寻找 Checkpoint (优先选 final 或 best)
                CKPT_PATH=$(ls ${CKPT_DIR}/*.ckpt 2>/dev/null | head -n 1)

                if [ -z "$CKPT_PATH" ]; then
                    echo "❌ 错误: 在 ${CKPT_DIR} 没找到 Checkpoint 文件，跳过 Evaluate。" >> "$LOG_FILE" 2>&1
                    exit 1
                fi

                echo "🎯 成功找到 Checkpoint: $CKPT_PATH" >> "$LOG_FILE" 2>&1

                # 步骤 C: 评估
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

