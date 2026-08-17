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
    # run_name = hyper_fno_none_mse_only_mass_modes32_seed42; 独占 GPU 1。
    "hyper_fno mse_only false ${DATASET_MASS} mass 32 32 512 100 none 1.7e-3 10 0 0 0 one_time 1"

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
            # --al_pool_dataset: 遗留参数 (al4pde 候选池由 lhs_v1_origin_v5.py 动态生成, 不读取 npz)
            # --al_initial_train_size 2000: 命令行传入但被 al_callback.py 硬编码覆盖为 15000
            #   K-Means 从 v1 LHS 常规样本中选 15000 个作为初始训练集
            # 候选池: 由 lhs_v1_origin_v5.py + lhs_v1_origin_v5_boundary.py 动态生成
            #   每 mutant 3 类采样各 300 个 (normal 常规 + density 突变参数邻域 + boundary 0值邻域)
            #   排除 3 个 WT 家族, 仅存参数 P_norm + Y_IC, 选中后才 ODE 仿真生成完整轨迹
            if [[ "$AL_STRATEGY" == al4pde_* || "$AL_STRATEGY" = "maal" ]]; then
                AL_PARAMS="$AL_PARAMS --al_pool_dataset $DATASET_V5 --al_initial_train_size 2000"
            fi
        else
            AL_PARAMS="--al_strategy none"
        fi

        RUN_NAME="${MODEL}_${AL_STRATEGY}_${LOSS_TYPE}_${DATASET_TAG}_modes${MODES}_seed${SEED}"
        EVAL_OUT_DIR="./eval_result/${RUN_NAME}"
        LOG_FILE="logs/${RUN_NAME}.log"

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

        echo "🚀 启动实验 -> Seed: $SEED | 策略: $AL_STRATEGY | GPU: $GPU_ID | 日志: $LOG_FILE"

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
                TRAIN_CKPT_DIR="./train_result/${RUN_NAME}"
                if [ -d "$TRAIN_CKPT_DIR" ]; then
                    rm -f "${TRAIN_CKPT_DIR}"/*.ckpt
                    echo "🧹 训练前已清理旧 checkpoint: ${TRAIN_CKPT_DIR}/*.ckpt" >> "$LOG_FILE" 2>&1
                fi
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
            else
                echo "⏭️ SKIP_TRAIN=1, 跳过训练, 直接评估已有 checkpoint" >> "$LOG_FILE" 2>&1
            fi

            # 步骤 B+C: 寻找 Checkpoint 并评估 (SKIP_EVAL=1 时整体跳过, 用于 batch/OOM 冒烟测试)
            if [ "$SKIP_EVAL" = "0" ]; then
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
