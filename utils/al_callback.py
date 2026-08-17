import torch
import torch.nn.functional as F
import numpy as np
import time
import pytorch_lightning as pl
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================
# 标准主动学习框架依赖
# ==============================================================
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# ==============================================================
# 1. 完美、完整地导入原版 LHS 引擎的所有核心组件
# ==============================================================
from budding_yeast_v2.data.lhs_v1_origin import (
    simulate_mutant,
    get_default_params,
    generate_lhs_tasks,
    generate_local_lhs_tasks,
    mutant_rules,
    eqns as _yeast_ode_eqns,
    SIM_TIMEOUT,
)

# ==============================================================
# al4pde 候选池生成: 使用 v5 采样算法 (普通采样 + density + boundary)
# ==============================================================
from budding_yeast_v2.data.lhs_v1_origin_v5 import (
    generate_lhs_tasks as generate_lhs_tasks_v5,
    generate_targeted_lhs_tasks as generate_targeted_lhs_tasks_density,
    initial_conds as initial_conds_v5,
)
# v5_boundary 的 boundary 采样: 连续参数范围统一 [0, center_val*(1+pert)], 强制 0 值邻域覆盖
from budding_yeast_v2.data.lhs_v1_origin_v5_boundary import (
    generate_targeted_lhs_tasks as generate_targeted_lhs_tasks_boundary,
)

# ==============================================================
# 2. 严格对齐原版文件 (第 300-310 行) 的参数提取与排序逻辑
# ==============================================================
_defaults = get_default_params()
_defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
# 保持与原版 np.savez 时完全一致的特征顺序
PARAM_KEYS = sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])

VARIABLE_NAMES = [
    "MASS", "CLN2", "CLB2", "CLB5", "SIC1", "CDC6", "C2", "C5", "F2", "F5",
    "SIC1P", "C2P", "C5P", "CDC6P", "F2P", "F5P", "SWI5T", "SWI5", "IEP", "CDC20T",
    "CDC20", "CDH1T", "CDH1", "CDC14T", "CDC14", "NET1T", "NET1", "RENT", "TEM1", "CDC15",
    "PPX", "PDS1", "ESP1", "ORI", "BUD", "SPN", "Vi20", "lte1", "BUB2"
]

# ==================================================================
# MASS 生物学先验 (源自 data/mass_dataset_generator.py 的"通路→机制→严重程度"分层采样框架)
# 用于 al4pde_mass 策略: 按"通路频率 × 机制频率"给候选样本打生物学优先级。
#   - 通路频率: 参数所属绑定组对应通路的 high/medium/low 频率 (6/3/1), 同参数多组时取最高频通路
#   - 机制频率: 由候选参数相对 WT 默认值的扰动模式推断粗机制类 (loss/gain/partial),
#     权重取自 Chen 2004 操作实例统计 (KO=50, GAL-OE=23, ts=17, MC=12, db=10)
# 详见 mass_dataset_generator.BINDING_GROUPS / PATHWAYS 的组→通路→频率映射。
# ==================================================================
_MASS_PARAM_PATHWAY_FREQ = {
    # high (6): G1 Cdh1 / G2 Cln2 / G3 Sic1 / G4 Cdc6 / G5 Cdc20 / G6 Clb5
    'init_CDH1T': 6.0, 'init_CDH1': 6.0, 'kscdh': 6.0,
    'ksn2p': 6.0, 'ksn2pp': 6.0,
    'ksc1p': 6.0, 'ksc1pp': 6.0,
    'ksf6p': 6.0, 'ksf6pp': 6.0, 'ksf6ppp': 6.0,
    'ks20p': 6.0, 'ks20pp': 6.0,
    'ksb5p': 6.0, 'ksb5pp': 6.0,
    # medium (3): G7/G11 Clb2 / G8/G12 Net1-Cdc14 / G9 Pds1 / G13 Tem1
    'kdb2P': 3.0, 'kdb2pp': 3.0,
    'ksb2p': 3.0, 'ksb2pp': 3.0,
    'kasrent': 3.0, 'kasrentp': 3.0,
    'ks14': 3.0, 'kppc1': 3.0, 'kppf6': 3.0,
    'ks1pdspp': 3.0, 'ks2pdspp': 3.0, 'kd2pdspp': 3.0, 'kd3pdspp': 3.0,
    'ka15pp': 3.0, 'Tem1T': 3.0,
    # low (1): G10 Swi5 / G14 Cdc15 / G15 Bck2(MC) / G16-G20 检查点
    'ksswip': 1.0, 'ksswipp': 1.0,
    'kkpnetpp': 1.0, 'Cdc15T': 1.0,
    'B0': 1.0,
    'ka20pp': 1.0, 'Vi20_active': 1.0, 'init_BUB2': 1.0, 'BUB2_active': 1.0,
}
# 机制频率权重 (Chen 2004 操作实例统计, 归一化前的原始权重)
_MASS_MECH_WEIGHTS = {'KO': 50.0, 'GAL-OE': 23.0, 'ts': 17.0, 'MC': 12.0, 'db': 10.0}


def solve_single_mutant_bridge(name, p_vec):
    """将 AL 选拔出的 141 维张量完美还原为 simulate_mutant 所需的字典"""
    if isinstance(p_vec, torch.Tensor):
        p_vec = p_vec.cpu().numpy()
        
    param_overrides = {k: v for k, v in zip(PARAM_KEYS, p_vec)}
    
    try:
        m_name, y_uniform, bio_label, pattern_label = simulate_mutant(name, param_overrides)
    except Exception as e:
        return None
    
    if y_uniform is None:
        return None
        
    X_dummy = np.zeros((39, 2, 500))
    return (X_dummy, y_uniform, p_vec, pattern_label)


class ActiveLearningCallback(pl.Callback):
    def __init__(self, trigger_every_n_epochs=10, strategy='is', num_add=5000,
                 perturbation=0.1, mae_threshold=0.1,
                 # ====== 标准主动学习框架参数 ======
                 initial_train_size=5000,
                 initial_selection='kmeans',
                 diversity_weight=0.5,
                 pool_subset_size=10000,
                 uncertainty_metric='variance',
                 random_seed=42,
                 # ====== mutant_aware 策略专属参数 ======
                 atlas_csv_path=None,
                 # ====== al4pde 框架专属参数 ======
                 al_pool_dataset=None,
                 target_weight=0.4,
                 mass_weight=0.5):
        """
        标准主动学习 (Active Learning) 回调模块。

        本模块同时支持两种 AL 范式：
        1. **标准池化 AL (Pool-Based)**：从已有 LHS 数据池中选取样本，适用于 'us' / 'is' 策略。
           - on_train_start 时通过 K-Means 聚类从全量 LHS 训练集中选出初始训练子集，
             剩余样本构成未标注池 (unlabeled pool)。
           - 每轮 AL 触发时，依据查询策略从池中选取最具信息量的样本加入训练集。
        2. **查询合成 AL (Query Synthesis)**：通过 LHS 引擎生成新候选样本，适用于 'rgs' / 'mutant_aware' 策略。

        Args:
            trigger_every_n_epochs: 每隔多少 epoch 触发一次 AL 采样
            strategy: 采样策略
                - 'random': 随机采样 (池化)
                - 'us': 不确定性采样 Uncertainty Sampling (标准池化 AL)
                - 'is': 重要性采样 Importance Sampling = 不确定性 × 多样性 (标准池化 AL)
                - 'wrs': 加权蓄水池采样
                - 'vessal': VeSSAL (不确定性+多样性)
                - 'hggs': HGGS 流形采样
                - 'rgs': Real-Guided Sampling (局部密集采样, 查询合成)
                - 'mutant_aware': Mutant-Aware AL (基于 atlas 数据的优先级加权采样, 查询合成)
                    利用 lhs_MAE_min 和 lhs_n_samples 历史 atlas 数据计算每个 Real Mutant 的
                    优先级，聚焦采样到 LHS 覆盖不足 + 模型表现差的突变体邻域。
                    需要 atlas_csv_path 参数指向 mutant_atlas_merged.csv 文件。
                    若 atlas 不可用，自动降级为 'rgs' 策略。
            num_add: 每轮 AL 要新增的样本数 (默认 5000)
            perturbation: RGS/mutant_aware 专用，连续参数的相对扰动幅度 (0.1 = ±10%)
            mae_threshold: RGS 专用，MAE 大于此阈值的 Real Mutant 优先密集采样

            initial_train_size: 标准AL初始训练集大小 (默认 5000，约占 50000+ 样本的 10%)
            initial_selection: 初始训练集选择方法
                - 'kmeans': K-Means 聚类选取最具多样性的样本 (推荐)
                - 'random': 随机选取
            diversity_weight: Importance Sampling 中多样性得分权重 (默认 0.5, 范围 [0, 1])
            pool_subset_size: 每轮 AL 从池中采样的子集大小 (默认 10000, 平衡效率与覆盖)
            uncertainty_metric: 不确定性度量方法
                - 'variance': 预测方差 (沿时间维度)
                - 'entropy': 预测熵
            random_seed: 随机种子，保证 AL 选择可复现
            atlas_csv_path: mutant_aware 策略专用，mutant_atlas_merged.csv 路径
                (含 lhs_MAE_min, lhs_n_samples 列；若 None 或加载失败，降级为 'rgs')
        """
        self.trigger_every_n_epochs = trigger_every_n_epochs
        self.strategy = strategy.lower()
        self.num_add = num_add
        self.perturbation = perturbation
        self.mae_threshold = mae_threshold
        # RGS 状态：缓存每轮 Real Mutant 的 MAE
        self._real_mutant_scores = {}

        # ====== 标准主动学习框架状态 ======
        self.initial_train_size = initial_train_size
        self.initial_selection = initial_selection
        self.diversity_weight = diversity_weight
        self.pool_subset_size = pool_subset_size
        self.uncertainty_metric = uncertainty_metric
        self.random_seed = random_seed

        # AL 池管理状态 (在 on_train_start 中初始化)
        self._al_initialized = False
        self._pool_X = None        # 未标注池的输入特征 (IC + time grid)
        self._pool_P = None        # 未标注池的参数向量 (归一化)
        self._pool_Y = None        # 未标注池的标签 (归一化, 模拟"标注"过程)
        self._pool_size = 0        # 当前池中剩余样本数
        self._train_size_history = []  # 训练集大小历史记录

        # ====== mutant_aware 策略状态 ======
        self.atlas_csv_path = atlas_csv_path
        self._mutant_atlas = None        # {mutant_name: {'lhs_mae_min', 'lhs_n_samples', 'pattern', 'lhs_mae_median'}}
        self._atlas_loaded = False       # atlas 是否成功加载
        self._global_lhs_budget = 0      # 全局 LHS 预算 (与 _rgs_sample 共享)

        # ====== al4pde 框架状态 ======
        self.al_pool_dataset = al_pool_dataset
        self.target_weight = target_weight
        # al4pde_mass 策略: MASS 生物学分层优先级权重 (diversity 权重 = 1 - mass_weight)
        self.mass_weight = mass_weight
        # diversity_weight 复用已有的 self.diversity_weight (IS 策略参数)
        # MAAL: acquisition = diversity_weight * diversity + target_weight * priority
        # 约束: diversity_weight + target_weight = 1 (在 _maal_sample 中自动归一化)
        self._al4pde_pool_loaded = False
        self._pool_mutant_names = None   # 候选池样本的 mutant name (用于 MAAL family 匹配)
        self._al4pde_t_grid = None       # al4pde 候选池的 t_grid (内存优化: 不预存 X, 动态构造)
        # 内存优化: 候选池只存 P_norm + Y_IC (Y[:, :, 0]), 完整 Y 按需从磁盘加载
        self._pool_Y_ic = None           # 候选池样本的 IC (归一化), shape (N, 39)
        self._al4pde_pool_idx = None     # 候选池样本在原始 npz 中的索引 (用于按需加载完整 Y)
        self._al4pde_pool_path = None    # 候选池 npz 文件路径 (用于按需加载)
        self._al4pde_mean_y = None       # 归一化统计量 (用于仿真后归一化 Y)
        self._al4pde_std_y = None
        self._pool_P_real = None        # 未归一化参数 (用于选中后 ODE 仿真)
        self._pool_shape_features_cached = None  # ODE 预计算的形状特征缓存 (diversity/maal 用)

        # ====== al4pde 频率/数量机制 (v2.2 P2/P3/P4 改进) ======
        # al4pde 原始 (run_active_learning.py + data_schedule.py):
        #   - num_al_iter=5, 每轮训练 num_epochs=500 ep, 末轮 is_last 不选数据
        #   - 原始 ExponentialSchedule: n_samples(al_iter) = first_batch_num * 2^al_iter * batch_size
        # v2.1: initial=15000, trigger=4ep, num_add=1000, 51 轮 (增量式)
        # v2.2 (已作废): initial=5000, 分段 trigger 8/4/8, num_add 递增 1000/3000/4000, 29 轮
        # v2.3 (外层重训式, 方案 A, 向原版 al4pde 靠拢):
        #   - 5 轮 AL, 每轮全新模型 + 全新余弦周期跑满 (100 epoch), 取最后一轮最后 epoch
        #   - 初始 5000 (沿用 v2.2 基线), 中间 4 次采集, 每次追加 19000, 总追加 76000 → 总 81000
        #   - 选样与训练解耦: 由 train.py 外层循环调用 al4pde_select_and_append,
        #     不再走 on_train_epoch_end (外层重训模式下该钩子对 al4pde 直接跳过)

        if self.strategy in ('al4pde_random', 'al4pde_uncertainty', 'al4pde_diversity',
                             'maal', 'al4pde_boundary', 'al4pde_mass'):
            print(f"[AL Manager] Override al4pde settings (v2.3 外层重训式): "
                  f"initial_train_size=5000, 5 轮, 每次采集追加 19000 (总 76000)")
            self.initial_train_size = 5000
            self.trigger_every_n_epochs = 4  # 保留字段, 外层重训模式下不生效
            self.num_add = 1000              # 保留字段, 外层重训模式下不生效
            self._al4pde_num_al_iter = 5        # v2.3: 5 轮 AL (前 4 轮采集, 末轮只训练)
            self._al4pde_per_round_add = 19000  # v2.3: 每次采集追加数 (76000 / 4)
        else:
            self._al4pde_num_al_iter = 6
            self._al4pde_per_round_add = 1000
        
        self._al4pde_al_iter = 0           # 当前 AL 轮次 (0-indexed, 每次选择后递增)
        self._outer_loop_mode = False      # v2.3: 外层重训模式 (由 train.py 启用, 关闭 in-loop 触发)

    # ==================================================================
    # 标准主动学习框架：初始训练集选择
    # ==================================================================
    def on_train_start(self, trainer, pl_module):
        """
        标准主动学习初始化（所有策略统一执行）：
        - 池化策略 (us/is/random/wrs/vessal/hggs)：通过 K-Means 聚类选出初始训练子集,
          剩余样本构成未标注池 (unlabeled pool), 后续从池中查询
        - 查询合成策略 (rgs/mutant_aware)：直接用全量数据开始训练, 无需池化
          (查询合成 AL 通过 LHS 引擎生成新样本, 不需要从池中查询;
           先在全量数据上预训练收敛, 再触发 AL 做定向采样"锦上添花")

        数据划分合规性：
        - 初始训练集和未标注池均仅包含 LHS 样本 (来自 dm.train_dataset_lhs)
        - 验证集和测试集不受影响，测试集仍同时包含 LHS 和 Real Mutants
        """
        # 所有 AL 策略都执行初始化 (none 已在外层过滤)
        if self.strategy == 'none':
            return

        # 仅在 global zero 进程执行初始化
        if not trainer.is_global_zero:
            return

        dm = trainer.datamodule
        if dm is None or not hasattr(dm, 'train_dataset_lhs'):
            print("[AL Manager] 警告: 未找到 train_dataset_lhs，跳过标准 AL 初始化")
            return

        # 获取全量 LHS 训练集
        full_X, full_P, full_Y = dm.train_dataset_lhs.tensors
        full_size = full_X.shape[0]

        print(f"\n[AL Manager] ===== 标准主动学习初始化 =====")
        print(f"[AL Manager] 策略: {self.strategy.upper()} | 全量 LHS 训练集: {full_size}")

        # ====== 查询合成策略 (rgs/mutant_aware): 跳过 K-Means, 直接用全量数据 ======
        # 设计理由:
        #   1. 查询合成 AL 通过 LHS 引擎生成新样本, 不需要从池中查询, 池化无意义
        #   2. 全量数据预训练让模型先收敛, 再触发 AL 定向采样, 避免"雪中送炭"式欠训练
        #   3. K-Means 在大数据集 (120000+ 样本 × 141 维) 上做 50000 簇聚类耗时数小时, 不可行
        #   4. 池化策略仍走 K-Means 初始化, 让 AL 有"发挥空间"
        if self.strategy in ('rgs', 'mutant_aware'):
            mode_desc = ("查询合成模式 (RGS 通过 LHS 引擎生成新样本)" if self.strategy == 'rgs'
                         else "查询合成模式 (Mutant-Aware 基于 atlas 优先级加权采样)")
            print(f"[AL Manager] {mode_desc}")
            print(f"[AL Manager] 跳过 K-Means 初始化, 直接用全量 {full_size} 样本开始训练")
            print(f"[AL Manager] AL 触发时机: epoch {self.trigger_every_n_epochs} 后开始定向采样")
            print(f"[AL Manager] 验证集/测试集保持不变 (仅含 LHS / LHS+Real)")
            print(f"[AL Manager] ===== 初始化完成 =====\n")

            self._al_initialized = True
            self._train_size_history.append(full_size)
            # 查询合成策略不需要未标注池
            self._pool_X = None
            self._pool_P = None
            self._pool_Y = None
            self._pool_size = 0

            # ====== mutant_aware 策略：加载 atlas 数据 ======
            if self.strategy == 'mutant_aware':
                self._load_mutant_atlas()
            return

        # ====== 池化策略 (us/is/random/wrs/vessal/hggs) + al4pde 框架: K-Means 初始化 ======
        initial_size = min(self.initial_train_size, full_size)
        print(f"[AL Manager] 初始训练集目标大小: {initial_size} | 选择方法: {self.initial_selection}")

        # 选择初始训练集索引
        if self.initial_selection == 'kmeans':
            selected_idx = self._kmeans_initial_selection(full_P, initial_size)
        else:
            # 随机选择
            rng = np.random.RandomState(self.random_seed)
            selected_idx = rng.choice(full_size, initial_size, replace=False)

        # 构建初始训练集和未标注池
        mask = np.ones(full_size, dtype=bool)
        mask[selected_idx] = False
        pool_idx = np.where(mask)[0]

        # 更新训练集为初始子集
        from torch.utils.data import TensorDataset
        dm.train_dataset_lhs = TensorDataset(
            full_X[selected_idx],
            full_P[selected_idx],
            full_Y[selected_idx]
        )
        # 同步更新 _train_idx_lhs: AL 替换了 train_dataset_lhs 为 selected_idx 子集,
        # _train_idx_lhs 也需对应缩小, 否则后续 _map_family_priority_train_based 等
        # 函数用 _train_idx_lhs 取名字时大小不匹配 (42661 vs 15000) 导致 IndexError
        if hasattr(dm, '_train_idx_lhs'):
            dm._train_idx_lhs = np.asarray(dm._train_idx_lhs)[selected_idx]

        # ====== al4pde 框架: 生成参数候选池 (v5 采样算法, 只含参数不含轨迹) ======
        if self.strategy in ('al4pde_random', 'al4pde_uncertainty', 'al4pde_diversity',
                             'maal', 'al4pde_boundary', 'al4pde_mass'):
            self._generate_param_only_pool(dm)
            # diversity/maal/boundary 需要轨迹形状特征: 预计算 ODE 仿真并缓存
            if self.strategy in ('al4pde_diversity', 'maal', 'al4pde_boundary', 'al4pde_mass'):
                self._precompute_pool_ode_shape_features(dm)
        else:
            # 标准池化策略: 保存未标注池 (v1 剩余样本)
            self._pool_X = full_X[pool_idx]
            self._pool_P = full_P[pool_idx]
            self._pool_Y = full_Y[pool_idx]
            self._pool_size = self._pool_X.shape[0]

        self._al_initialized = True
        self._train_size_history.append(initial_size)

        print(f"[AL Manager] 初始训练集: {initial_size} 样本 (K-Means={self.initial_selection=='kmeans'})")
        print(f"[AL Manager] 未标注池: {self._pool_size} 样本 | 策略: {self.strategy}")
        print(f"[AL Manager] 验证集/测试集保持不变 (仅含 LHS / LHS+Real)")
        print(f"[AL Manager] ===== 初始化完成 =====\n")

    def _kmeans_initial_selection(self, full_P, initial_size):
        """
        K-Means 聚类初始训练集选择：
        - 在参数空间 (141维) 上做 K-Means 聚类，簇数 = initial_size
        - 从每个簇中选取离簇心最近的样本，确保初始训练集覆盖整个参数空间

        Args:
            full_P: (N, 141) 归一化参数张量
            initial_size: 初始训练集大小

        Returns:
            selected_idx: 选中的样本索引数组
        """
        P_np = full_P.cpu().numpy()
        n_clusters = min(initial_size, P_np.shape[0])

        print(f"[AL Manager] K-Means 聚类: {n_clusters} 簇, {P_np.shape[0]} 样本...")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_seed,
            n_init=10,
            max_iter=300
        )
        cluster_labels = kmeans.fit_predict(P_np)
        centers = kmeans.cluster_centers_

        # 从每个簇中选取离簇心最近的样本
        selected_idx = []
        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                continue
            # 计算簇内样本到簇心的距离
            distances = np.linalg.norm(P_np[cluster_indices] - centers[cluster_id], axis=1)
            nearest_local_idx = cluster_indices[np.argmin(distances)]
            selected_idx.append(nearest_local_idx)

        selected_idx = np.array(selected_idx)
        print(f"[AL Manager] K-Means 完成: 选中 {len(selected_idx)} 个多样性样本")
        return selected_idx

    # ==================================================================
    # v2.3 外层重训式 al4pde: 供 train.py 外层循环调用的公开接口
    # ==================================================================
    def enable_outer_loop_mode(self):
        """v2.3: 启用外层重训模式 (选样由 train.py 外层循环驱动, on_train_epoch_end 不再触发)。"""
        self._outer_loop_mode = True

    def al4pde_setup(self, dm, device='cuda'):
        """v2.3 外层重训式: 一次性初始化 (缩至初始子集 + 生成候选池 + ODE 预计算)。
        供 train.py 在外层循环开始前调用一次 (等价于原 on_train_start 的 al4pde 分支)。"""
        from torch.utils.data import TensorDataset
        full_X, full_P, full_Y = dm.train_dataset_lhs.tensors
        full_size = full_X.shape[0]
        initial_size = min(self.initial_train_size, full_size)
        print(f"[al4pde v2.3] 初始训练集: {initial_size}/{full_size} (选择方法={self.initial_selection})")
        if self.initial_selection == 'kmeans':
            selected_idx = self._kmeans_initial_selection(full_P, initial_size)
        else:
            rng = np.random.RandomState(self.random_seed)
            selected_idx = rng.choice(full_size, initial_size, replace=False)
        dm.train_dataset_lhs = TensorDataset(full_X[selected_idx], full_P[selected_idx], full_Y[selected_idx])
        if hasattr(dm, '_train_idx_lhs'):
            dm._train_idx_lhs = np.asarray(dm._train_idx_lhs)[selected_idx]
        # 生成参数候选池 (v5 采样); diversity/maal/boundary 额外做 ODE 形状特征预计算
        self._generate_param_only_pool(dm)
        if self.strategy in ('al4pde_diversity', 'maal', 'al4pde_boundary', 'al4pde_mass'):
            self._precompute_pool_ode_shape_features(dm)
        self._al_initialized = True
        self._al4pde_al_iter = 0
        self._train_size_history.append(len(dm.train_dataset_lhs))

    def al4pde_select_and_append(self, pl_module, dm, device='cuda'):
        """v2.3 外层重训式: 用当前(上一轮训练完的)模型选样并追加到 train_dataset_lhs。
        返回本次追加的样本数。"""
        from torch.utils.data import TensorDataset
        if pl_module is not None:
            pl_module = pl_module.to(device)
            pl_module.eval()
        if self.strategy == 'al4pde_random':
            bd = self._al4pde_random_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_uncertainty':
            bd = self._al4pde_uncertainty_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_diversity':
            bd = self._al4pde_diversity_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_boundary':
            bd = self._al4pde_boundary_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'maal':
            bd = self._maal_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_mass':
            bd = self._al4pde_mass_sample(None, pl_module, dm, device, 0)
        else:
            bd = None
        if bd is None:
            print("[al4pde v2.3] 本轮候选池为空, 未追加样本")
            if pl_module is not None:
                pl_module.train()
            return 0
        old_X, old_P, old_Y = dm.train_dataset_lhs.tensors
        dm.train_dataset_lhs = TensorDataset(
            torch.cat([old_X, bd['X'].to(old_X.device)], dim=0),
            torch.cat([old_P, bd['P'].to(old_X.device)], dim=0),
            torch.cat([old_Y, bd['Y'].to(old_Y.device)], dim=0),
        )
        if 'names' in bd and hasattr(dm, '_mutant_names'):
            new_names = list(bd['names'])
            n_existing = len(dm._mutant_names)
            dm._mutant_names = list(dm._mutant_names) + new_names
            new_indices = np.arange(n_existing, n_existing + len(new_names))
            if hasattr(dm, '_train_idx_lhs'):
                dm._train_idx_lhs = np.concatenate([np.asarray(dm._train_idx_lhs), new_indices])
            else:
                dm._train_idx_lhs = new_indices
        self._al4pde_al_iter += 1
        self._train_size_history.append(len(dm.train_dataset_lhs))
        n_added = int(bd['X'].shape[0])
        print(f"[al4pde v2.3] 追加 {n_added} 个样本 → 训练集 {len(dm.train_dataset_lhs)} "
              f"(al_iter={self._al4pde_al_iter})")
        if pl_module is not None:
            pl_module.train()
        return n_added

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        # v2.3: al4pde 外层重训模式下, 选样由 train.py 外层循环负责, 关闭 in-loop 触发
        if getattr(self, '_outer_loop_mode', False) and self._is_al4pde_strategy():
            return
        # v2.2 P3: al4pde 分段式触发频率 (8/4/8 ep), 适配训练阶段
        #   早期 (ep 1-30): 每 8 ep (模型未稳定, 稀疏触发)
        #   中期 (ep 30-120): 每 4 ep (主力阶段, 密集触发)
        #   后期 (ep 120-150): 每 8 ep (过拟合期, 减少干扰)
        if self._is_al4pde_strategy():
            if current_epoch <= 30:
                trigger_interval = 8
            elif current_epoch <= 120:
                trigger_interval = 4
            else:
                trigger_interval = 8
        else:
            trigger_interval = self.trigger_every_n_epochs

        if current_epoch == 0 or current_epoch % trigger_interval != 0:
            return

        # al4pde 框架: 最后一轮 AL 不选数据 (对齐 al4pde run_active_learning.py is_last 逻辑)
        # al4pde 原始: for al_iter in range(num_al_iter): train → if not is_last: select
        # 5 轮 AL 中前 4 轮选数据 (al_iter 0,1,2,3), 末轮 (al_iter 4) 只训练不选
        if self._is_al4pde_strategy() and self._al4pde_al_iter >= self._al4pde_num_al_iter - 1:
            if self._al4pde_al_iter == self._al4pde_num_al_iter - 1:
                print(f"[AL Manager] Epoch {current_epoch}: al4pde 末轮 (al_iter={self._al4pde_al_iter}), "
                      f"对齐 al4pde is_last 逻辑, 跳过采样")
            return

        print(f"\n[AL Manager] Epoch {current_epoch} 触发主动学习采样: {self.strategy.upper()}")
        if self._is_al4pde_strategy():
            scheduled_add = self._al4pde_scheduled_num_add()
            print(f"[AL Manager] al4pde Data Schedule: al_iter={self._al4pde_al_iter}, "
                  f"本轮新增 {scheduled_add} 个样本")
        dm = trainer.datamodule
        device = pl_module.device
        broadcast_data = None

        if trainer.is_global_zero:
            # 查询合成策略 (RGS)：通过 LHS 引擎生成新样本，但仍走标准 AL 框架
            if self.strategy == 'rgs':
                broadcast_data = self._rgs_sample(trainer, pl_module, dm, device, current_epoch)
            # 查询合成策略 (Mutant-Aware)：基于 atlas 优先级加权采样
            elif self.strategy == 'mutant_aware':
                broadcast_data = self._mutant_aware_sample(trainer, pl_module, dm, device, current_epoch)
            # ====== al4pde 框架策略 (从 v5 候选池查询) ======
            elif self.strategy == 'al4pde_random':
                broadcast_data = self._al4pde_random_sample(trainer, pl_module, dm, device, current_epoch)
            elif self.strategy == 'al4pde_uncertainty':
                broadcast_data = self._al4pde_uncertainty_sample(trainer, pl_module, dm, device, current_epoch)
            elif self.strategy == 'al4pde_diversity':
                broadcast_data = self._al4pde_diversity_sample(trainer, pl_module, dm, device, current_epoch)
            elif self.strategy == 'al4pde_boundary':
                broadcast_data = self._al4pde_boundary_sample(trainer, pl_module, dm, device, current_epoch)
            elif self.strategy == 'maal':
                broadcast_data = self._maal_sample(trainer, pl_module, dm, device, current_epoch)
            elif self.strategy == 'al4pde_mass':
                broadcast_data = self._al4pde_mass_sample(trainer, pl_module, dm, device, current_epoch)
            # 所有池化策略 (us/is/random/wrs/vessal/hggs)：统一走标准 AL 入口
            elif self._al_initialized:
                broadcast_data = self._standard_al_sample(trainer, pl_module, dm, device, current_epoch)
            # 兜底：未初始化时降级为旧逻辑
            else:
                broadcast_data = self._legacy_sample(trainer, pl_module, dm, device, current_epoch)

        if trainer.world_size > 1:
            broadcast_data = trainer.strategy.broadcast(broadcast_data, src=0)

        if broadcast_data is not None:
            from torch.utils.data import TensorDataset

            old_X, old_P, old_Y = dm.train_dataset_lhs.tensors

            updated_X = torch.cat([old_X, broadcast_data['X'].to(old_X.device)], dim=0)
            updated_P = torch.cat([old_P, broadcast_data['P'].to(old_X.device)], dim=0)
            updated_Y = torch.cat([old_Y, broadcast_data['Y'].to(old_Y.device)], dim=0)

            dm.train_dataset_lhs = TensorDataset(updated_X, updated_P, updated_Y)
            self._train_size_history.append(len(dm.train_dataset_lhs))
            print(f"[AL Manager] 训练集成功扩容！当前新样本总数: {len(dm.train_dataset_lhs)}")

            # 同步更新 _mutant_names 和 _train_idx_lhs: 新增样本来自候选池 (不在原全量数据集中),
            # 需追加名字和索引, 否则后续 AL 迭代中 _map_family_priority_train_based 等
            # 函数用 _train_idx_lhs 取名字时大小不匹配导致 IndexError
            if 'names' in broadcast_data and hasattr(dm, '_mutant_names'):
                new_names = list(broadcast_data['names'])
                n_existing = len(dm._mutant_names)
                dm._mutant_names = list(dm._mutant_names) + new_names
                new_indices = np.arange(n_existing, n_existing + len(new_names))
                if hasattr(dm, '_train_idx_lhs'):
                    dm._train_idx_lhs = np.concatenate([np.asarray(dm._train_idx_lhs), new_indices])
                else:
                    dm._train_idx_lhs = new_indices

            # al4pde: 递增 AL 轮次计数器 (驱动 ExponentialSchedule 数量增长)
            if self._is_al4pde_strategy():
                self._al4pde_al_iter += 1
                print(f"[AL Manager] al4pde AL 轮次递增 → al_iter={self._al4pde_al_iter} "
                      f"(共 {self._al4pde_num_al_iter} 轮, 末轮不选)")

            # 上传 AL 采样对比图到 wandb
            if 'sample_plots' in broadcast_data and trainer.logger and hasattr(trainer.logger, 'experiment'):
                import wandb
                log_dict = {'epoch': current_epoch}
                for plot_key, fig in broadcast_data['sample_plots'].items():
                    log_dict[f'AL_Samples/{plot_key}'] = wandb.Image(fig)
                    plt.close(fig)
                trainer.logger.experiment.log(log_dict)
                print(f"[AL Manager] 已上传 {len(broadcast_data['sample_plots'])} 张采样对比图到 wandb")

    # ==================================================================
    # 标准主动学习框架：池化 AL 核心逻辑
    # ==================================================================
    def _standard_al_sample(self, trainer, pl_module, dm, device, current_epoch):
        """
        标准池化主动学习采样入口（统一处理所有池化策略）：
        1. 从未标注池中采样子集 (pool_subset_size) 以平衡效率
        2. 用模型对子集进行推理，计算不确定性得分
        3. 根据查询策略 (US/IS/WRS/VeSSAL/HGGS/Random) 选取 num_add 个样本
        4. 将选中样本从池中迁移到训练集 ("标注"过程)
        5. 更新池状态

        支持的策略：
        - 'random': 随机采样
        - 'us': Uncertainty Sampling (top-K 不确定性)
        - 'is': Importance Sampling (不确定性 × 多样性)
        - 'wrs': Weighted Reservoir Sampling (不确定性加权随机)
        - 'vessal': VeSSAL (不确定性 + 多样性，动态权重)
        - 'hggs': HGGS (基于 K-Means 聚类的分层多样性采样)

        Returns:
            broadcast_data: 包含选中样本的字典，或 None (池为空时)
        """
        # 检查池是否为空
        if self._pool_size == 0:
            print("[AL Manager] 未标注池已耗尽，跳过本轮 AL 采样")
            return None

        num_to_select = min(self.num_add, self._pool_size)
        print(f"[AL Manager] 池剩余: {self._pool_size} | 本轮选取: {num_to_select} | 策略: {self.strategy.upper()}")

        # Step 1: 从池中采样子集 (提高大规模池的效率)
        pool_indices = self._subsample_pool()

        # Step 2: 计算不确定性得分 (random/hggs 策略不需要)
        if self.strategy in ['us', 'is', 'wrs', 'vessal']:
            uncertainty_scores = self._compute_uncertainty(pl_module, device, pool_indices)
        else:
            uncertainty_scores = None  # random / hggs / pial 策略不需要传统不确定性

        # Step 3: 根据策略选取样本
        if self.strategy == 'us':
            selected_local = self._query_uncertainty_sampling(uncertainty_scores, num_to_select)
        elif self.strategy == 'is':
            selected_local = self._query_importance_sampling(
                uncertainty_scores, pool_indices, dm, num_to_select)
        elif self.strategy == 'wrs':
            selected_local = self._query_weighted_reservoir_sampling(
                uncertainty_scores, num_to_select, current_epoch)
        elif self.strategy == 'vessal':
            selected_local = self._query_vessal(
                uncertainty_scores, pool_indices, dm, num_to_select, current_epoch)
        elif self.strategy == 'hggs':
            selected_local = self._query_hggs(pool_indices, dm, num_to_select)
        elif self.strategy == 'pial':
            # Physics-Informed Active Learning: ODE 残差 + 轨迹多样性
            selected_local = self._query_physics_informed(
                pl_module, device, pool_indices, dm, num_to_select, current_epoch)
        else:  # random
            rng = np.random.RandomState(self.random_seed + current_epoch)
            selected_local = rng.choice(len(pool_indices), num_to_select, replace=False)

        # 将局部索引映射回池的全局索引
        selected_pool_idx = pool_indices[selected_local]

        # Step 4: 从池中提取选中样本并迁移到训练集
        broadcast_data = self._move_samples_to_training_set(selected_pool_idx, current_epoch)

        return broadcast_data

    def _subsample_pool(self):
        """
        从未标注池中采样子集，用于后续不确定性计算。
        当池大小超过 pool_subset_size 时随机采样，否则使用全池。

        Returns:
            pool_indices: 池中用于本轮 AL 评估的样本索引数组
        """
        if self._pool_size <= self.pool_subset_size:
            return np.arange(self._pool_size)

        rng = np.random.RandomState(self.random_seed + len(self._train_size_history))
        pool_indices = rng.choice(self._pool_size, self.pool_subset_size, replace=False)
        return np.sort(pool_indices)

    def _compute_uncertainty(self, pl_module, device, pool_indices):
        """
        计算池子集中每个样本的不确定性得分。

        不确定性度量：
        - 'variance': 模型预测沿时间维度的方差均值 (越高越不确定)
        - 'entropy': 预测分布的熵

        Args:
            pl_module: Lightning 模块
            device: 计算设备
            pool_indices: 池中样本索引数组

        Returns:
            uncertainty_scores: (N,) 不确定性得分数组
        """
        pl_module.eval()
        batch_size = 2000
        uncertainty_scores = []

        with torch.no_grad():
            for i in range(0, len(pool_indices), batch_size):
                end_idx = min(i + batch_size, len(pool_indices))
                batch_indices = pool_indices[i:end_idx]

                # 从池中获取批次数据
                X_batch = self._pool_X[batch_indices].to(device)
                P_batch = self._pool_P[batch_indices].to(device)

                # 模型预测: (B, 39, 500)
                preds = pl_module.forward_ic_time(X_batch, P_batch)

                if self.uncertainty_metric == 'entropy':
                    # 预测熵: 将预测视为概率分布，计算沿时间维度的熵
                    preds_abs = torch.abs(preds) + 1e-8
                    preds_norm = preds_abs / preds_abs.sum(dim=-1, keepdim=True)
                    entropy = -torch.sum(preds_norm * torch.log(preds_norm), dim=-1)
                    scores = entropy.mean(dim=1).cpu().numpy()
                else:
                    # 默认: 预测方差 (沿时间维度)
                    variance = torch.var(preds, dim=-1)  # (B, 39)
                    scores = variance.mean(dim=1).cpu().numpy()  # (B,)

                uncertainty_scores.append(scores)

        pl_module.train()
        return np.concatenate(uncertainty_scores)

    def _query_uncertainty_sampling(self, uncertainty_scores, num_to_select):
        """
        标准 Uncertainty Sampling 查询策略：
        选取不确定性得分最高的 num_to_select 个样本。

        Args:
            uncertainty_scores: (N,) 不确定性得分数组
            num_to_select: 要选取的样本数

        Returns:
            selected_idx: 选中样本在子集中的局部索引
        """
        num_to_select = min(num_to_select, len(uncertainty_scores))
        # 按不确定性降序排列，选取 top-K
        selected_idx = np.argsort(uncertainty_scores)[-num_to_select:]

        print(f"[US] 不确定性范围: [{uncertainty_scores.min():.6f}, {uncertainty_scores.max():.6f}] | "
              f"选中 top-{num_to_select} (阈值: {uncertainty_scores[selected_idx].min():.6f})")

        return selected_idx

    def _query_importance_sampling(self, uncertainty_scores, pool_indices, dm, num_to_select):
        """
        标准 Importance Sampling 查询策略：
        综合不确定性和多样性选取样本。

        得分 = (1 - diversity_weight) * uncertainty_norm + diversity_weight * diversity_norm

        多样性度量：样本到当前训练集的最小欧氏距离 (越远越多样)

        Args:
            uncertainty_scores: (N,) 不确定性得分
            pool_indices: 池中样本索引
            dm: 数据模块 (用于获取当前训练集)
            num_to_select: 要选取的样本数

        Returns:
            selected_idx: 选中样本在子集中的局部索引
        """
        # 归一化不确定性得分
        u_min, u_max = uncertainty_scores.min(), uncertainty_scores.max()
        if u_max - u_min > 1e-8:
            uncertainty_norm = (uncertainty_scores - u_min) / (u_max - u_min)
        else:
            uncertainty_norm = np.zeros_like(uncertainty_scores)

        # 计算多样性得分：样本到训练集的最小距离
        diversity_scores = self._compute_diversity(pool_indices, dm)
        d_min, d_max = diversity_scores.min(), diversity_scores.max()
        if d_max - d_min > 1e-8:
            diversity_norm = (diversity_scores - d_min) / (d_max - d_min)
        else:
            diversity_norm = np.zeros_like(diversity_scores)

        # 综合得分: 加权组合
        w = self.diversity_weight
        combined_scores = (1.0 - w) * uncertainty_norm + w * diversity_norm

        # 选取综合得分最高的样本
        num_to_select = min(num_to_select, len(combined_scores))
        selected_idx = np.argsort(combined_scores)[-num_to_select:]

        print(f"[IS] 不确定性: [{u_min:.6f}, {u_max:.6f}] | "
              f"多样性: [{d_min:.6f}, {d_max:.6f}] | "
              f"权重: uncertainty={1.0-w:.2f}, diversity={w:.2f} | "
              f"选中 top-{num_to_select}")

        return selected_idx

    def _query_weighted_reservoir_sampling(self, uncertainty_scores, num_to_select, current_epoch):
        """
        标准 Weighted Reservoir Sampling (WRS) 查询策略：
        基于不确定性得分作为权重，进行加权随机采样 (无放回)。

        与 US 的区别：US 选 top-K (贪心)，WRS 按权重概率随机抽取，
        既能偏向不确定样本，又能保留多样性 (避免全部聚焦在最难样本)。

        Args:
            uncertainty_scores: (N,) 不确定性得分
            num_to_select: 要选取的样本数
            current_epoch: 当前 epoch (用于随机种子)

        Returns:
            selected_idx: 选中样本在子集中的局部索引
        """
        num_to_select = min(num_to_select, len(uncertainty_scores))

        # 将不确定性转换为概率权重 (softmax 归一化，强化对比)
        u_min = uncertainty_scores.min()
        u_shifted = uncertainty_scores - u_min  # 平移到非负
        # 使用平方强化高不确定性样本的权重
        weights = u_shifted ** 2 + 1e-8
        probs = weights / weights.sum()

        # 加权随机采样 (无放回)
        rng = np.random.RandomState(self.random_seed + current_epoch)
        selected_idx = rng.choice(len(uncertainty_scores), num_to_select,
                                   p=probs, replace=False)

        print(f"[WRS] 不确定性: [{uncertainty_scores.min():.6f}, {uncertainty_scores.max():.6f}] | "
              f"加权随机选取 {num_to_select} 个 (max权重: {probs.max():.6f})")

        return selected_idx

    def _query_vessal(self, uncertainty_scores, pool_indices, dm, num_to_select, current_epoch):
        """
        标准 VeSSAL 查询策略：
        动态平衡不确定性和多样性，使用基于训练进度的动态权重。

        VeSSAL 特点：随着训练进行，逐渐增加多样性权重
        (早期侧重不确定性快速降低误差，后期侧重多样性避免过拟合)。

        得分 = (1 - w_dynamic) * uncertainty_norm + w_dynamic * diversity_norm
        w_dynamic = diversity_weight * (current_epoch / max_epochs_estimate)

        Args:
            uncertainty_scores: (N,) 不确定性得分
            pool_indices: 池中样本索引
            dm: 数据模块
            num_to_select: 要选取的样本数
            current_epoch: 当前 epoch

        Returns:
            selected_idx: 选中样本在子集中的局部索引
        """
        # 归一化不确定性
        u_min, u_max = uncertainty_scores.min(), uncertainty_scores.max()
        if u_max - u_min > 1e-8:
            uncertainty_norm = (uncertainty_scores - u_min) / (u_max - u_min)
        else:
            uncertainty_norm = np.zeros_like(uncertainty_scores)

        # 计算多样性
        diversity_scores = self._compute_diversity(pool_indices, dm)
        d_min, d_max = diversity_scores.min(), diversity_scores.max()
        if d_max - d_min > 1e-8:
            diversity_norm = (diversity_scores - d_min) / (d_max - d_min)
        else:
            diversity_norm = np.zeros_like(diversity_scores)

        # 动态权重：随训练进度增加多样性权重
        # 估计训练进度 (基于历史记录)
        progress = min(1.0, len(self._train_size_history) / 10.0)  # 假设 10 轮 AL 后达到最大多样性权重
        w_dynamic = self.diversity_weight * progress

        # 综合得分
        combined_scores = (1.0 - w_dynamic) * uncertainty_norm + w_dynamic * diversity_norm

        # 选取综合得分最高的样本
        num_to_select = min(num_to_select, len(combined_scores))
        selected_idx = np.argsort(combined_scores)[-num_to_select:]

        print(f"[VeSSAL] 不确定性: [{u_min:.6f}, {u_max:.6f}] | "
              f"多样性: [{d_min:.6f}, {d_max:.6f}] | "
              f"动态权重: uncertainty={1.0-w_dynamic:.2f}, diversity={w_dynamic:.2f} (进度={progress:.2f}) | "
              f"选中 top-{num_to_select}")

        return selected_idx

    def _query_hggs(self, pool_indices, dm, num_to_select):
        """
        标准 HGGS (Hierarchical Greedy Graph Sampling) 查询策略：
        基于 K-Means 聚类的分层多样性采样，纯多样性驱动 (不依赖模型预测)。

        算法：
        1. 在池子集参数空间上做 K-Means 聚类 (K = num_to_select)
        2. 从每个簇中选取离簇心最近的样本
        3. 保证选中样本均匀覆盖整个参数空间

        适用于：模型预测不可靠的早期阶段，或需要最大化空间覆盖的场景。

        Args:
            pool_indices: 池中样本索引
            dm: 数据模块
            num_to_select: 要选取的样本数

        Returns:
            selected_idx: 选中样本在子集中的局部索引
        """
        num_to_select = min(num_to_select, len(pool_indices))

        # 获取池子集的参数
        pool_P = self._pool_P[pool_indices].cpu().numpy()

        # K-Means 聚类：将池子集分为 num_to_select 个簇
        n_clusters = min(num_to_select, len(pool_P))
        print(f"[HGGS] K-Means 聚类: {n_clusters} 簇, {len(pool_P)} 样本...")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_seed + len(self._train_size_history),
            n_init=10,
            max_iter=300
        )
        cluster_labels = kmeans.fit_predict(pool_P)
        centers = kmeans.cluster_centers_

        # 从每个簇中选取离簇心最近的样本
        selected_local = []
        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_local_indices = np.where(cluster_mask)[0]
            if len(cluster_local_indices) == 0:
                continue
            # 计算簇内样本到簇心的距离
            distances = np.linalg.norm(pool_P[cluster_local_indices] - centers[cluster_id], axis=1)
            nearest_local = cluster_local_indices[np.argmin(distances)]
            selected_local.append(nearest_local)

        selected_local = np.array(selected_local)
        print(f"[HGGS] 分层多样性采样完成: 选中 {len(selected_local)} 个样本 "
              f"(覆盖 {n_clusters} 个参数空间簇)")

        return selected_local

    def _compute_diversity(self, pool_indices, dm):
        """
        计算池子集中每个样本的多样性得分。
        多样性 = 样本参数到当前训练集参数的最小欧氏距离。

        为提高效率，从训练集中随机采样代表点计算距离。

        Args:
            pool_indices: 池中样本索引
            dm: 数据模块

        Returns:
            diversity_scores: (N,) 多样性得分数组
        """
        # 获取池子集的参数
        pool_P = self._pool_P[pool_indices].cpu().numpy()

        # 获取当前训练集的参数 (采样代表点以提高效率)
        train_P = dm.train_dataset_lhs.tensors[1].cpu().numpy()
        n_representatives = min(2000, train_P.shape[0])
        rng = np.random.RandomState(self.random_seed)
        rep_idx = rng.choice(train_P.shape[0], n_representatives, replace=False)
        train_P_rep = train_P[rep_idx]

        # 计算每个池样本到训练集代表点的最小距离
        # 分批计算以避免内存溢出
        batch_size = 1000
        diversity_scores = []
        for i in range(0, pool_P.shape[0], batch_size):
            end_idx = min(i + batch_size, pool_P.shape[0])
            batch = pool_P[i:end_idx]
            # (batch_size, n_representatives) 距离矩阵
            distances = euclidean_distances(batch, train_P_rep)
            # 每个样本的多样性得分 = 到最近训练样本的距离
            min_distances = distances.min(axis=1)
            diversity_scores.append(min_distances)

        return np.concatenate(diversity_scores)

    # ==================================================================
    # Physics-Informed Active Learning (PIAL) v2
    # 重写: MC Dropout 不确定性主导 + PCA 轨迹多样性 + ODE 残差辅助
    # ==================================================================
    def _query_physics_informed(self, pl_module, device, pool_indices, dm,
                                 num_to_select, current_epoch):
        """
        Physics-Informed Active Learning v2 查询策略。

        三信号融合选取样本：
        1. **MC Dropout 不确定性** (主导, α=0.6→0.5)：对参数 p 加微小高斯噪声
           前向 K 次，计算预测方差。方差大 → 模型对该样本不确定 → 信息量大。
           **关键：不需要真实标签，不需要 ODE 方程，对所有模型通用。**
        2. **PCA 轨迹多样性** (次之, β=0.3→0.4)：将轨迹 (39,T) 用 PCA 降到
           20 维，在低维空间计算到训练集的最小欧氏距离。避免维度灾难。
        3. **ODE 残差** (辅助, γ=0.1)：模型预测代入 ODE 的残差，仅作弱辅助信号。

        综合 score = α · normalize(uncertainty) + β · normalize(diversity)
                     + γ · normalize(residual)

        动态权重：早期重不确定性（快速降误差），后期重多样性（避免过拟合），
                  残差始终作弱辅助。

        Args:
            pl_module: Lightning 模块
            device: 计算设备
            pool_indices: 池中样本索引
            dm: 数据模块
            num_to_select: 选取样本数
            current_epoch: 当前 epoch

        Returns:
            selected_idx: 选中样本在子集中的局部索引
        """
        print(f"\n[PIAL-v2] ===== Physics-Informed Active Learning (v2) =====")
        print(f"[PIAL-v2] 池子集大小: {len(pool_indices)} | 选取: {num_to_select}")

        # ---- Step 1: MC Dropout 不确定性 (主导信号) ----
        uncertainty_scores = self._compute_mc_dropout_uncertainty(
            pl_module, device, pool_indices, dm)

        # ---- Step 2: PCA 轨迹多样性 (次信号) ----
        pred_trajectories = self._predict_pool_trajectories(
            pl_module, device, pool_indices, dm)
        diversity_scores = self._compute_trajectory_diversity_pca(
            pred_trajectories, dm)

        # ---- Step 3: ODE 残差 (辅助信号) ----
        residual_scores = self._compute_ode_residual(
            pred_trajectories, pool_indices, dm)

        # ---- Step 4: 归一化并综合 ----
        def normalize(arr):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            lo, hi = np.min(arr), np.max(arr)
            if hi - lo < 1e-12:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        uncertainty_norm = normalize(uncertainty_scores)
        diversity_norm = normalize(diversity_scores)
        residual_norm = normalize(residual_scores)

        # 动态权重：早期重不确定性，后期重多样性，残差始终辅助
        max_epochs = getattr(pl_module, 'max_epochs', 200)
        progress = min(1.0, current_epoch / max(1, max_epochs))
        alpha = 0.6 - 0.1 * progress   # 不确定性: 0.6 → 0.5 (主导)
        beta = 0.3 + 0.1 * progress    # 多样性: 0.3 → 0.4 (次之)
        gamma = 0.1                     # 残差: 0.1 (辅助, 始终低权重)

        combined_scores = (alpha * uncertainty_norm
                         + beta * diversity_norm
                         + gamma * residual_norm)

        print(f"[PIAL-v2] 不确定性: min={uncertainty_scores.min():.4f} "
              f"median={np.median(uncertainty_scores):.4f} "
              f"max={uncertainty_scores.max():.4f}")
        print(f"[PIAL-v2] 多样性(PCA): min={diversity_scores.min():.4f} "
              f"median={np.median(diversity_scores):.4f} "
              f"max={diversity_scores.max():.4f}")
        print(f"[PIAL-v2] 残差(辅助): min={residual_scores.min():.4f} "
              f"median={np.median(residual_scores):.4f} "
              f"max={residual_scores.max():.4f}")
        print(f"[PIAL-v2] 权重: α(不确定)={alpha:.3f} β(多样)={beta:.3f} "
              f"γ(残差)={gamma:.3f} (epoch={current_epoch}, progress={progress:.2f})")

        # ---- Step 5: 选取 top-K ----
        num_to_select = min(num_to_select, len(combined_scores))
        selected_idx = np.argsort(combined_scores)[-num_to_select:]

        print(f"[PIAL-v2] 选中 top-{num_to_select} (综合得分阈值: "
              f"{combined_scores[selected_idx].min():.4f})")

        return selected_idx

    def _compute_mc_dropout_uncertainty(self, pl_module, device, pool_indices, dm,
                                         n_mc_samples=10, noise_std=0.02):
        """
        MC Dropout 不确定性估计（输入扰动法，对所有模型通用）。

        原理：对同一样本的参数 p 加微小高斯噪声，前向 K 次，计算预测方差。
        方差大 → 模型对该参数区域敏感 → 不确定性高 → 信息量大。

        与传统 MC Dropout 的区别：
        - 传统 MC Dropout 需要模型有 Dropout 层且推理时保持开启
        - 输入扰动法对所有模型通用，等效于贝叶斯神经网络的后验近似

        Args:
            pl_module: Lightning 模块
            device: 计算设备
            pool_indices: 池中样本索引
            dm: 数据模块
            n_mc_samples: MC 采样次数 (默认 10)
            noise_std: 参数扰动标准差 (默认 0.02, 归一化空间)

        Returns:
            uncertainty_scores: (N,) 每个样本的预测方差均值
        """
        pl_module.eval()
        batch_size = 500
        all_uncertainties = []

        # 获取参数归一化统计量，确保扰动在归一化空间
        p_std = dm.p_std.to(device)  # (1, 141)

        with torch.no_grad():
            for i in range(0, len(pool_indices), batch_size):
                end_idx = min(i + batch_size, len(pool_indices))
                batch_indices = pool_indices[i:end_idx]

                X_batch = self._get_pool_X_batch(batch_indices).to(device)
                P_batch = self._pool_P[batch_indices].to(device)  # (B, 141)
                B = P_batch.shape[0]

                # MC 采样: 对参数加噪声前向 K 次
                preds_list = []
                for _ in range(n_mc_samples):
                    # 在归一化参数空间加高斯噪声
                    noise = torch.randn_like(P_batch) * noise_std
                    P_perturbed = P_batch + noise
                    preds = pl_module.forward_ic_time(X_batch, P_perturbed)  # (B, 39, T)
                    preds_list.append(preds)

                # 计算预测方差: 沿 MC 采样维度求方差，再对变量和时间取平均
                preds_stack = torch.stack(preds_list, dim=0)  # (K, B, 39, T)
                variance = torch.var(preds_stack, dim=0)      # (B, 39, T)
                uncertainty = variance.mean(dim=(1, 2))        # (B,)
                all_uncertainties.append(uncertainty.cpu().numpy())

        pl_module.train()
        return np.concatenate(all_uncertainties)

    def _predict_pool_trajectories(self, pl_module, device, pool_indices, dm):
        """
        对池子集进行模型推理，返回反归一化后的预测轨迹。

        Args:
            pl_module: Lightning 模块
            device: 计算设备
            pool_indices: 池中样本索引
            dm: 数据模块 (提供 mean_y/std_y 用于反归一化)

        Returns:
            pred_trajectories: (N, 39, T) numpy 数组，真实值空间
        """
        pl_module.eval()
        batch_size = 1000
        all_preds = []

        mean_y = dm.mean_y.to(device)  # (1, 39, 1)
        std_y = dm.std_y.to(device)    # (1, 39, 1)

        with torch.no_grad():
            for i in range(0, len(pool_indices), batch_size):
                end_idx = min(i + batch_size, len(pool_indices))
                batch_indices = pool_indices[i:end_idx]

                X_batch = self._get_pool_X_batch(batch_indices).to(device)
                P_batch = self._pool_P[batch_indices].to(device)

                # 模型预测: (B, 39, T) 归一化空间
                preds_norm = pl_module.forward_ic_time(X_batch, P_batch)

                # 反归一化到真实值空间 (ODE eqns 在真实值空间定义)
                preds_real = preds_norm * std_y + mean_y  # 广播: (B, 39, T)
                all_preds.append(preds_real.cpu().numpy())

        pl_module.train()
        return np.concatenate(all_preds, axis=0)  # (N, 39, T)

    def _compute_ode_residual(self, pred_trajectories, pool_indices, dm):
        """
        计算 ODE 残差得分（带 event 突变掩码）。

        核心思想（导师要求）：
        - 残差 = 模型预测解 û(t) 代入物理 ODE 后方程不等于 0 的偏差
        - **不是** 预测值与真实值的差（很多场景没有真实值）
        - ODE: du/dt = f(u, p)
        - 数值微分: dû/dt ≈ 中心差分
        - 残差: r(t) = dû/dt - f(û(t), p)
        - 得分: mean_t ||r(t)||₂

        Event 突变处理（关键）：
        - 出芽酵母 ODE 系统有 4 种 terminal events（分裂/S期退出/DNA复制/纺锤体通过）
        - Event 触发时变量值被硬重置（如 BUD→0, SPN→0, MASS 衰减）
        - 插值后突变被"涂抹"到小区间，中心差分在这些点产生虚假巨大导数
        - 解决方案：检测 event 点 → 构建时间掩码 → 只在连续区间计算残差

        Args:
            pred_trajectories: (N, 39, T) 模型预测轨迹 (真实值空间)
            pool_indices: 池中样本索引 (用于获取参数)
            dm: 数据模块 (用于反归一化参数)

        Returns:
            residual_scores: (N,) 每个样本的平均 ODE 残差范数（仅在连续区间）
        """
        N, V, T = pred_trajectories.shape  # V=39
        dt = dm.t_max / (T - 1)  # 真实时间步长 (min)

        # 反归一化参数向量 → ODE 参数字典
        pool_P_norm = self._pool_P[pool_indices].cpu().numpy()  # (N, 141)
        p_mean = dm.p_mean.cpu().numpy().flatten()  # (141,)
        p_std = dm.p_std.cpu().numpy().flatten()    # (141,)
        pool_P_real = pool_P_norm * p_std + p_mean  # (N, 141) 真实参数值

        residual_scores = np.zeros(N)

        for i in range(N):
            u = pred_trajectories[i]  # (39, T)

            # ---- Step 1: 检测 event 突变点，构建时间掩码 ----
            mask = self._detect_event_mask(u)

            # ---- Step 2: 数值微分 (中心差分) ----
            du_dt = np.zeros_like(u)  # (39, T)
            du_dt[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dt)
            du_dt[:, 0] = (u[:, 1] - u[:, 0]) / dt
            du_dt[:, -1] = (u[:, -1] - u[:, -2]) / dt

            # ---- Step 3: 构建 ODE 参数字典 ----
            p_dict = dict(zip(PARAM_KEYS, pool_P_real[i]))
            p_dict['is_mutant_104'] = False

            # ---- Step 4: 计算 f(u, p) at each timestep ----
            f_u = np.zeros_like(u)  # (39, T)
            for t_idx in range(T):
                try:
                    f_u[:, t_idx] = _yeast_ode_eqns(0, u[:, t_idx], p_dict)
                except Exception:
                    f_u[:, t_idx] = 0.0

            # ---- Step 5: 残差 r(t) = dû/dt - f(u, p)，仅在掩码区间计算 ----
            residual = du_dt - f_u  # (39, T)

            # 防御性处理: 早期模型预测可能产生 inf/nan, 用 nan_to_num 替换为有限值
            residual = np.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)

            # 应用掩码：只在连续区间（mask=True）的时间步计算残差
            valid_count = mask.sum()
            if valid_count < T * 0.1:
                # 掩码过度（超过 90% 被屏蔽），降级为全轨迹计算
                # 防止异常轨迹导致无有效残差
                residual_scores[i] = np.sqrt(np.mean(residual ** 2))
            else:
                masked_residual = residual[:, mask]  # (39, valid_count)
                residual_scores[i] = np.sqrt(np.mean(masked_residual ** 2))

        # 最终防御: 确保所有得分有限 (避免 normalize 产生 nan)
        residual_scores = np.nan_to_num(residual_scores, nan=0.0,
                                         posinf=1e6, neginf=-1e6)
        return residual_scores

    def _detect_event_mask(self, u, event_window=4, event_tol=0.08,
                            jump_ratio=0.3):
        """
        检测轨迹中的 event 突变点，构建时间掩码。

        出芽酵母 ODE 系统有 4 种 terminal events，触发时变量被硬重置：
        - event1: CLB2=0.3↓ (分裂) → BUD/SPN→0, MASS衰减, lte1→0.1
        - event2: CLB2+CLB5=0.2↓ (S期退出) → ORI→0
        - event3: ORI=1.0↑ (DNA复制) → Vi20/BUB2激活
        - event4: SPN=1.0↑ (纺锤体通过) → Vi20/lte1/BUB2重置

        检测策略（双重保险）：
        1. **事件条件检测**：检查 4 个 event 的触发条件是否接近满足
        2. **硬跳变检测**：检测受 event 影响变量的"归零/数量级跳变"
           （用变量值域作为基准，而非中位数，避免平缓变量误触发）

        在检测到的 event 点附近开一个窗口（±event_window 步），因为线性插值
        会把突变"涂抹"到一个区间内。窗口内的残差被屏蔽。

        Args:
            u: (39, T) 预测轨迹 (真实值空间)
            event_window: 掩码窗口半径（时间步数），默认 4 步 ≈ 1.7 min
            event_tol: 事件触发条件的容差
            jump_ratio: 跳变检测阈值（占变量值域的比例，0.3=30%）

        Returns:
            mask: (T,) 布尔数组，True=连续区间（可计算残差），False=event 区间
        """
        T = u.shape[1]
        mask = np.ones(T, dtype=bool)

        # 变量索引
        nMASS, nCLB2, nCLB5 = 0, 2, 3
        nORI, nBUD, nSPN = 33, 34, 35
        nVi20, nlte1, nBUB2 = 36, 37, 38

        # ====== 策略 1: 事件条件检测 ======
        clb2 = u[nCLB2]
        clb5 = u[nCLB5]
        ori = u[nORI]
        spn = u[nSPN]

        # event1: CLB2 ≈ 0.3 且在下降
        event1_points = (np.abs(clb2 - 0.3) < event_tol) & (np.diff(clb2, prepend=clb2[0]) < 0)

        # event2: CLB2+CLB5 ≈ 0.2 且在下降
        clb2_clb5 = clb2 + clb5
        event2_points = (np.abs(clb2_clb5 - 0.2) < event_tol) & (np.diff(clb2_clb5, prepend=clb2_clb5[0]) < 0)

        # event3: ORI ≈ 1.0 且在上升
        event3_points = (np.abs(ori - 1.0) < event_tol) & (np.diff(ori, prepend=ori[0]) > 0)

        # event4: SPN ≈ 1.0 且在上升
        event4_points = (np.abs(spn - 1.0) < event_tol) & (np.diff(spn, prepend=spn[0]) > 0)

        all_event_points = event1_points | event2_points | event3_points | event4_points

        # ====== 策略 2: 受 event 影响变量的硬跳变检测 ======
        # 只检测"归零/数量级跳变"等极端情况
        # 用变量值域（max-min）作为基准，而非中位数，避免平缓变量误触发
        event_vars = [nMASS, nCLB2, nCLB5, nORI, nBUD, nSPN, nVi20, nlte1, nBUB2]
        for v in event_vars:
            d = np.abs(np.diff(u[v], prepend=u[v, 0]))
            value_range = np.max(u[v]) - np.min(u[v])
            if value_range > 1e-8:
                # 跳变幅度超过变量值域的 jump_ratio 才认为是 event 突变
                abnormal = d > jump_ratio * value_range
                all_event_points |= abnormal

        # ====== 扩展掩码窗口 ======
        # 线性插值会把突变"涂抹"到附近几个时间步
        event_indices = np.where(all_event_points)[0]
        for t in event_indices:
            t_start = max(0, t - event_window)
            t_end = min(T, t + event_window + 1)
            mask[t_start:t_end] = False

        return mask

    def _compute_trajectory_diversity(self, pred_trajectories, dm):
        """
        计算轨迹多样性得分。

        在**轨迹空间**（39×T 行为空间）而非参数空间度量多样性。
        多样性 = 样本预测轨迹到当前训练集轨迹的最小欧氏距离。

        与 _compute_diversity (参数空间) 的区别：
        - 参数空间多样性：两个参数相近的样本可能有完全不同的动态行为
        - 轨迹空间多样性：直接衡量"行为是否已被覆盖"，更符合 AL 的目标

        Args:
            pred_trajectories: (N, 39, T) 池子集预测轨迹 (真实值空间)
            dm: 数据模块 (用于获取训练集轨迹)

        Returns:
            diversity_scores: (N,) 每个样本到训练集的最小轨迹距离
        """
        N, V, T = pred_trajectories.shape

        # 从训练集采样代表点，使用其真实标签轨迹（归一化）反归一化到真实值空间
        train_X, train_P, train_Y_norm = dm.train_dataset_lhs.tensors
        n_representatives = min(500, train_X.shape[0])
        rng = np.random.RandomState(self.random_seed)
        rep_idx = rng.choice(train_X.shape[0], n_representatives, replace=False)

        # 训练集代表点的真实轨迹（标签），反归一化到真实值空间
        train_Y_norm_np = train_Y_norm[rep_idx].cpu().numpy()  # (R, 39, T) 归一化
        mean_y = dm.mean_y.cpu().numpy().reshape(-1)  # (39,)
        std_y = dm.std_y.cpu().numpy().reshape(-1)    # (39,)
        train_Y_real = train_Y_norm_np * std_y[:, None] + mean_y[:, None]  # (R, 39, T)

        # 展平轨迹为向量 (39*T,) 用于距离计算
        pool_flat = pred_trajectories.reshape(N, -1)  # (N, 39*T)
        train_flat = train_Y_real.reshape(n_representatives, -1)  # (R, 39*T)

        # 分批计算距离以避免内存溢出
        batch_size = 500
        diversity_scores = []
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            batch = pool_flat[i:end_idx]
            distances = euclidean_distances(batch, train_flat)
            min_distances = distances.min(axis=1)
            diversity_scores.append(min_distances)

        return np.concatenate(diversity_scores)

    def _compute_trajectory_diversity_pca(self, pred_trajectories, dm, n_components=20):
        """
        PCA 降维后的轨迹多样性得分。

        解决原版 _compute_trajectory_diversity 的维度灾难问题：
        - 原版: 展平为 (39*T=19500,) 维向量算欧氏距离，高维下距离趋同
        - PCA 版: 先降到 20 维，保留 95%+ 方差，在低维空间算距离

        算法：
        1. 合并池预测轨迹和训练集轨迹，统一展平
        2. PCA 降到 n_components 维（拟合在训练集上，避免数据泄露）
        3. 在低维空间计算池样本到训练集的最小欧氏距离

        Args:
            pred_trajectories: (N, 39, T) 池子集预测轨迹 (真实值空间)
            dm: 数据模块 (用于获取训练集轨迹)
            n_components: PCA 降维目标维度 (默认 20)

        Returns:
            diversity_scores: (N,) 每个样本到训练集的最小轨迹距离(PCA空间)
        """
        from sklearn.decomposition import PCA

        N, V, T = pred_trajectories.shape

        # 从训练集采样代表点
        train_X, train_P, train_Y_norm = dm.train_dataset_lhs.tensors
        n_representatives = min(500, train_X.shape[0])
        rng = np.random.RandomState(self.random_seed)
        rep_idx = rng.choice(train_X.shape[0], n_representatives, replace=False)

        # 训练集代表点的真实轨迹，反归一化到真实值空间
        train_Y_norm_np = train_Y_norm[rep_idx].cpu().numpy()  # (R, 39, T)
        mean_y = dm.mean_y.cpu().numpy().reshape(-1)  # (39,)
        std_y = dm.std_y.cpu().numpy().reshape(-1)    # (39,)
        train_Y_real = train_Y_norm_np * std_y[:, None] + mean_y[:, None]  # (R, 39, T)

        # 展平轨迹
        pool_flat = pred_trajectories.reshape(N, -1)        # (N, 39*T)
        train_flat = train_Y_real.reshape(n_representatives, -1)  # (R, 39*T)

        # PCA 拟合在训练集上（避免池数据泄露影响主成分方向）
        n_comp = min(n_components, n_representatives, train_flat.shape[1])
        pca = PCA(n_components=n_comp, random_state=self.random_seed)
        pca.fit(train_flat)

        # 降维
        pool_pca = pca.transform(pool_flat)     # (N, n_comp)
        train_pca = pca.transform(train_flat)   # (R, n_comp)

        # 低维空间计算最小欧氏距离
        batch_size = 500
        diversity_scores = []
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            batch = pool_pca[i:end_idx]
            distances = euclidean_distances(batch, train_pca)
            min_distances = distances.min(axis=1)
            diversity_scores.append(min_distances)

        explained_var = pca.explained_variance_ratio_.sum()
        print(f"[PIAL-v2] PCA: {train_flat.shape[1]}→{n_comp} 维, "
              f"保留方差: {explained_var:.1%}")

        return np.concatenate(diversity_scores)

    def _move_samples_to_training_set(self, selected_pool_idx, current_epoch):
        """
        将选中样本从未标注池迁移到训练集。
        这是标准 AL 的"标注"步骤——在模拟环境中，标签已存在 (ODE 预计算)，
        迁移即相当于完成标注并加入训练集。

        Args:
            selected_pool_idx: 池中选中样本的全局索引
            current_epoch: 当前 epoch

        Returns:
            broadcast_data: 包含选中样本数据的字典
        """
        # 提取选中样本
        selected_X = self._pool_X[selected_pool_idx]
        selected_P = self._pool_P[selected_pool_idx]
        selected_Y = self._pool_Y[selected_pool_idx]

        # 从池中移除选中样本 (使用 mask)
        mask = np.ones(self._pool_size, dtype=bool)
        mask[selected_pool_idx] = False
        self._pool_X = self._pool_X[mask]
        self._pool_P = self._pool_P[mask]
        self._pool_Y = self._pool_Y[mask]
        self._pool_size = self._pool_X.shape[0]

        print(f"[AL Manager] 样本迁移完成: {len(selected_pool_idx)} 个样本已加入训练集 | "
              f"池剩余: {self._pool_size}")

        # 组装广播数据 (与现有框架兼容)
        return {
            'X': selected_X,
            'Y': selected_Y,
            'P': selected_P,
            'names': [f'AL_Pool_Epoch{current_epoch}_{i}' for i in range(len(selected_pool_idx))],
            'labels': ['Pattern_Unknown'] * len(selected_pool_idx),
            'sample_plots': {}  # 标准池化 AL 不生成对比图 (无 ODE 重解算)
        }

    # ==================================================================
    # RGS: Real-Guided Sampling 查询合成策略 (标准 AL 框架内的查询合成)
    # ==================================================================
    def _rgs_sample(self, trainer, pl_module, dm, device, current_epoch):
        """
        RGS (Real-Guided Sampling) 查询合成策略：
        在标准 AL 框架内，RGS 采用"查询合成"范式——不从未标注池中选取，
        而是通过 LHS 引擎生成新的候选样本，并用 ODE 物理验证作为"标注"。

        标准 AL 循环对应关系：
        - 初始训练集选择: on_train_start (K-Means，与其他策略一致)
        - 模型训练: Lightning Trainer
        - 样本选择 (查询合成): 本方法，通过 LHS 引擎生成候选
        - 标签更新: ODE 物理验证 (solve_ivp) 生成真实标签
        - 模型迭代优化: 下一轮 epoch

        RGS 核心逻辑：
        1. 在测试集 (Real Mutant) 上推理，计算每个 Mutant 的 MAE
        2. 按 MAE 排序，聚焦 top-K 最差样本分配 RGS 名额
        3. 在差样本的参数邻域内做局部 LHS 扰动 (90% 预算)
        4. 全局 LHS 采样保持多样性 (10% 预算)
        5. ODE 验证全部候选 → 用真实 MAE (模型预测 vs ODE真值) 选取最难的 num_add 个
        """
        # 初始化全局 LHS 预算 (会被 _allocate_rgs_quota 设置)
        self._global_lhs_budget = 0

        # Step 1: 评估 Real Mutant 的表现 (MAE)
        print("[RGS] Step 1: 评估 Real Mutant 当前表现 (MAE)...")
        mutant_maes = self._evaluate_real_mutants(pl_module, dm, device)
        
        if not mutant_maes:
            print("[RGS] 无法评估 Real Mutant，降级为全局 LHS 采样")
            return self._legacy_sample(trainer, pl_module, dm, device, current_epoch)

        # Step 2: 按 MAE 分配采样名额 (聚焦 top-K，不撒胡椒面)
        print(f"[RGS] Step 2: 按 MAE 分配采样名额 (mae_threshold={self.mae_threshold})...")
        sampling_plan = self._allocate_rgs_quota(mutant_maes)
        
        n_target_mutants = len(sampling_plan)
        n_rgs_samples = sum(sampling_plan.values())
        global_budget = self._global_lhs_budget
        print(f"[RGS] RGS 局部采样: {n_target_mutants} 个突变体, {n_rgs_samples} 样本 | "
              f"全局 LHS: {global_budget} 样本")

        # Step 3: 生成全部候选 (不做预筛选，留给 ODE 后的真实 MAE 选取)
        all_params_list = []

        # Step 3a: RGS 局部采样候选
        if sampling_plan:
            print(f"[RGS] Step 3a: 调用 generate_local_lhs_tasks (perturbation={self.perturbation*100:.0f}%)...")
            rgs_tasks_dict = generate_local_lhs_tasks(
                target_mutants=sampling_plan,
                perturbation=self.perturbation
            )
            
            if rgs_tasks_dict:
                for m_name, overrides in rgs_tasks_dict.items():
                    p_vec = [overrides.get(k, _defaults[k]) for k in PARAM_KEYS]
                    all_params_list.append((f"RGS_Epoch{current_epoch}_{m_name}", p_vec))
                print(f"[RGS] 局部采样引擎返回 {len(rgs_tasks_dict)} 个候选")
            else:
                print("[RGS] 局部采样引擎未生成任何样本")
                global_budget += n_rgs_samples  # 转给全局 LHS

        # Step 3b: 全局 LHS 采样候选 (保持多样性)
        if global_budget > 0:
            print(f"[RGS] Step 3b: 全局 LHS 采样 (保持多样性)...")
            per_mutant = max(1, global_budget // len(mutant_rules))
            global_tasks_dict = generate_lhs_tasks(per_mutant)
            
            p_pool_list_g = []
            pool_names_g = []
            for m_name, overrides in global_tasks_dict.items():
                p_vec = [overrides.get(k, _defaults[k]) for k in PARAM_KEYS]
                p_pool_list_g.append(p_vec)
                pool_names_g.append(m_name)
                
            P_pool_g = np.array(p_pool_list_g)
            actual_pool_size_g = len(P_pool_g)
            global_num_add = min(global_budget, actual_pool_size_g)
            
            idx_sel = np.random.choice(actual_pool_size_g, global_num_add, replace=False)
            for idx in idx_sel:
                all_params_list.append((f"Global_Epoch{current_epoch}_{pool_names_g[idx]}", P_pool_g[idx]))
            
            print(f"[RGS] 全局 LHS 返回 {actual_pool_size_g} 个候选，随机选取 {global_num_add} 个")

        if not all_params_list:
            return None

        print(f"[RGS] 合并候选总数: {len(all_params_list)}")

        # Step 3.5: 参数合理性校验（在昂贵的 ODE 验证前剔除明显异常的参数）
        all_params_list = self._validate_param_ranges(all_params_list)

        # Step 4: ODE 物理验证全部候选
        valid_results, valid_names = self._physics_validate_and_prune(all_params_list, dm)
        
        if not valid_results:
            return None

        # Step 5: 用真实 MAE (模型预测 vs ODE 真值) 选取最难的 num_add 个
        selected_results, selected_names = self._select_by_real_mae(
            pl_module, dm, device, valid_results, valid_names, current_epoch)

        if not selected_results:
            return None

        # Step 6: 绘制采样对比图 (每种 Pattern 各一张)
        sample_plots = self._plot_al_sample_comparison(
            selected_results, selected_names, dm, current_epoch)

        # Step 7: 组装广播数据
        return {
            'X': torch.stack([torch.tensor(r[0]) for r in selected_results]).float(),
            'Y': torch.stack([torch.tensor(r[1]) for r in selected_results]).float(),
            'P': torch.stack([torch.tensor(r[2]) for r in selected_results]).float(),
            'names': selected_names,
            'labels': [r[3] for r in selected_results],
            'sample_plots': sample_plots
        }

    # ==================================================================
    # Mutant-Aware 策略：基于 atlas 数据的优先级加权采样
    # ==================================================================
    def _load_mutant_atlas(self):
        """
        加载 mutant_atlas_merged.csv，解析 lhs_MAE_min 和 lhs_n_samples 列。
        存储为 self._mutant_atlas = {mutant_name: {'lhs_mae_min', 'lhs_n_samples', 'pattern', 'lhs_mae_median'}}

        若文件不存在或路径未提供，打印警告并标记降级 (_atlas_loaded=False)。
        """
        import csv

        if self.atlas_csv_path is None:
            print("[Mutant-Aware] 警告: 未提供 atlas_csv_path，策略将降级为 RGS")
            self._atlas_loaded = False
            return

        try:
            self._mutant_atlas = {}
            with open(self.atlas_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mutant_name = row['Mutant']
                    self._mutant_atlas[mutant_name] = {
                        'lhs_mae_min': float(row['lhs_MAE_min']),
                        'lhs_n_samples': int(float(row['lhs_n_samples'])),
                        'pattern': row.get('Pattern', 'Unknown'),
                        'lhs_mae_median': float(row.get('lhs_MAE_median', 0.0)),
                    }
            self._atlas_loaded = True
            print(f"[Mutant-Aware] Atlas 加载成功: {len(self._mutant_atlas)} 个突变体 | "
                  f"源: {self.atlas_csv_path}")

            # 打印 top-5 高优先级突变体摘要
            sorted_atlas = sorted(self._mutant_atlas.items(),
                                  key=lambda x: x[1]['lhs_mae_min'], reverse=True)
            print("[Mutant-Aware] Top-5 lhs_MAE_min:")
            for name, info in sorted_atlas[:5]:
                print(f"  - {name}: lhs_MAE_min={info['lhs_mae_min']:.4f}, "
                      f"lhs_n_samples={info['lhs_n_samples']}")
        except Exception as e:
            print(f"[Mutant-Aware] 警告: Atlas 加载失败 ({e})，策略将降级为 RGS")
            self._mutant_atlas = None
            self._atlas_loaded = False

    def _sample_target_with_retry(self, sample_fn, mode_name, all_mutants,
                                  target_count, max_retries=3):
        """带补采的 target 采样, 确保每 mutant 最终有 target_count 个样本

        去重 (is_near_mother + is_near_any_real) 会丢弃部分样本, 对不足的 mutant
        自动补采直到达标或达到 max_retries 上限。

        Args:
            sample_fn: generate_targeted_lhs_tasks_density 或 _boundary
            mode_name: 'density' 或 'boundary' (用于打印和键名)
            all_mutants: 参与采样的 mutant 名列表
            target_count: 每 mutant 目标样本数
            max_retries: 最大补采轮数

        Returns:
            dict: {f"{m}_{mode_name}_{i:03d}": sample_dict}
        """
        PERTURBATION = 0.30
        result = {m: [] for m in all_mutants}

        # 初始家族配置: 全 mutant, 每 mutant target_count 个
        families = {
            m: {'mutants': [m], 'num_per_mutant': target_count,
                'perturbation': PERTURBATION, 'mode': mode_name}
            for m in all_mutants
        }

        for retry in range(max_retries):
            tasks = sample_fn(target_families=families,
                              default_num_per_mutant=target_count)
            # 按 mutant 分组 (键名格式: {m_name}_LHS_T{idx:03d})
            for name, sample_dict in tasks.items():
                if '_LHS_T' in name:
                    m_name = name.split('_LHS_T')[0]
                else:
                    m_name = name.rsplit('_LHS_', 1)[0]
                if m_name in result:
                    result[m_name].append(sample_dict)

            # 检查不足的 mutant
            insufficient = [m for m in all_mutants
                            if len(result[m]) < target_count]
            if not insufficient:
                print(f"[al4pde] {mode_name} 采样: 全部 {len(all_mutants)} 个 mutant "
                      f"达到 {target_count} 个")
                break

            # 补采: 对不足的 mutant 请求差额 + buffer
            buffer = 20
            families = {
                m: {'mutants': [m],
                    'num_per_mutant': target_count - len(result[m]) + buffer,
                    'perturbation': PERTURBATION, 'mode': mode_name}
                for m in insufficient
            }
            deficit = sum(target_count - len(result[m]) for m in insufficient)
            print(f"[al4pde] {mode_name} 补采第 {retry+1} 轮: {len(insufficient)} 个 "
                  f"mutant 不足, deficit 总计 {deficit} 个")

        # 截取每 mutant target_count 个, 重命名键名避免冲突
        # 键名格式: {m}_LHS_D{idx} (density) 或 {m}_LHS_B{idx} (boundary)
        # 保留 _LHS_ 子串使现有 split('_LHS_')[0] 提取逻辑自动兼容
        suffix = '_LHS_D' if mode_name == 'density' else '_LHS_B'
        final = {}
        short_count = 0
        for m in all_mutants:
            samples = result[m][:target_count]
            if len(samples) < target_count:
                short_count += 1
            for i, sample_dict in enumerate(samples):
                final[f"{m}{suffix}{i:03d}"] = sample_dict

        if short_count > 0:
            print(f"[al4pde] {mode_name} 警告: {short_count} 个 mutant 仍未达到 "
                  f"{target_count} 个 (去重丢弃过多)")

        return final

    def _generate_param_only_pool(self, dm):
        """生成只含参数的候选池 (实验开始前调用)

        候选池构建规范 (3 类采样, 每 mutant 各 300 个):
        - 常规采样 (normal): generate_lhs_tasks_v5(300), 全局均匀 LHS
        - target_density: 突变参数邻域密集采样 (mode=density)
        - target_boundary: 突变参数 0 值邻域采样 (mode=boundary, v5_boundary 算法)
        - 暂时弃用 anchor 模式
        - 去重后不足 300 个的 mutant 自动补采, 确保最终 300 个
        - 候选池内部不去重
        - 候选池只存参数信息 (P + Y_IC), 不含完整时间序列
        - 选中后通过 ODE 仿真生成真实轨迹 (al4pde 原始项目机制)

        内存优化: 只存 P_norm + Y_IC + P_real, 完整 Y 选中后 ODE 生成
        """
        NUM_PER_TYPE = 300
        # v2.2 优先级 3: cdh1 复杂突变体定向增强
        # 背景: 41/42/45/21/122 等含 cdh1 的多突变体在所有策略下都灾难性差 (Rel L2 上千)。
        # 改进: density 采样数量从 300 提升到 500, 给 AL 更多 cdh1 邻域候选。
        # 注1: perturbation 不能从 0.30 降到 0.15 (v5.py 内 max(..., 0.30) 强制下界)。
        # 注2: boundary 采样保持 300 (用户建议只增强 density)。
        # 注3: 去重机制 (is_near_mother + is_near_any_real) 不变, _sample_target_with_retry
        #      的 retry 机制确保去重后仍能达到目标数量, 保证参数对多样性。
        CDH1_DENSITY_COUNT = 500
        pure_wt = ['1_WT_Glc', '2.1_WT_Gal', '2.2_WT_Raff']
        all_mutants = [m for m in mutant_rules.keys() if m not in pure_wt]
        # 识别 cdh1 家族 (含 cdh1 的多突变体, 不区分大小写)
        cdh1_mutants = [m for m in all_mutants if 'cdh1' in m.lower()]
        non_cdh1_mutants = [m for m in all_mutants if 'cdh1' not in m.lower()]

        print(f"[al4pde] ===== 生成参数候选池 (3 类采样: normal/density/boundary) =====")
        print(f"[al4pde] 参与 mutant: {len(all_mutants)} 个 (排除 {len(pure_wt)} 个 WT)")
        print(f"[al4pde] 每类每 mutant: {NUM_PER_TYPE} 个 (cdh1 家族 density={CDH1_DENSITY_COUNT})")
        print(f"[al4pde] cdh1 家族: {len(cdh1_mutants)} 个 | 非 cdh1: {len(non_cdh1_mutants)} 个")

        # 1. 常规采样 (全局均匀 LHS)
        # 键名格式: {m}_LHS_{idx} (generate_lhs_tasks_v5 原始格式, 不加前缀)
        print(f"[al4pde] 1/3 常规采样: generate_lhs_tasks_v5({NUM_PER_TYPE})")
        normal_tasks = generate_lhs_tasks_v5(NUM_PER_TYPE)
        normal_tasks_renamed = dict(normal_tasks)  # 保持原键名
        print(f"[al4pde] 常规采样生成 {len(normal_tasks_renamed)} 个样本")

        # 2. target_density 采样 (突变参数邻域, v5 的 density 模式)
        # v2.2: cdh1 家族 500 个, 非 cdh1 家族 300 个 (分两次调用)
        print(f"[al4pde] 2/3 target_density 采样 (mode=density, pert=0.30)")
        density_tasks = {}
        if cdh1_mutants:
            print(f"[al4pde]   cdh1 家族 density 采样: {len(cdh1_mutants)} 个 mutant × {CDH1_DENSITY_COUNT}")
            cdh1_density = self._sample_target_with_retry(
                generate_targeted_lhs_tasks_density, 'density',
                cdh1_mutants, CDH1_DENSITY_COUNT)
            density_tasks.update(cdh1_density)
        if non_cdh1_mutants:
            print(f"[al4pde]   非 cdh1 家族 density 采样: {len(non_cdh1_mutants)} 个 mutant × {NUM_PER_TYPE}")
            non_cdh1_density = self._sample_target_with_retry(
                generate_targeted_lhs_tasks_density, 'density',
                non_cdh1_mutants, NUM_PER_TYPE)
            density_tasks.update(non_cdh1_density)
        print(f"[al4pde] density 采样生成 {len(density_tasks)} 个样本 "
              f"(cdh1: {len(cdh1_mutants) * CDH1_DENSITY_COUNT} 期望 + "
              f"非 cdh1: {len(non_cdh1_mutants) * NUM_PER_TYPE} 期望)")

        # 3. target_boundary 采样 (0 值邻域, v5_boundary 的 boundary 模式)
        print(f"[al4pde] 3/3 target_boundary 采样 (mode=boundary, pert=0.30)")
        boundary_tasks = self._sample_target_with_retry(
            generate_targeted_lhs_tasks_boundary, 'boundary',
            all_mutants, NUM_PER_TYPE)
        print(f"[al4pde] boundary 采样生成 {len(boundary_tasks)} 个样本")

        # 合并 (候选池内部不去重)
        all_tasks = {}
        all_tasks.update(normal_tasks_renamed)
        all_tasks.update(density_tasks)
        all_tasks.update(boundary_tasks)
        print(f"[al4pde] 合并候选池: {len(all_tasks)} 个样本 (内部不去重)")

        # 提取参数向量 + 构造 IC
        y0_base = initial_conds_v5()  # (39,) 固定初始条件

        pool_names = []
        pool_params = []
        pool_ics = []

        for name, overrides in all_tasks.items():
            p_vec = [overrides.get(k, _defaults[k]) for k in PARAM_KEYS]
            pool_names.append(name)
            pool_params.append(p_vec)

            # 构造 IC: 用 initial_conds 覆盖 init_CDH1T/init_CDH1/init_BUB2
            # nCDH1T=21, nCDH1=22, nBUB2=38 (与 lhs_v1_origin_v5.py 变量索引一致)
            y0 = y0_base.copy()
            if 'init_CDH1T' in overrides:
                y0[21] = overrides['init_CDH1T']
            if 'init_CDH1' in overrides:
                y0[22] = overrides['init_CDH1']
            if 'init_BUB2' in overrides:
                y0[38] = overrides['init_BUB2']
            pool_ics.append(y0)

        # 转为张量
        P_tensor = torch.tensor(np.array(pool_params), dtype=torch.float32)    # (N, 141) 未归一化
        Y_ic_tensor = torch.tensor(np.array(pool_ics), dtype=torch.float32)    # (N, 39) 未归一化

        # 归一化统计量 (与 yeast_datamodule.py 完全一致)
        mean_y = dm.mean_y  # (1, 39, 1)
        std_y = dm.std_y     # (1, 39, 1)
        p_mean = dm.p_mean   # (1, 141)
        p_std = dm.p_std     # (1, 141)

        # 归一化 P
        P_norm = (P_tensor - p_mean) / p_std
        P_norm = torch.nan_to_num(P_norm, nan=0.0, posinf=10.0, neginf=-10.0)

        # 归一化 Y_IC
        mean_y_ic = mean_y.squeeze(0).squeeze(-1)  # (39,)
        std_y_ic = std_y.squeeze(0).squeeze(-1)    # (39,)
        Y_ic_norm = (Y_ic_tensor - mean_y_ic) / std_y_ic
        Y_ic_norm = torch.nan_to_num(Y_ic_norm, nan=0.0, posinf=10.0, neginf=-10.0)

        # 构造 t_grid (共享, 不占额外内存)
        T_STEPS = 500
        t_grid = torch.linspace(0, 1, T_STEPS).reshape(1, 1, T_STEPS)

        # 统计 3 类采样分布 (仅用于打印)
        # 键名格式: {m}_LHS_{idx} (normal), {m}_LHS_D{idx} (density), {m}_LHS_B{idx} (boundary)
        normal_count = sum(1 for n in pool_names
                           if '_LHS_' in str(n) and '_LHS_D' not in str(n)
                           and '_LHS_B' not in str(n))
        density_count = sum(1 for n in pool_names if '_LHS_D' in str(n))
        boundary_count = sum(1 for n in pool_names if '_LHS_B' in str(n))

        # 存储候选池 (只含参数, 不含完整 Y)
        self._pool_X = None
        self._pool_P = P_norm               # (N, 141) 归一化参数
        self._pool_Y = None                  # 不预存完整 Y
        self._pool_Y_ic = Y_ic_norm          # (N, 39) 归一化 IC
        self._al4pde_t_grid = t_grid
        self._pool_mutant_names = np.array(pool_names)
        self._pool_size = len(pool_names)
        self._al4pde_pool_loaded = True

        # 存储未归一化参数 (用于选中后 ODE 仿真)
        self._pool_P_real = P_tensor         # (N, 141) 未归一化

        # 保存归一化统计量 (用于仿真后归一化 Y)
        self._al4pde_mean_y = mean_y
        self._al4pde_std_y = std_y

        # 不再需要磁盘加载相关变量
        self._al4pde_pool_idx = None
        self._al4pde_pool_path = None

        mem_gb = (P_norm.nelement() * 4 + Y_ic_norm.nelement() * 4 +
                  P_tensor.nelement() * 4) / (1024**3)
        print(f"[al4pde] 候选池构建完成: {self._pool_size} 样本 "
              f"(normal={normal_count}, density={density_count}, boundary={boundary_count}) | "
              f"P_norm(mean={P_norm.mean():.3f}, std={P_norm.std():.3f}), "
              f"Y_ic_norm(mean={Y_ic_norm.mean():.3f}, std={Y_ic_norm.std():.3f}) | "
              f"内存={mem_gb:.2f} GB (只存 P + Y_IC + P_real, 完整 Y 选中后 ODE 生成)")

    def _precompute_pool_ode_shape_features(self, dm):
        """预计算候选池所有样本的 ODE 真实轨迹形状特征并缓存

        用户洞察: diversity 不能只看参数距离, 需结合 mass pattern (轨迹形状)。
        但候选池只有参数+IC, 没有轨迹, 用模型预测不靠谱 (训练早期模型差)。
        正确做法: 用 ODE 求解器解出真实轨迹, 提取形状特征后缓存。

        过滤逻辑 (对齐 lhs_v1_origin.py):
        1. 仿真失败 (返回 None) → 丢弃
        2. max_mass > 10.0 → 丢弃 (lhs_v1_origin.py L371)
        3. 动态浓度异常 → 丢弃 (lhs_v1_origin.py L900-935)
           - RELATIVE_MULTIPLIER = 2.0 (训练集 max × 2)
           - ABSOLUTE_SAFE_LIMIT = 2.0 (最低阈值)
           - 检查变量 1-38 (跳过 MASS=0)

        本方法在 AL 初始化时调用一次 (on_train_start):
        1. 对候选池所有样本并行 ODE 仿真 (solve_single_mutant_bridge)
        2. 过滤极端值 (mass>10 / 浓度爆炸)
        3. 从真实轨迹提取 332 维形状特征 (MASS pattern + 全变量统计 + 参数)
        4. 缓存特征供后续所有 AL 轮次使用 (ODE 轨迹不随训练变化)
        """
        n_pool = self._pool_size
        print(f"[al4pde] ===== 预计算 ODE 形状特征 (diversity/maal 专用) =====")
        print(f"[al4pde] 对 {n_pool} 个候选池样本并行 ODE 仿真...")

        t_start = time.time()
        sim_results = [None] * n_pool
        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = {
                i: executor.submit(
                    solve_single_mutant_bridge,
                    str(self._pool_mutant_names[i]),
                    self._pool_P_real[i].numpy()
                )
                for i in range(n_pool)
            }
            done = 0
            for i, f in futures.items():
                try:
                    sim_results[i] = f.result(timeout=SIM_TIMEOUT + 30)
                except FutureTimeoutError:
                    print(f"  ⚠️ 候选池样本 {i} ({self._pool_mutant_names[i]}) 进程级超时,跳过")
                    sim_results[i] = None
                done += 1
                if done % max(1, n_pool // 10) == 0 or done == n_pool:
                    print(f"  ODE 仿真进度: {done}/{n_pool} ({100*done//n_pool}%)")

        elapsed_ode = time.time() - t_start
        n_sim_ok = sum(1 for r in sim_results if r is not None)
        n_sim_fail = n_pool - n_sim_ok
        print(f"[al4pde] ODE 仿真完成: {n_sim_ok}/{n_pool} 成功, {n_sim_fail} 失败, 耗时 {elapsed_ode:.0f}s")

        # === 过滤阶段 1: 仿真失败 ===
        valid_mask = np.array([r is not None for r in sim_results])

        # === 过滤阶段 2: 极端值过滤 (对齐 lhs_v1_origin.py) ===
        # 计算动态浓度阈值: 取 real mutants 每个变量的历史最大值 × 2.0 (最低 2.0)
        # 用 real mutants 包络作为生物学合理上界, 比用 LHS 训练集更可靠
        Y_real_norm = dm.test_dataset_real.tensors[2].cpu()
        std_y_cpu = dm.std_y.cpu()
        mean_y_cpu = dm.mean_y.cpu()
        Y_real_raw = Y_real_norm * std_y_cpu + mean_y_cpu
        real_envelope = Y_real_raw.max(dim=-1)[0].max(dim=0)[0].numpy()
        dynamic_thresholds = np.maximum(real_envelope * 2.0, 2.0)

        n_mass_discard = 0
        n_conc_discard = 0
        for i in range(n_pool):
            if not valid_mask[i]:
                continue
            _, y_uniform, _, _ = sim_results[i]
            max_mass = float(np.max(y_uniform[0]))  # y_uniform: (39, 500), MASS = y[0]

            # 过滤 2a: max_mass > 10.0 (lhs_v1_origin.py L371)
            if max_mass > 10.0:
                valid_mask[i] = False
                n_mass_discard += 1
                continue

            # 过滤 2b: 动态浓度异常 (lhs_v1_origin.py L916-924)
            lhs_max_vals = np.max(y_uniform, axis=1)
            is_monster = any(lhs_max_vals[v] > dynamic_thresholds[v] for v in range(1, 39))
            if is_monster:
                valid_mask[i] = False
                n_conc_discard += 1

        n_valid = int(valid_mask.sum())
        n_total_discard = n_pool - n_valid
        print(f"[al4pde] 过滤完成: {n_valid}/{n_pool} 通过 | "
              f"仿真失败={n_sim_fail}, mass>10={n_mass_discard}, 浓度爆炸={n_conc_discard}")

        # 移除未通过过滤的样本
        if n_total_discard > 0:
            self._pool_P = self._pool_P[valid_mask]
            if self._pool_Y_ic is not None:
                self._pool_Y_ic = self._pool_Y_ic[valid_mask]
            if self._pool_mutant_names is not None:
                self._pool_mutant_names = self._pool_mutant_names[valid_mask]
            if self._pool_P_real is not None:
                self._pool_P_real = self._pool_P_real[valid_mask]
            self._pool_size = n_valid
            sim_results = [r for r, m in zip(sim_results, valid_mask) if m]
            print(f"[al4pde] 移除 {n_total_discard} 个无效样本, 候选池剩余 {self._pool_size}")

        # 从真实 ODE 轨迹提取形状特征
        features = []
        for idx, res in enumerate(sim_results):
            _, y_uniform, p_vec, _ = res  # y_uniform: (39, 500) 真实轨迹
            y_tensor = torch.from_numpy(y_uniform).float().unsqueeze(0)  # (1, 39, 500)
            p_norm = self._pool_P[idx:idx+1]  # (1, 141) 已归一化 (与过滤后池对齐)
            feat = self._extract_shape_features(y_tensor, p_norm)  # (1, 332)
            features.append(feat.squeeze(0))

        self._pool_shape_features_cached = torch.stack(features, dim=0)  # (N, 332)
        mem_mb = self._pool_shape_features_cached.nelement() * 4 / (1024**2)
        print(f"[al4pde] 形状特征缓存完成: {self._pool_shape_features_cached.shape}, "
              f"内存={mem_mb:.1f} MB")

    def _get_pool_X_batch(self, indices):
        """获取候选池样本的 X = [IC, t_grid] (内存优化版)

        标准池化路径 (us/is/random/pial): self._pool_X 已预存, 直接索引返回
        al4pde 路径 (内存优化): 从 self._pool_Y_ic 动态构造 X
          - IC = Y_ic.unsqueeze(2).expand(-1, -1, T)  (视图, 不占额外内存)
          - t_grid = self._al4pde_t_grid.expand(B, 39, T)  (共享, 不占额外内存)
          - X = stack([IC, t_grid], dim=2)  (B, 39, 2, T)  — 仅此 batch 占内存

        Args:
            indices: 样本索引 (numpy array / slice / list / tensor)

        Returns:
            X: (B, 39, 2, T) tensor
        """
        if self._pool_X is not None:
            return self._pool_X[indices]
        # al4pde 优化路径: 从 Y_IC 动态构造 X (不需要完整 Y)
        Y_ic_batch = self._pool_Y_ic[indices]  # (B, 39)
        T = self._al4pde_t_grid.shape[2]
        IC = Y_ic_batch.unsqueeze(2).expand(-1, -1, T)  # (B, 39, T) — 视图
        t_grid = self._al4pde_t_grid.to(IC.device).expand(IC.shape[0], 39, T)
        return torch.stack([IC, t_grid], dim=2)  # (B, 39, 2, T)

    def _mutant_aware_sample(self, trainer, pl_module, dm, device, current_epoch):
        """
        Mutant-Aware 查询合成策略：
        利用 atlas 历史(lhs_MAE_min, lhs_n_samples) + 当前模型 MAE 计算每个 Real Mutant
        的优先级，聚焦采样到 LHS 覆盖不足 + 模型表现差的突变体邻域。

        核心公式：
        priority = 0.5 * norm(lhs_MAE_min) + 0.3 * (1 - lhs_n_samples/500) + 0.2 * norm(current_real_mae)

        若 atlas 不可用，降级为 _rgs_sample。
        """
        # Step 0: 检查 atlas 是否可用，否则降级
        if not self._atlas_loaded or self._mutant_atlas is None:
            print("[Mutant-Aware] Atlas 不可用，降级为 RGS 策略")
            return self._rgs_sample(trainer, pl_module, dm, device, current_epoch)

        self._global_lhs_budget = 0

        # Step 1: 评估 Real Mutant 当前表现 (MAE) - 复用 RGS 的评估逻辑
        print("[Mutant-Aware] Step 1: 评估 Real Mutant 当前表现 (MAE)...")
        mutant_maes = self._evaluate_real_mutants(pl_module, dm, device)

        if not mutant_maes:
            print("[Mutant-Aware] 无法评估 Real Mutant，降级为全局 LHS 采样")
            return self._legacy_sample(trainer, pl_module, dm, device, current_epoch)

        # Step 2: 计算每个 mutant 的 priority score
        print("[Mutant-Aware] Step 2: 计算优先级 (atlas + 当前 MAE)...")
        mutant_priorities = self._compute_mutant_priorities(mutant_maes)

        # Step 3: 选 top-K mutant 并分配配额
        print("[Mutant-Aware] Step 3: 按 priority 分配采样名额...")
        sampling_plan = self._allocate_mutant_aware_quota(mutant_priorities)

        n_target_mutants = len(sampling_plan)
        n_rgs_samples = sum(sampling_plan.values())
        global_budget = self._global_lhs_budget
        print(f"[Mutant-Aware] 定向采样: {n_target_mutants} 个突变体, {n_rgs_samples} 样本 | "
              f"全局 LHS: {global_budget} 样本")

        # Step 4: 生成全部候选
        all_params_list = []

        # Step 4a: 定向局部采样候选 - 复用 generate_local_lhs_tasks
        if sampling_plan:
            print(f"[Mutant-Aware] Step 4a: 调用 generate_local_lhs_tasks (perturbation={self.perturbation*100:.0f}%)...")
            rgs_tasks_dict = generate_local_lhs_tasks(
                target_mutants=sampling_plan,
                perturbation=self.perturbation
            )

            if rgs_tasks_dict:
                for m_name, overrides in rgs_tasks_dict.items():
                    p_vec = [overrides.get(k, _defaults[k]) for k in PARAM_KEYS]
                    all_params_list.append((f"MA_Epoch{current_epoch}_{m_name}", p_vec))
                print(f"[Mutant-Aware] 局部采样引擎返回 {len(rgs_tasks_dict)} 个候选")
            else:
                print("[Mutant-Aware] 局部采样引擎未生成任何样本")
                global_budget += n_rgs_samples  # 转给全局 LHS

        # Step 4b: 全局 LHS 采样候选 (保持多样性)
        if global_budget > 0:
            print(f"[Mutant-Aware] Step 4b: 全局 LHS 采样 (保持多样性)...")
            per_mutant = max(1, global_budget // len(mutant_rules))
            global_tasks_dict = generate_lhs_tasks(per_mutant)

            p_pool_list_g = []
            pool_names_g = []
            for m_name, overrides in global_tasks_dict.items():
                p_vec = [overrides.get(k, _defaults[k]) for k in PARAM_KEYS]
                p_pool_list_g.append(p_vec)
                pool_names_g.append(m_name)

            P_pool_g = np.array(p_pool_list_g)
            actual_pool_size_g = len(P_pool_g)
            global_num_add = min(global_budget, actual_pool_size_g)

            idx_sel = np.random.choice(actual_pool_size_g, global_num_add, replace=False)
            for idx in idx_sel:
                all_params_list.append((f"MA_Global_Epoch{current_epoch}_{pool_names_g[idx]}", P_pool_g[idx]))

            print(f"[Mutant-Aware] 全局 LHS 返回 {actual_pool_size_g} 个候选，随机选取 {global_num_add} 个")

        if not all_params_list:
            return None

        print(f"[Mutant-Aware] 合并候选总数: {len(all_params_list)}")

        # Step 5: 参数合理性校验 (复用 RGS)
        all_params_list = self._validate_param_ranges(all_params_list)

        # Step 6: ODE 物理验证 (复用 RGS)
        valid_results, valid_names = self._physics_validate_and_prune(all_params_list, dm)

        if not valid_results:
            return None

        # Step 7: 用真实 MAE 选取 (复用 RGS)
        selected_results, selected_names = self._select_by_real_mae(
            pl_module, dm, device, valid_results, valid_names, current_epoch)

        if not selected_results:
            return None

        # Step 8: 绘制采样对比图 (复用 RGS)
        sample_plots = self._plot_al_sample_comparison(
            selected_results, selected_names, dm, current_epoch)

        # Step 9: 组装广播数据
        return {
            'X': torch.stack([torch.tensor(r[0]) for r in selected_results]).float(),
            'Y': torch.stack([torch.tensor(r[1]) for r in selected_results]).float(),
            'P': torch.stack([torch.tensor(r[2]) for r in selected_results]).float(),
            'names': selected_names,
            'labels': [r[3] for r in selected_results],
            'sample_plots': sample_plots
        }

    def _compute_mutant_priorities(self, mutant_maes):
        """
        计算每个 Real Mutant 的优先级分数。

        priority = 0.5 * norm(lhs_MAE_min) + 0.3 * (1 - lhs_n_samples/500) + 0.2 * norm(current_real_mae)

        - lhs_MAE_min: LHS 覆盖不足量化 (机制 A)，越高越优先
        - lhs_n_samples: LHS 密度不足量化 (机制 B)，越少越优先
        - current_real_mae: 模型当前表现，越差越优先

        对于 atlas 中不存在的突变体，使用中性默认值。
        """
        # 收集 atlas 数据
        atlas_names = list(self._mutant_atlas.keys())
        lhs_mae_mins = np.array([self._mutant_atlas[n]['lhs_mae_min'] for n in atlas_names])
        lhs_n_samples = np.array([self._mutant_atlas[n]['lhs_n_samples'] for n in atlas_names])

        # 归一化函数
        def normalize(arr):
            lo, hi = np.min(arr), np.max(arr)
            if hi - lo < 1e-12:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        lhs_mae_norm = normalize(lhs_mae_mins)
        density_penalty = 1.0 - np.clip(lhs_n_samples / 500.0, 0, 1)  # 越少越接近 1

        # 构建 atlas 索引
        atlas_idx = {n: i for i, n in enumerate(atlas_names)}

        # 收集当前 MAE
        current_maes = np.array(list(mutant_maes.values()))
        current_mae_norm = normalize(current_maes)
        mae_idx = {n: i for i, n in enumerate(mutant_maes.keys())}

        # 计算优先级
        mutant_priorities = {}
        for name in mutant_maes.keys():
            if name in atlas_idx:
                i = atlas_idx[name]
                score_atlas_mae = 0.5 * lhs_mae_norm[i]
                score_density = 0.3 * density_penalty[i]
            else:
                # atlas 中不存在的突变体：中性默认值
                score_atlas_mae = 0.5 * 0.5
                score_density = 0.3 * 0.5

            j = mae_idx[name]
            score_current_mae = 0.2 * current_mae_norm[j]

            priority = score_atlas_mae + score_density + score_current_mae
            mutant_priorities[name] = priority

        # 打印 top-5
        sorted_p = sorted(mutant_priorities.items(), key=lambda x: x[1], reverse=True)
        print("[Mutant-Aware] Top-5 优先级突变体:")
        for name, pri in sorted_p[:5]:
            atlas_info = self._mutant_atlas.get(name, {})
            print(f"  - {name}: priority={pri:.4f} | "
                  f"lhs_MAE_min={atlas_info.get('lhs_mae_min', 'N/A')}, "
                  f"lhs_n={atlas_info.get('lhs_n_samples', 'N/A')}, "
                  f"cur_mae={mutant_maes[name]:.4f}")

        return mutant_priorities

    def _allocate_mutant_aware_quota(self, mutant_priorities):
        """
        按 priority 加权分配采样配额。

        规则：
        - 90% 预算给定向 RGS (按 priority 加权)，10% 给全局 LHS (保持多样性)
        - 筛选高优先级突变体：lhs_MAE_min > 0.5 或 lhs_n_samples < 300
        - 聚焦 top-K = max(20, 30% of poor mutants)
        - 每个 mutant 至少 30 样本，最多占总预算的 30% (避免单一家族垄断)
        """
        total_budget = self.num_add
        rgs_budget = int(total_budget * 0.9)
        global_budget = total_budget - rgs_budget

        # 筛选高优先级突变体 (满足任一阈值)
        poor_mutants = {}
        for name, priority in mutant_priorities.items():
            atlas_info = self._mutant_atlas.get(name, {})
            lhs_mae_min = atlas_info.get('lhs_mae_min', 0.0)
            lhs_n = atlas_info.get('lhs_n_samples', 500)

            # 阈值判断：lhs_MAE_min > 0.5 或 N_LHS < 300
            is_high_mae = lhs_mae_min > 0.5
            is_low_density = lhs_n < 300

            if is_high_mae or is_low_density:
                poor_mutants[name] = priority

        if not poor_mutants:
            # 没有满足阈值的突变体，全部预算给全局 LHS
            print("[Mutant-Aware] 无满足阈值的高优先级突变体，全部预算给全局 LHS")
            self._global_lhs_budget = total_budget
            return {}

        # 聚焦 top-K 最优先的突变体
        sorted_poor = sorted(poor_mutants.items(), key=lambda x: x[1], reverse=True)
        top_k = max(20, int(len(sorted_poor) * 0.3))
        top_k = min(top_k, len(sorted_poor))
        focused_mutants = sorted_poor[:top_k]

        # 按 priority 加权分配配额
        total_priority = sum(p for _, p in focused_mutants)
        max_per_mutant = int(rgs_budget * 0.3)  # 单个突变体最多占 30%

        sampling_plan = {}
        for name, pri in focused_mutants:
            quota = max(30, int(rgs_budget * pri / total_priority))
            quota = min(quota, max_per_mutant)  # 上限钳制
            sampling_plan[name] = quota

        # 如果钳制导致总量不足，按比例重新分配剩余
        allocated = sum(sampling_plan.values())
        remaining = rgs_budget - allocated
        if remaining > 0 and focused_mutants:
            for name, pri in focused_mutants:
                if remaining <= 0:
                    break
                extra = min(remaining, max_per_mutant - sampling_plan.get(name, 0))
                if extra > 0:
                    sampling_plan[name] = sampling_plan.get(name, 0) + extra
                    remaining -= extra

        print(f"[Mutant-Aware] 高优先级突变体 ({len(poor_mutants)}个): "
              f"聚焦 top-{top_k} 分配 {sum(sampling_plan.values())} 名额")
        for name, pri in focused_mutants[:5]:
            atlas_info = self._mutant_atlas.get(name, {})
            print(f"  - {name}: priority={pri:.4f}, quota={sampling_plan.get(name, 0)} | "
                  f"lhs_MAE_min={atlas_info.get('lhs_mae_min', 'N/A')}, "
                  f"lhs_n={atlas_info.get('lhs_n_samples', 'N/A')}")
        if len(focused_mutants) > 5:
            print(f"  - ... 还有 {len(focused_mutants)-5} 个聚焦样本")

        self._global_lhs_budget = global_budget
        return sampling_plan

    def _validate_param_ranges(self, all_params_list):
        """
        RGS 参数合理性校验：在 ODE 验证前剔除明显异常的参数组合。

        校验规则：
        1. 非负性：所有参数必须 >= 0
        2. 上界约束：连续参数不超过默认值的 5 倍（避免数值爆炸）
        3. 特殊参数约束：mdt > 0, B0 > 0, C0 > 0 等关键参数有物理意义下界

        Args:
            all_params_list: [(name, p_vec), ...] 候选参数列表

        Returns:
            过滤后的候选参数列表
        """
        # 关键参数的物理下界（必须大于此值）
        param_lower_bounds = {
            'mdt': 10.0,      # 细胞周期时间至少 10 分钟
            'B0': 0.001,      # 生长速率
            'C0': 0.001,      # CLN3 系数
            'Dn3': 0.001,     # CLN3 解除抑制系数
            'ESP1T': 0.001,   # ESP1 总量
        }

        # 关键参数的相对上界（不超过默认值的多少倍）
        param_upper_multiplier = {
            'mdt': 5.0, 'B0': 5.0, 'C0': 5.0, 'Dn3': 5.0, 'Jn3': 5.0,
            'ksn2p': 5.0, 'ksn2pp': 5.0, 'kdn2': 5.0,
        }

        valid_list = []
        rejected_count = 0

        for name, p_vec in all_params_list:
            p_dict = dict(zip(PARAM_KEYS, p_vec)) if not isinstance(p_vec, dict) else p_vec
            is_valid = True

            # 1. 非负性校验
            for k, v in p_dict.items():
                if isinstance(v, (int, float, np.floating)) and v < 0:
                    is_valid = False
                    break

            # 2. 关键参数下界校验
            if is_valid:
                for k, lower in param_lower_bounds.items():
                    if k in p_dict:
                        v = p_dict[k]
                        if isinstance(v, (int, float, np.floating)) and v < lower:
                            is_valid = False
                            break

            # 3. 关键参数上界校验（相对默认值）
            if is_valid:
                for k, mult in param_upper_multiplier.items():
                    if k in p_dict and k in _defaults:
                        v = p_dict[k]
                        if isinstance(v, (int, float, np.floating)):
                            upper = _defaults[k] * mult
                            if v > upper:
                                is_valid = False
                                break

            if is_valid:
                valid_list.append((name, p_vec))
            else:
                rejected_count += 1

        if rejected_count > 0:
            print(f"[RGS] 参数合理性校验剔除 {rejected_count} 个异常参数组合 "
                  f"(剩余 {len(valid_list)}/{len(all_params_list)})")

        return valid_list

    def _evaluate_real_mutants(self, pl_module, dm, device):
        """在 Real Mutant 测试集上推理，计算每个样本的真实值 MAE (反归一化后)"""
        pl_module.eval()
        mutant_maes = {}
        
        test_dataset = dm.test_dataset_real
        # 获取 Real Mutant 名称列表
        real_names = []
        if hasattr(dm, '_mutant_names') and hasattr(dm, '_test_idx_real'):
            real_names = [str(dm._mutant_names[i]) for i in dm._test_idx_real]
        
        # 获取反归一化参数
        mean_y = dm.mean_y.cpu()  # (1, 39, 1)
        std_y = dm.std_y.cpu()    # (1, 39, 1)
        
        with torch.no_grad():
            for idx in range(len(test_dataset)):
                X, P, Y_true_norm = test_dataset[idx]
                X_batch = X.unsqueeze(0).to(device)
                P_batch = P.unsqueeze(0).to(device)
                
                pred_norm = pl_module.forward_ic_time(X_batch, P_batch).squeeze(0).cpu()
                
                # 反归一化到真实值空间
                pred_real = pred_norm * std_y + mean_y
                Y_true_real = Y_true_norm * std_y + mean_y
                
                # 计算真实值空间的 MAE
                mae = torch.mean(torch.abs(pred_real - Y_true_real)).item()
                
                name = real_names[idx] if idx < len(real_names) else f"Real_{idx}"
                mutant_maes[name] = mae
        
        pl_module.train()
        
        # 打印摘要
        maes = list(mutant_maes.values())
        n_poor = sum(1 for m in maes if m > self.mae_threshold)
        print(f"[RGS] Real Mutant 评估完成 (真实值空间): mean_mae={np.mean(maes):.4f}, "
              f"max={np.max(maes):.4f}, poor(>{self.mae_threshold})={n_poor}/{len(maes)}")
        
        self._real_mutant_scores = mutant_maes
        return mutant_maes

    def _allocate_rgs_quota(self, mutant_maes):
        """
        按 Real Mutant 的 MAE 分配 RGS 采样名额：
        - 只对 top-K 最差样本分配 RGS 名额 (聚焦，不撒胡椒面)
        - 其余预算分配给全局 LHS 采样以保持多样性
        - K = max(20, 差样本数的 30%)，避免预算过于分散
        """
        total_budget = self.num_add
        rgs_budget = int(total_budget * 0.9)   # 90% 给 RGS 局部采样
        global_budget = total_budget - rgs_budget  # 10% 给全局 LHS 保持多样性

        # 分层：差样本
        poor_mutants = {k: v for k, v in mutant_maes.items() if v > self.mae_threshold}

        sampling_plan = {}

        if poor_mutants:
            # 聚焦 top-K 最差样本，不撒胡椒面
            sorted_poor = sorted(poor_mutants.items(), key=lambda x: x[1], reverse=True)
            top_k = max(20, int(len(sorted_poor) * 0.3))
            top_k = min(top_k, len(sorted_poor))
            focused_mutants = sorted_poor[:top_k]

            # 按 MAE 的平方比例分配 (MAE² 越大，名额越多，强化聚焦)
            weights = {k: v ** 2 for k, v in focused_mutants}
            total_weight = sum(weights.values())
            for name, w in weights.items():
                quota = max(30, int(rgs_budget * w / total_weight))
                sampling_plan[name] = quota

            print(f"[RGS] 差样本 ({len(poor_mutants)}个, MAE>{self.mae_threshold}): "
                  f"聚焦 top-{top_k} 分配 {rgs_budget} 名额")
            for name, mae in focused_mutants[:5]:
                print(f"  - {name}: mae={mae:.4f}, quota={sampling_plan.get(name, 0)}")
            if len(focused_mutants) > 5:
                print(f"  - ... 还有 {len(focused_mutants)-5} 个聚焦样本")
        else:
            # 没有差样本时，全部预算给全局 LHS
            global_budget = total_budget

        # 好样本不分配 RGS 名额 (省下预算给全局 LHS)

        # 记录全局 LHS 预算 (在 _rgs_sample 中使用)
        self._global_lhs_budget = global_budget

        return sampling_plan

    def _infer_pattern(self, mutant_name):
        """从 mutant_rules 推断突变体的 Pattern 类型"""
        rules = mutant_rules.get(mutant_name, {})
        rules_str = str(rules)
        
        has_cdh1_ko = 'kscdh' in rules_str and '0.0' in rules_str
        has_apc_ts = 'ks20p' in rules_str and '0.0' in rules_str and 'ks20pp' in rules_str and '0.0' in rules_str
        has_multi_ko = sum(1 for v in rules.values() if v == 0.0) >= 3
        
        if (has_cdh1_ko and has_multi_ko) or has_apc_ts:
            return 'Pattern_C'
        elif has_multi_ko:
            return 'Pattern_C'
        else:
            return 'Pattern_B'

    def _select_by_real_mae(self, pl_module, dm, device, valid_results, valid_names, current_epoch):
        """
        用真实 MAE (模型预测 vs ODE 真值，反归一化后) 选取样本。
        策略：以 Real Mutant 的 MAE 为锚点，选取"跳一跳够得着"的样本。
        - MAE 上限 = max(Real Mutant 最大 MAE × 5, 10.0)：排除异常样本
        - MAE 下限 = Real Mutant 中位数 MAE × 0.5：排除太容易的样本
        - 在区间内按 MAE 降序选取
        """
        mean_y = dm.mean_y.cpu()  # (1, 39, 1)
        std_y = dm.std_y.cpu()    # (1, 39, 1)
        
        pl_module.eval()
        candidate_maes = []
        with torch.no_grad():
            for res in valid_results:
                X, y_uniform, p_vec, _ = res
                # 模型预测 (归一化空间)
                P_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).to(device)
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
                pred_norm = pl_module.forward_ic_time(X_tensor, P_tensor).squeeze(0).cpu()
                
                # 反归一化到真实值空间
                pred_real = (pred_norm * std_y + mean_y).squeeze(-1)  # (39, 500)
                y_real = torch.tensor(y_uniform, dtype=torch.float32)  # (39, 500) 已经是真实值
                
                # 真实值空间 MAE
                mae = torch.mean(torch.abs(pred_real - y_real)).item()
                candidate_maes.append(mae)
        pl_module.train()
        
        candidate_maes = np.array(candidate_maes)
        num_to_select = min(self.num_add, len(valid_results))
        
        # 以 Real Mutant 的 MAE 为锚点设定区间
        real_maes = np.array(list(self._real_mutant_scores.values())) if self._real_mutant_scores else np.array([1.0])
        real_max = np.max(real_maes)
        real_median = np.median(real_maes)
        
        mae_upper = max(real_max * 5.0, 10.0)    # 超过 Real 最差 5 倍 → 异常
        mae_lower = real_median * 0.5              # 低于 Real 中位数一半 → 太容易
        
        # 在 [mae_lower, mae_upper] 区间内选取 MAE 最大的
        mask = (candidate_maes >= mae_lower) & (candidate_maes <= mae_upper)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) < num_to_select:
            # 区间内不够，逐步放宽上限
            for multiplier in [10.0, 20.0, 50.0]:
                mae_upper_relaxed = max(real_max * multiplier, 50.0)
                mask = (candidate_maes >= mae_lower) & (candidate_maes <= mae_upper_relaxed)
                valid_indices = np.where(mask)[0]
                if len(valid_indices) >= num_to_select:
                    print(f"[RGS] MAE 上限放宽至 {mae_upper_relaxed:.1f} (Real max × {multiplier:.0f})")
                    break
        
        if len(valid_indices) == 0:
            # 最终降级: 取候选中位数附近
            mae_median = np.median(candidate_maes)
            mask = candidate_maes <= mae_median * 2
            valid_indices = np.where(mask)[0]
        
        if len(valid_indices) == 0:
            valid_indices = np.arange(len(valid_results))
        
        # 在有效区间内按 MAE 降序排列，选取前 num_to_select 个
        sorted_within = valid_indices[np.argsort(candidate_maes[valid_indices])[::-1]]
        idx_sel = sorted_within[:num_to_select]
        
        n_excluded_low = int(np.sum(candidate_maes < mae_lower))
        n_excluded_high = int(np.sum(candidate_maes > mae_upper))
        
        selected_results = [valid_results[i] for i in idx_sel]
        selected_names = [valid_names[i] for i in idx_sel]
        
        print(f"[RGS] 真实 MAE 选取: Real锚点 median={real_median:.2f}, max={real_max:.2f} | "
              f"区间=[{mae_lower:.2f}, {mae_upper:.2f}] | "
              f"排除太易={n_excluded_low}, 排除太难={n_excluded_high} | "
              f"选取 {len(idx_sel)}/{len(valid_results)} 个 "
              f"(MAE: {candidate_maes[idx_sel].min():.4f} ~ {candidate_maes[idx_sel].max():.4f})")
        
        return selected_results, selected_names

    # ==================================================================
    # 传统采样策略 (random, us, is, wrs, vessal, hggs)
    # ==================================================================
    def _legacy_sample(self, trainer, pl_module, dm, device, current_epoch):
        """原有的全局 LHS 采样逻辑"""
        per_mutant_samples = 166 
        print(f"[AL Manager] 调用 generate_lhs_tasks({per_mutant_samples}) 靶向生成候选海选池...")
        
        lhs_tasks_dict = generate_lhs_tasks(per_mutant_samples)
        
        p_pool_list = []
        pool_names = []
        for m_name, overrides in lhs_tasks_dict.items():
            p_vec = [overrides.get(k, _defaults[k]) for k in PARAM_KEYS]
            p_pool_list.append(p_vec)
            pool_names.append(m_name)
            
        P_pool = np.array(p_pool_list)
        actual_pool_size = len(P_pool)
        print(f"[AL Manager] LHS 引擎成功返回 {actual_pool_size} 个候选参数！")
        
        current_num_add = min(self.num_add, actual_pool_size)
        new_params_list = []

        if self.strategy in ['us', 'is', 'wrs', 'vessal']:
            pl_module.eval()
            with torch.no_grad():
                errors = []
                for i in range(0, actual_pool_size, 2000):
                    end_idx = min(i+2000, actual_pool_size)
                    P_batch = torch.tensor(P_pool[i:end_idx], dtype=torch.float32).to(device)
                    X_dummy = torch.zeros((P_batch.shape[0], 39, 2, 500)).to(device)
                    preds = pl_module.forward_ic_time(X_dummy, P_batch)
                    
                    err_batch = torch.var(preds, dim=-1).mean(dim=1).cpu().numpy()
                    errors.append(err_batch)
                errors = np.concatenate(errors)
            pl_module.train()

            if self.strategy == 'is':
                e_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
                probs = e_norm / e_norm.sum()
                idx_sel = np.random.choice(actual_pool_size, current_num_add, p=probs, replace=False)
            
            elif self.strategy == 'us':
                idx_sel = np.argsort(errors)[-current_num_add:]
            
            else:
                idx_sel = np.random.choice(actual_pool_size, current_num_add, replace=False)
                
        else:
            idx_sel = np.random.choice(actual_pool_size, current_num_add, replace=False)

        for idx in idx_sel:
            new_params_list.append((f"AL_Epoch{current_epoch}_{pool_names[idx]}", P_pool[idx]))

        # 物理验证与动态剔除
        valid_results, valid_names = self._physics_validate_and_prune(new_params_list, dm)
        
        if not valid_results:
            return None

        # 绘制采样对比图 (每种 Pattern 各一张)
        sample_plots = self._plot_al_sample_comparison(
            valid_results, valid_names, dm, current_epoch)

        return {
            'X': torch.stack([torch.tensor(r[0]) for r in valid_results]).float(),
            'Y': torch.stack([torch.tensor(r[1]) for r in valid_results]).float(),
            'P': torch.stack([torch.tensor(r[2]) for r in valid_results]).float(),
            'names': valid_names,
            'labels': [r[3] for r in valid_results],
            'sample_plots': sample_plots
        }

    # ==================================================================
    # 采样对比图绘制
    # ==================================================================
    def _plot_al_sample_comparison(self, valid_results, valid_names, dm, current_epoch):
        """
        为每种 Pattern (A, B, C) 各绘制一张对比图：
        红色虚线 = AL 采样结果的 ODE 轨迹 (真实值空间)
        蓝色实线 = 对应 Real Mutant 的 ODE 轨迹 (真实值空间)
        标题包含反归一化后的 MAE/MSE/RelL2/Corr 指标
        图布局、标题格式与 budding_yeast_v2 的 SimToRealVisualizerCallback 完全一致
        """
        from budding_yeast_v2.utils.metrics import calculate_metrics

        # 按 Pattern 分组 valid_results
        pattern_groups = {'Pattern_A': [], 'Pattern_B': [], 'Pattern_C': []}
        for i, res in enumerate(valid_results):
            pat_label = str(res[3])
            for pat_key in pattern_groups:
                if pat_key in pat_label:
                    pattern_groups[pat_key].append(i)
                    break
            else:
                pattern_groups['Pattern_B'].append(i)  # 默认归入 Pattern_B

        # 获取 Real Mutant 的真实值轨迹 (反归一化)
        mean_y = dm.mean_y.cpu()  # (1, 39, 1)
        std_y = dm.std_y.cpu()    # (1, 39, 1)
        test_dataset = dm.test_dataset_real
        real_names = []
        if hasattr(dm, '_mutant_names') and hasattr(dm, '_test_idx_real'):
            real_names = [str(dm._mutant_names[i]) for i in dm._test_idx_real]

        # 构建 Real Mutant 的 Pattern 查找表
        real_pattern_map = {}
        if hasattr(dm, 'pattern_labels') and hasattr(dm, '_test_idx_real'):
            for idx_pos, data_idx in enumerate(dm._test_idx_real):
                pat = str(dm.pattern_labels[data_idx]).split(':')[0]
                name = real_names[idx_pos] if idx_pos < len(real_names) else f"Real_{idx_pos}"
                real_pattern_map[name] = pat

        # 为每种 Pattern 选一个代表样本绘图
        sample_plots = {}
        t_max = getattr(dm, 't_max', 210)

        for pat_key, indices in pattern_groups.items():
            if not indices:
                continue

            # 选第一个该 Pattern 的样本
            sample_idx = indices[0]
            res = valid_results[sample_idx]
            sample_name = valid_names[sample_idx] if sample_idx < len(valid_names) else f"Sample_{sample_idx}"
            sample_y_uniform = res[1]  # (39, 500) 真实值空间 (ODE 求解器输出)

            # 找一个同 Pattern 的 Real Mutant 作为对比
            real_y_uniform = None
            real_name = "N/A"
            metrics_str = ""
            for rname, rpat in real_pattern_map.items():
                if rpat == pat_key:
                    ridx = real_names.index(rname) if rname in real_names else -1
                    if ridx >= 0:
                        _, _, Y_real_norm = test_dataset[ridx]
                        # Y_real_norm: (39, 500), std_y/mean_y: (1, 39, 1) → 广播后 squeeze 得 (39, 500)
                        real_y_uniform = (Y_real_norm * std_y + mean_y).squeeze(0).numpy()
                        real_name = rname

                        # 计算真实值空间的指标
                        sample_tensor = torch.tensor(sample_y_uniform).unsqueeze(0).float()  # (1, 39, 500)
                        real_tensor = torch.tensor(real_y_uniform).unsqueeze(0).float()      # (1, 39, 500)
                        metrics = calculate_metrics(sample_tensor, real_tensor)
                        metrics_str = (f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4e} "
                                      f"| Rel L2: {metrics['Relative L2']:.4f} | Corr: {metrics['Correlation']:.4f}")
                        break

            # 绘制 5×8 子图，与 SimToRealVisualizerCallback 格式一致
            fig, axes = plt.subplots(5, 8, figsize=(28, 16))
            axes = axes.flatten()
            t = np.linspace(0, t_max, sample_y_uniform.shape[1])

            for var_idx in range(39):
                ax = axes[var_idx]
                # AL 采样轨迹 (红色虚线)
                ax.plot(t, sample_y_uniform[var_idx], 'r--', label='AL Sample', linewidth=2)
                # Real Mutant 轨迹 (蓝色实线)
                if real_y_uniform is not None:
                    ax.plot(t, real_y_uniform[var_idx], 'b-', label='Real Mutant', linewidth=2)

                var_name = VARIABLE_NAMES[var_idx] if var_idx < len(VARIABLE_NAMES) else f"Var {var_idx}"
                ax.set_title(f'{var_idx}: {var_name}', fontweight='bold', fontsize=10)
                ax.set_xlabel('Time (min)')
                if var_idx == 0:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            for i in range(39, len(axes)):
                axes[i].set_visible(False)

            title_str = (f"[{pat_key}] AL Sample vs Real Mutant (Epoch {current_epoch + 1})\n"
                        f"Sample: {sample_name} | Real: {real_name}\n"
                        f"{metrics_str}")
            plt.suptitle(title_str, fontsize=18, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.93])

            sample_plots[f'{pat_key}_Epoch{current_epoch}'] = fig

        return sample_plots

    # ==================================================================
    # 公共工具方法
    # ==================================================================
    def _physics_validate_and_prune(self, new_params_list, dm):
        """ODE 物理验证 + 动态浓度剔除，返回 (valid_results, valid_names)"""
        print(f"[AL Manager] 调用 solve_ivp 物理验证 {len(new_params_list)} 个候选细胞...")
        valid_results = []
        valid_names = []
        
        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = [(name, executor.submit(solve_single_mutant_bridge, name, p))
                       for name, p in new_params_list]
            done = 0
            n_total = len(futures)
            for name, f in futures:
                try:
                    res = f.result(timeout=SIM_TIMEOUT + 30)
                except FutureTimeoutError:
                    print(f"  ⚠️ 物理验证样本 {name} 进程级超时,跳过")
                    res = None
                if res is not None:
                    valid_results.append(res)
                    valid_names.append(name)
                done += 1
                if done % max(1, n_total // 10) == 0 or done == n_total:
                    print(f"  物理验证进度: {done}/{n_total} ({100*done//n_total}%)")
        
        if len(valid_results) > 0:
            Y_real_norm = dm.test_dataset_real.tensors[2].cpu()
            std_y_cpu = dm.std_y.cpu()
            mean_y_cpu = dm.mean_y.cpu()
            Y_real_raw = Y_real_norm * std_y_cpu + mean_y_cpu
            
            real_envelope = Y_real_raw.max(dim=-1)[0].max(dim=0)[0].numpy()
            dynamic_thresholds = np.maximum(real_envelope * 2.0, 2.0)
            
            pruned_results = []
            pruned_names = []
            concentration_drop_count = 0
            
            for i, res in enumerate(valid_results):
                y_uniform = res[1] 
                lhs_max_vals = np.max(y_uniform, axis=1)
                
                is_monster = False
                for var_idx in range(1, 39):
                    if lhs_max_vals[var_idx] > dynamic_thresholds[var_idx]:
                        is_monster = True
                        break 
                        
                if is_monster:
                    concentration_drop_count += 1
                else:
                    pruned_results.append(res)
                    pruned_names.append(valid_names[i])
                    
            if concentration_drop_count > 0:
                print(f"[AL Manager] 动态浓度剔除系统销毁了 {concentration_drop_count} 个动态超限细胞！")
            
            valid_results = pruned_results
            valid_names = pruned_names

        if len(valid_results) > 0:
            print(f"[AL Manager] 最终存活并允许入库的细胞: {len(valid_results)} 个")

        return valid_results, valid_names

    # ==================================================================
    # al4pde 框架: 频率/数量机制 (对齐 al4pde 原始 ExponentialSchedule)
    # ==================================================================
    def _is_al4pde_strategy(self):
        """判断是否为 al4pde 框架策略 (使用 ExponentialSchedule 数量机制 + 末轮不选)"""
        return self.strategy in ('al4pde_random', 'al4pde_uncertainty',
                                 'al4pde_diversity', 'maal', 'al4pde_boundary', 'al4pde_mass')

    def _al4pde_scheduled_num_add(self):
        """al4pde v2.3 外层重训式: 每次采集固定追加 _al4pde_per_round_add 个 (76000 / 4 = 19000)。

        (v2.2 已作废的递增式 1000/3000/4000 不再使用。)
        """
        return getattr(self, '_al4pde_per_round_add', 1000)

    # ==================================================================
    # al4pde 框架: 4 种 AL 策略 (从 v5 候选池查询)
    # ==================================================================
    def _al4pde_random_sample(self, trainer, pl_module, dm, device, current_epoch):
        """al4pde Random: 从候选池随机选 scheduled_num_add 个样本 (ExponentialSchedule)"""
        if self._pool_size == 0:
            print("[al4pde_random] 候选池已空, 跳过本轮 AL")
            return None
        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        sel_idx = np.random.choice(self._pool_size, n_select, replace=False)
        print(f"[al4pde_random] 随机选取 {n_select} 个样本 (池剩余: {self._pool_size})")
        return self._build_broadcast_from_pool(sel_idx, dm)

    def _al4pde_uncertainty_sample(self, trainer, pl_module, dm, device, current_epoch):
        """al4pde Uncertainty: MC dropout top_k 选取 (复用 _compute_mc_dropout_uncertainty)"""
        if self._pool_size == 0:
            print("[al4pde_uncertainty] 候选池已空, 跳过本轮 AL")
            return None
        print(f"[al4pde_uncertainty] 计算 MC dropout 不确定性 (池大小: {self._pool_size})...")
        pool_indices = np.arange(self._pool_size)
        # 复用已有 MC dropout 实现 (n_mc_samples=5 加速)
        unc_scores = self._compute_mc_dropout_uncertainty(pl_module, device, pool_indices, dm,
                                                          n_mc_samples=5)
        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        sel_idx = np.argsort(unc_scores)[-n_select:]
        print(f"[al4pde_uncertainty] top_k 选取 {n_select} 个样本 (unc range: "
              f"{unc_scores[sel_idx].min():.4f}~{unc_scores[sel_idx].max():.4f})")
        return self._build_broadcast_from_pool(sel_idx, dm)

    def _al4pde_diversity_sample(self, trainer, pl_module, dm, device, current_epoch):
        """al4pde Diversity: bmdal_reg MaxDistSelectionMethod (轨迹形状感知特征)

        改进: 原版用 spatial_mean(39维) 丢失时间形状信息,
        现用形状感知特征(332维): MASS全曲线 + 一阶差分 + 全变量统计 + 参数,
        各组 z-score 标准化后 MaxDist 选择, 兼顾曲线形状和参数多样性。
        """
        if self._pool_size == 0:
            print("[al4pde_diversity] 候选池已空, 跳过本轮 AL")
            return None
        print(f"[al4pde_diversity] 计算轨迹形状感知特征 (MASS pattern + 全变量统计 + 参数)...")
        pool_features = self._compute_pool_features_shape_aware(pl_module, dm, device)
        train_features = self._compute_train_features_shape_aware(pl_module, dm, device)
        # 用 train 统计量分组 z-score 标准化 (防止参数等高方差组主导距离)
        pool_features, train_features = self._standardize_shape_features(pool_features, train_features)
        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        print(f"[al4pde_diversity] bmdal_reg MaxDistSelection 选取 {n_select} 个样本 "
              f"(shape_feat: {pool_features.shape}, train_feat: {train_features.shape})...")
        sel_idx = self._bmdal_max_dist_select(pool_features, train_features, n_select)
        return self._build_broadcast_from_pool(sel_idx, dm)

    def _maal_sample(self, trainer, pl_module, dm, device, current_epoch):
        """
        MAAL (Mutant-Aware Active Learning): diversity + target priority 加权。

        acquisition_score = (1 - target_weight) * diversity + target_weight * target_priority
        - diversity: 候选样本到训练集的最小欧氏距离 (轨迹形状感知特征: MASS pattern + 全变量统计 + 参数)
        - target_priority: 完全基于训练集计算 (零数据泄露)
            = 0.5*同家族MC dropout不确定性 + 0.3*候选到训练集同家族参数距离 + 0.2*家族覆盖度
          不再访问 dm.test_dataset_real, 严格遵守数据隔离原则。

        权重通过 --al_target_weight 控制 (默认 0.4), diversity 权重 = 1 - target_weight。
        """
        if self._pool_size == 0:
            print("[MAAL] 候选池已空, 跳过本轮 AL")
            return None

        # Step 1: 对候选池和训练集做 forward, 得到轨迹形状感知特征
        print(f"[MAAL] Step 1: 计算轨迹形状感知特征 (池: {self._pool_size})...")
        pool_features = self._compute_pool_features_shape_aware(pl_module, dm, device)
        train_features = self._compute_train_features_shape_aware(pl_module, dm, device)
        # 用 train 统计量分组 z-score 标准化
        pool_features, train_features = self._standardize_shape_features(pool_features, train_features)

        # Step 2: 计算 diversity score (候选到训练集的最小距离)
        print("[MAAL] Step 2: 计算 diversity score (min distance to train)...")
        diversity_scores = self._compute_diversity_scores(pool_features, train_features)
        diversity_scores = self._normalize_array(diversity_scores)

        # Step 3: 计算 target_priority (完全基于训练集: 同家族MC不确定性 + 参数距离 + 覆盖度)
        # 数据隔离: 不访问 dm.test_dataset_real, 零数据泄露
        print("[MAAL] Step 3: 计算 target_priority (训练集版: 同家族MC不确定性 + 参数距离 + 覆盖度)...")
        family_uncertainty = self._compute_train_family_uncertainty(pl_module, dm, device)
        target_priorities = self._map_family_priority_train_based(family_uncertainty, dm)
        target_priorities = self._normalize_array(target_priorities)

        # Step 4: 加权 acquisition
        div_w = 1.0 - self.target_weight
        tgt_w = self.target_weight
        acquisition = div_w * diversity_scores + tgt_w * target_priorities
        print(f"[MAAL] Step 4: 加权 acquisition (diversity={div_w:.2f}, target={tgt_w:.2f}) | "
              f"score range: {acquisition.min():.4f}~{acquisition.max():.4f}")

        # Step 5: top_k 选取
        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        sel_idx = np.argsort(acquisition)[-n_select:]
        print(f"[MAAL] Step 5: top_k 选取 {n_select} 个样本")
        return self._build_broadcast_from_pool(sel_idx, dm)

    def _al4pde_boundary_sample(self, trainer, pl_module, dm, device, current_epoch):
        """
        al4pde Boundary: 借鉴 lhs_v1_origin_v5_boundary.py 的 0 值邻域采样思想。

        老师观点: v5_boundary 的 target 采样只做 boundary 采样 (0 邻域覆盖) 是合理的,
        AL 算法可以借鉴这种做法。本策略将 boundary 思想从数据生成阶段迁移到 AL 选择阶段:
        优先选择突变参数接近 0 边界 (敲除边界) 的候选样本。

        acquisition = (1 - boundary_weight) * diversity + boundary_weight * boundary_proximity

        - diversity: 候选样本到训练集的最小欧氏距离 (轨迹形状感知特征: MASS pattern + 全变量统计 + 参数)
        - boundary_proximity: 候选样本突变参数接近 0 边界的程度 [0, 1]
            对每个突变参数 p (来自父突变体的 mutant_rules):
            - val: 候选参数值 (未归一化)
            - center_val: 父突变体的突变值 (mutant_rules[parent][p], KO 时为 0)
            - ref_val: 默认参数值 (_defaults[p])
            - u_bound = ref_val * (1+pert)  if zero_ko else center_val * (1+pert)
              (与 lhs_v1_origin_v5_boundary.py 的 boundary 采样上界完全一致)
            - proximity_p = clip(1 - val / u_bound, 0, 1)
              val=0 → proximity=1 (纯 boundary, 完全敲除)
              val=u_bound → proximity=0 (过表达端)
            boundary_proximity = mean(proximity_p) 跨所有突变参数
            越高 → 越接近 0 边界 → 越优先选择

        权重通过 --al_target_weight 控制 (默认 0.4), boundary_weight = target_weight。
        数据隔离: 仅用 _pool_P_real 和 _pool_mutant_names (候选池自带), 不访问测试集。
        """
        if self._pool_size == 0:
            print("[al4pde_boundary] 候选池已空, 跳过本轮 AL")
            return None

        # Step 1: 计算轨迹形状感知特征 (与 diversity/maal 一致)
        print(f"[al4pde_boundary] Step 1: 计算轨迹形状感知特征 (池: {self._pool_size})...")
        pool_features = self._compute_pool_features_shape_aware(pl_module, dm, device)
        train_features = self._compute_train_features_shape_aware(pl_module, dm, device)
        pool_features, train_features = self._standardize_shape_features(pool_features, train_features)

        # Step 2: 计算 diversity score (候选到训练集的最小距离)
        print("[al4pde_boundary] Step 2: 计算 diversity score (min distance to train)...")
        diversity_scores = self._compute_diversity_scores(pool_features, train_features)
        diversity_scores = self._normalize_array(diversity_scores)

        # Step 3: 计算 boundary proximity (0 值邻域接近度, 借鉴 v5_boundary)
        print("[al4pde_boundary] Step 3: 计算 boundary proximity (0 值邻域接近度)...")
        boundary_scores = self._compute_boundary_proximity_scores()
        boundary_scores = self._normalize_array(boundary_scores)

        # Step 4: 加权 acquisition
        div_w = 1.0 - self.target_weight
        bnd_w = self.target_weight
        acquisition = div_w * diversity_scores + bnd_w * boundary_scores
        print(f"[al4pde_boundary] Step 4: 加权 acquisition (diversity={div_w:.2f}, boundary={bnd_w:.2f}) | "
              f"score range: {acquisition.min():.4f}~{acquisition.max():.4f}")

        # Step 5: top_k 选取
        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        sel_idx = np.argsort(acquisition)[-n_select:]
        print(f"[al4pde_boundary] Step 5: top_k 选取 {n_select} 个样本")
        return self._build_broadcast_from_pool(sel_idx, dm)

    def _compute_boundary_proximity_scores(self, perturbation=0.1):
        """计算候选池每个样本的 boundary proximity (借鉴 lhs_v1_origin_v5_boundary.py 0 值邻域采样)

        对每个候选样本, 计算其突变参数接近 0 边界 (敲除边界) 的程度:
        1. 从 _pool_mutant_names 提取父突变体名 (split '_LHS_')
        2. 从 mutant_rules 获取父突变体的突变参数 (mutated_params)
        3. 对每个突变参数, 按边界采样的上界公式计算接近度:
           - val = pool_P_real[i, param_idx]  (候选值, 未归一化)
           - center_val = mutant_rules[parent][param]  (父突变值, KO=0)
           - ref_val = _defaults[param]  (默认值)
           - zero_ko = (center_val == 0.0 and ref_val > 0.0)
           - u_bound = ref_val * (1+pert)  if zero_ko else center_val * (1+pert)
             (与 lhs_v1_origin_v5_boundary.py L1425-1432 完全一致)
           - proximity_p = clip(1 - val / u_bound, 0, 1)
        4. boundary_score = mean(proximity_p) 跨所有突变参数

        数据隔离: 仅访问候选池自带信息 (_pool_P_real, _pool_mutant_names) 和
                  全局静态查表数据 (mutant_rules, _defaults), 不访问测试集。

        Args:
            perturbation: 统一扰动幅度 (默认 0.1, v5_boundary 多数家族的保守估计;
                          boundary 采样的上界 = val * (1+pert), pert 越大上界越宽容)

        Returns:
            scores: (N,) numpy array, 每个候选样本的 boundary proximity [0, 1]
        """
        scores = np.zeros(self._pool_size)
        pool_P_real = self._pool_P_real.numpy() if hasattr(self._pool_P_real, 'numpy') else np.array(self._pool_P_real)

        # 构建 param_name → index 映射 (PARAM_KEYS 顺序, 与 _pool_P_real 对齐)
        param_to_idx = {k: i for i, k in enumerate(PARAM_KEYS)}

        n_matched = 0      # 成功匹配到 mutant_rules 的候选数
        n_boundary_high = 0  # boundary_score > 0.5 的候选数 (接近 0 边界)

        for i in range(self._pool_size):
            name = str(self._pool_mutant_names[i])
            # 候选池名格式: "73_cdc20_ts_LHS_T001" → 父突变体 "73_cdc20_ts"
            parent_name = name.split('_LHS_')[0]

            if parent_name not in mutant_rules:
                scores[i] = 0.5  # 默认中性值 (无突变规则, 如 WT 或未匹配)
                continue

            rules = mutant_rules[parent_name]
            mutated_params = list(rules.keys())

            proximities = []
            for p_name in mutated_params:
                if p_name not in param_to_idx:
                    continue  # 跳过非参数键 (如 is_mutant_104, 在 PARAM_KEYS 中被排除)

                p_idx = param_to_idx[p_name]
                val = float(pool_P_real[i, p_idx])
                center_val = float(rules[p_name])
                ref_val = float(_defaults.get(p_name, 0.0))

                # 严格对齐 lhs_v1_origin_v5_boundary.py L1422-1432 的 boundary 采样上界
                zero_ko = (center_val == 0.0 and ref_val > 0.0)
                if zero_ko:
                    # KO 突变: 用默认值作为上界参考 (覆盖 0 到过表达)
                    u_bound = ref_val * (1.0 + perturbation)
                else:
                    # 非 KO 突变: 覆盖从完全敲除 (0) 到过表达 (1+pert)
                    u_bound = center_val * (1.0 + perturbation)

                if u_bound <= 1e-12:
                    # 极端情况: 父突变值和默认值都为 0, 用 val 本身衡量
                    proximity = 1.0 if val < 1e-12 else 0.0
                else:
                    proximity = float(np.clip(1.0 - val / u_bound, 0.0, 1.0))

                proximities.append(proximity)

            if proximities:
                scores[i] = float(np.mean(proximities))
                n_matched += 1
                if scores[i] > 0.5:
                    n_boundary_high += 1
            else:
                scores[i] = 0.5  # 默认中性值

        print(f"[al4pde_boundary] boundary proximity: matched {n_matched}/{self._pool_size} 候选 | "
              f"high(>0.5): {n_boundary_high} | "
              f"score range: {scores.min():.4f}~{scores.max():.4f}, mean={scores.mean():.4f}")
        return scores

    def _al4pde_mass_sample(self, trainer, pl_module, dm, device, current_epoch):
        """
        al4pde MASS: 借鉴 data/mass_dataset_generator.py 的"通路→机制→严重程度"分层采样思想,
        将其从"数据生成阶段"迁移到"AL 选择阶段": 优先选择生物学上高频、机制典型的候选样本。

        acquisition = (1 - mass_weight) * diversity + mass_weight * mass_priority
        - diversity: 候选到训练集的最小欧氏距离 (轨迹形状感知特征: MASS pattern + 全变量统计 + 参数),
          防止选样冗余
        - mass_priority: 生物学分层优先级 [0,1], 每个候选按其扰动参数的"通路频率 × 机制频率"打分
            * 通路频率: 参数所属绑定组对应通路的 high/medium/low 频率 (6/3/1), 取参数最高频通路
            * 机制频率: 由候选参数相对 WT 默认值的扰动模式推断机制类 (loss/gain/partial),
              权重取自 Chen 2004 操作实例统计 (KO/GAL-OE/ts)
          越高 → 越像生物学高频、机制典型的突变 → 越优先选择

        权重通过 --al_mass_weight 控制 (默认 0.5), diversity 权重 = 1 - mass_weight。
        数据隔离: 仅用 _pool_P_real / _pool_mutant_names (候选池自带) + 静态生物学先验表, 不访问测试集。
        """
        if self._pool_size == 0:
            print("[al4pde_mass] 候选池已空, 跳过本轮 AL")
            return None

        # Step 1: 计算轨迹形状感知特征 (与 diversity/maal/boundary 一致)
        print(f"[al4pde_mass] Step 1: 计算轨迹形状感知特征 (池: {self._pool_size})...")
        pool_features = self._compute_pool_features_shape_aware(pl_module, dm, device)
        train_features = self._compute_train_features_shape_aware(pl_module, dm, device)
        pool_features, train_features = self._standardize_shape_features(pool_features, train_features)

        # Step 2: 计算 diversity score (候选到训练集的最小距离)
        print("[al4pde_mass] Step 2: 计算 diversity score (min distance to train)...")
        diversity_scores = self._compute_diversity_scores(pool_features, train_features)
        diversity_scores = self._normalize_array(diversity_scores)

        # Step 3: 计算 MASS 生物学分层优先级 (通路频率 × 机制频率)
        print("[al4pde_mass] Step 3: 计算 MASS 生物学分层优先级 (通路频率 × 机制频率)...")
        mass_scores = self._compute_mass_priority_scores()
        mass_scores = self._normalize_array(mass_scores)

        # Step 4: 加权 acquisition
        div_w = 1.0 - self.mass_weight
        mass_w = self.mass_weight
        acquisition = div_w * diversity_scores + mass_w * mass_scores
        print(f"[al4pde_mass] Step 4: 加权 acquisition (diversity={div_w:.2f}, mass={mass_w:.2f}) | "
              f"score range: {acquisition.min():.4f}~{acquisition.max():.4f}")

        # Step 5: top_k 选取
        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        sel_idx = np.argsort(acquisition)[-n_select:]
        print(f"[al4pde_mass] Step 5: top_k 选取 {n_select} 个样本")
        return self._build_broadcast_from_pool(sel_idx, dm)

    def _compute_mass_priority_scores(self):
        """计算候选池每个样本的 MASS 生物学分层优先级 (借鉴 mass_dataset_generator.py)。

        向量化实现:
          1. 收集 MASS 相关参数 (在 _MASS_PARAM_PATHWAY_FREQ 中的参数) 的列索引、通路频率、WT 默认值
          2. 对每个候选样本, 找出相对 WT 默认值被扰动的参数 (|val-default| > rel_tol*max(|default|,eps))
          3. 对每个扰动参数按扰动模式推断粗机制类, 取归一化机制频率:
             - loss  (val≈0, default>0)      → KO
             - gain  (val>default*(1+tol))   → GAL-OE
             - partial (其余扰动, 部分残余)   → ts
          4. contrib = (pathway_freq/6) * mech_prob; score = mean(contrib) 跨所有扰动参数 (无扰动 → 0)

        数据隔离: 仅访问 _pool_P_real + 静态先验表, 不访问测试集。
        """
        pool_P_real = self._pool_P_real.numpy() if hasattr(self._pool_P_real, 'numpy') else np.array(self._pool_P_real)
        param_to_idx = {k: i for i, k in enumerate(PARAM_KEYS)}

        # 机制权重归一化 (转为概率)
        mech_total = sum(_MASS_MECH_WEIGHTS.values())
        ko_p = _MASS_MECH_WEIGHTS['KO'] / mech_total
        oe_p = _MASS_MECH_WEIGHTS['GAL-OE'] / mech_total
        ts_p = _MASS_MECH_WEIGHTS['ts'] / mech_total
        rel_tol = 0.05  # 相对容差: |val-default| > 5%*max(|default|,eps) 视为扰动

        # 收集 MASS 相关参数的列索引 / 通路频率 / 默认值
        cols, pw_freqs, defs = [], [], []
        for p_name, pw in _MASS_PARAM_PATHWAY_FREQ.items():
            if p_name in param_to_idx:
                cols.append(param_to_idx[p_name])
                pw_freqs.append(pw / 6.0)      # 通路频率归一化到 [0,1]
                defs.append(float(_defaults.get(p_name, 0.0)))
        if not cols:
            return np.full(self._pool_size, 0.5)
        cols = np.asarray(cols)
        pw_freqs = np.asarray(pw_freqs)
        defs = np.asarray(defs)

        P = pool_P_real[:, cols]                              # (N, K)
        perturbed = np.abs(P - defs) > (rel_tol * np.maximum(np.abs(defs), 1e-9))
        loss = (P <= 1e-9) & (defs > 1e-9)
        gain = P > defs * (1.0 + rel_tol)
        partial = perturbed & (~loss) & (~gain)
        mech = loss * ko_p + gain * oe_p + partial * ts_p     # (N, K)
        contrib = (pw_freqs[None, :] * mech) * perturbed      # (N, K); 未扰动位为 0
        n_pert = perturbed.sum(axis=1)
        scores = contrib.sum(axis=1) / np.maximum(n_pert, 1)  # mean over perturbed params

        n_scored = int((n_pert > 0).sum())
        print(f"[al4pde_mass] mass priority: scored {n_scored}/{self._pool_size} 候选 | "
              f"range: {scores.min():.4f}~{scores.max():.4f}, mean={scores.mean():.4f}")
        return scores

    # ====== al4pde 框架辅助方法 ======
    def _compute_pool_features_spatial_mean(self, pl_module, dm, device, batch_size=200):
        """对候选池做 forward, 返回轨迹 spatial_mean 特征 (al4pde 默认 traj_transform)

        features(c) = model(X(c), params(c)).mean(time_dim) -> (N, 39)
        """
        pl_module.eval()
        features = []
        with torch.no_grad():
            for i in range(0, self._pool_size, batch_size):
                end_idx = min(i + batch_size, self._pool_size)
                X_batch = self._get_pool_X_batch(slice(i, end_idx)).to(device)
                P_batch = self._pool_P[i:end_idx].to(device)
                # forward: (B, 39, 2, T) + (B, 141) -> (B, 39, T)
                preds = pl_module.forward_ic_time(X_batch, P_batch)
                # spatial_mean: 对时间维取 mean -> (B, 39)
                feat = preds.mean(dim=2).cpu()
                features.append(feat)
        pl_module.train()
        return torch.cat(features, dim=0)

    def _compute_train_features_spatial_mean(self, pl_module, dm, device, batch_size=200):
        """对训练集做 forward, 返回轨迹 spatial_mean 特征"""
        train_X, train_P, _ = dm.train_dataset_lhs.tensors
        n_train = len(train_P)
        pl_module.eval()
        features = []
        with torch.no_grad():
            for i in range(0, n_train, batch_size):
                end_idx = min(i + batch_size, n_train)
                X_batch = train_X[i:end_idx].to(device)
                P_batch = train_P[i:end_idx].to(device)
                preds = pl_module.forward_ic_time(X_batch, P_batch)
                feat = preds.mean(dim=2).cpu()
                features.append(feat)
        pl_module.train()
        return torch.cat(features, dim=0)

    # ====== 轨迹形状感知特征 (改进版: 替代 spatial_mean) ======
    # 用户洞察: 参数微小差异可能导致曲线形状剧变 (ODE 非线性 + Goldbeter-Koshland 项),
    # diversity 不能只看参数距离或轨迹均值, 需结合 mass pattern (增长曲线形状) 综合考量。
    # 特征组成 (各组独立 z-score 标准化, 用 train 统计量, 防止量纲主导):
    #   1. MASS 轨迹降采样 (32 点): 捕捉增长模式 (指数/线性/GAL诱导阶跃等 Pattern)
    #   2. MASS 一阶差分统计 (3 维): mean/std/max, 区分增长模式 (线性=std小, 指数=std大)
    #   3. 全变量轨迹统计 (39×4=156 维): mean/std/max/min, 捕捉整体形状
    #   4. 参数空间 (141 维): 归一化参数, 辅助参数层多样性
    # 总维度: 32 + 3 + 156 + 141 = 332
    _SHAPE_FEAT_GROUP_SIZES = [32, 3, 39, 39, 39, 39, 141]

    # ====== v2.2 方案 A: 分组特征权重 (al4pde_diversity/maal/boundary 共用) ======
    # 背景: v2.0/v2.1 实验显示 diversity 在 lhs_test 上 Rel L2=0.099 (最佳) 但 real_mutants
    #   上 0.361 (次差), 典型过拟合到 LHS 形状分布。形状特征 (尤其 MASS 32 维) 主导了
    #   MaxDist 选择, 导致选出的样本形状独特但参数空间覆盖不足。
    # 改进: z-score 标准化后对各组施加权重, 增强参数空间多样性敏感度:
    #   - 参数向量 (141 维) ×2.5: 让 MaxDist 更关注参数空间覆盖
    #   - MASS 轨迹降采样 (32 维) ×0.5: 降低形状特征对选择的过度主导
    #   - 其他组 (差分统计/全变量统计) 权重保持 1.0
    # 注意: 权重在 z-score 标准化之后施加, 不影响 mean/std 统计量的计算。
    _SHAPE_FEAT_GROUP_WEIGHTS = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5]

    def _extract_shape_features(self, preds, params):
        """从模型预测轨迹和参数中提取形状感知特征 (raw, 未标准化)

        Args:
            preds: (B, 39, T) 模型预测轨迹
            params: (B, 141) 归一化参数
        Returns:
            features: (B, 332) raw 特征 (拼接, 未标准化)
        """
        B, V, T = preds.shape
        groups = []

        # 1. MASS 轨迹降采样 (变量 0, 用户关键洞察: mass pattern 决定曲线类别)
        mass_traj = preds[:, 0, :]  # (B, T)
        mass_ds = F.interpolate(
            mass_traj.unsqueeze(1), size=32, mode='linear', align_corners=False
        ).squeeze(1)  # (B, 32)
        groups.append(mass_ds)

        # 2. MASS 一阶差分统计 (区分指数/线性/阶跃增长)
        mass_diff = mass_traj[:, 1:] - mass_traj[:, :-1]  # (B, T-1)
        groups.append(torch.stack([
            mass_diff.mean(dim=1),     # 平均增长率
            mass_diff.std(dim=1),      # 增长率波动 (线性≈0, 指数>0)
            mass_diff.max(dim=1)[0],   # 最大增长 (捕捉 GAL 诱导阶跃)
        ], dim=1))  # (B, 3)

        # 3. 全变量轨迹统计 (捕捉整体形状, 不仅是均值)
        groups.append(preds.mean(dim=2))     # (B, 39)
        groups.append(preds.std(dim=2))      # (B, 39)
        groups.append(preds.max(dim=2)[0])   # (B, 39)
        groups.append(preds.min(dim=2)[0])   # (B, 39)

        # 4. 参数空间特征 (辅助: 参数层多样性, 已归一化)
        groups.append(params)  # (B, 141)

        return torch.cat(groups, dim=1)  # (B, 332)

    def _standardize_shape_features(self, pool_feat, train_feat):
        """用 train 统计量对 pool+train 特征做分组 z-score 标准化 + 分组权重

        各组独立标准化, 防止高方差组 (如参数) 主导距离。
        用 train 的 mean/std (而非 pool) 避免候选池分布偏移影响标准化基准。

        v2.2 方案 A: 标准化后施加 _SHAPE_FEAT_GROUP_WEIGHTS 权重,
        增强参数空间 (141 维) 敏感度 ×2.5, 降低 MASS 曲线 (32 维) 权重 ×0.5,
        缓解 diversity 过拟合到 LHS 形状分布的问题。

        Args:
            pool_feat: (N, 332) raw pool 特征
            train_feat: (M, 332) raw train 特征
        Returns:
            pool_std, train_std: 标准化+加权后的特征
        """
        assert pool_feat.shape[1] == sum(self._SHAPE_FEAT_GROUP_SIZES)
        assert len(self._SHAPE_FEAT_GROUP_SIZES) == len(self._SHAPE_FEAT_GROUP_WEIGHTS)
        idx = 0
        pool_parts, train_parts = [], []
        for size, weight in zip(self._SHAPE_FEAT_GROUP_SIZES, self._SHAPE_FEAT_GROUP_WEIGHTS):
            train_g = train_feat[:, idx:idx + size]
            mean = train_g.mean(dim=0, keepdim=True)
            std = train_g.std(dim=0, keepdim=True)
            # z-score 标准化后施加分组权重 (v2.2 方案 A)
            pool_std_g = (pool_feat[:, idx:idx + size] - mean) / (std + 1e-8)
            train_std_g = (train_g - mean) / (std + 1e-8)
            pool_parts.append(pool_std_g * weight)
            train_parts.append(train_std_g * weight)
            idx += size
        return torch.cat(pool_parts, dim=1), torch.cat(train_parts, dim=1)

    def _compute_pool_features_shape_aware(self, pl_module, dm, device, batch_size=200):
        """返回候选池的轨迹形状感知特征 (raw, 未标准化)

        改进: 优先用 ODE 预计算缓存 (真实轨迹), 无缓存时回退到模型预测。
        """
        if self._pool_shape_features_cached is not None:
            # ODE 预计算缓存 (初始化时已仿真, 真实轨迹)
            return self._pool_shape_features_cached

        # 回退: 模型预测 (适用于未预计算的策略, 如调试)
        print("[al4pde] 警告: 无 ODE 预计算缓存, 回退到模型预测 (不推荐)")
        pl_module.eval()
        features = []
        with torch.no_grad():
            for i in range(0, self._pool_size, batch_size):
                end_idx = min(i + batch_size, self._pool_size)
                X_batch = self._get_pool_X_batch(slice(i, end_idx)).to(device)
                P_batch = self._pool_P[i:end_idx].to(device)
                preds = pl_module.forward_ic_time(X_batch, P_batch)
                feat = self._extract_shape_features(preds, P_batch).cpu()
                features.append(feat)
        pl_module.train()
        return torch.cat(features, dim=0)

    def _compute_train_features_shape_aware(self, pl_module, dm, device, batch_size=200):
        """从训练集真实 Y 轨迹提取形状特征 (raw, 未标准化)

        改进: 训练集已有真实轨迹 (数据集中的 Y), 直接用, 不需要模型预测。
        """
        _, train_P, train_Y = dm.train_dataset_lhs.tensors
        n_train = len(train_P)
        features = []
        for i in range(0, n_train, batch_size):
            end_idx = min(i + batch_size, n_train)
            Y_batch = train_Y[i:end_idx]      # (B, 39, T) 真实轨迹 (归一化空间)
            P_batch = train_P[i:end_idx]      # (B, 141) 归一化参数
            # Y 是归一化的, 但形状特征在分组 z-score 后与归一化无关
            feat = self._extract_shape_features(Y_batch, P_batch).cpu()
            features.append(feat)
        return torch.cat(features, dim=0)

    def _bmdal_max_dist_select(self, pool_features, train_features, n_select):
        """用 bmdal_reg MaxDistSelectionMethod 选 batch (参考 al4pde distance_pool_based.py)"""
        from bmdal_reg.bmdal.feature_data import TensorFeatureData
        from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
        from bmdal_reg.bmdal.features import Features
        from bmdal_reg.bmdal.selection import MaxDistSelectionMethod

        n_features = pool_features.shape[1]
        train_ft = Features(IdentityFeatureMap(n_features=n_features),
                            TensorFeatureData(train_features))
        pool_ft = Features(IdentityFeatureMap(n_features=n_features),
                           TensorFeatureData(pool_features))
        sel_method = MaxDistSelectionMethod(pool_features=pool_ft, train_features=train_ft,
                                            sel_with_train=True)
        sel_idx = sel_method.select(n_select).detach().cpu().numpy()
        return sel_idx

    def _compute_diversity_scores(self, pool_features, train_features, chunk=500):
        """计算每个候选样本到训练集的最小欧氏距离 (diversity score)

        Args:
            pool_features: (N, D) torch.Tensor
            train_features: (M, D) torch.Tensor

        Returns:
            scores: (N,) numpy array, 每个候选样本到训练集的最近距离
        """
        scores = []
        for i in range(0, len(pool_features), chunk):
            p_chunk = pool_features[i:i + chunk]  # (chunk, D)
            # dist: (chunk, M)
            dist = torch.cdist(p_chunk, train_features)
            min_dist = dist.min(dim=1)[0]  # (chunk,)
            scores.append(min_dist)
        return torch.cat(scores, dim=0).numpy()

    def _map_family_priority(self, mutant_maes, dm=None):
        """将 Real Mutant MAE 映射到候选池样本的 target_priority (改进版)

        改进点 (解决原版"赢者通吃"问题):
        1. MAE 百分位排名 (权重 0.5): 用百分位替代原始 MAE, 避免极端值主导
        2. 参数空间距离 (权重 0.3): 候选样本到匹配 Real Mutant 的参数距离越近, priority 越高
        3. 家族覆盖度 (权重 0.2): 候选池中同家族样本数越少 (覆盖不足), priority 越高

        候选池样本的 mutant_name 格式: "73_cdc20_ts_LHS_T001"
        Real Mutant 的 mutant_name 格式: "73_cdc20_ts"
        匹配规则: 候选样本的 family 前缀 (split('_LHS_')[0]) 是 Real Mutant 名的前缀

        Args:
            mutant_maes: {mutant_name: mae} dict
            dm: 数据模块 (用于获取 Real Mutant 参数向量, 计算参数空间距离)

        Returns:
            priorities: (N,) numpy array, 每个候选样本的 target_priority
        """
        priorities = np.zeros(self._pool_size)
        # 预处理 mutant_maes: 按名字长度降序, 优先匹配更具体的名字
        sorted_maes = sorted(mutant_maes.items(), key=lambda x: len(x[0]), reverse=True)

        # 1. MAE 百分位排名 (替代原始 MAE, 避免极端值主导)
        mae_values = np.array(list(mutant_maes.values()))
        mae_percentile = {}
        for name, mae in mutant_maes.items():
            mae_percentile[name] = float((mae_values <= mae).mean())

        # 2. 家族覆盖度统计 (候选池中每个家族的样本数)
        family_counts = {}
        for name in self._pool_mutant_names:
            family = str(name).split('_LHS_')[0]
            family_counts[family] = family_counts.get(family, 0) + 1
        max_family_count = max(family_counts.values()) if family_counts else 1

        # 3. Real Mutant 参数向量 (用于参数空间距离计算)
        real_params = {}
        if dm is not None and hasattr(dm, 'test_dataset_real'):
            test_dataset = dm.test_dataset_real
            real_names = []
            if hasattr(dm, '_mutant_names') and hasattr(dm, '_test_idx_real'):
                real_names = [str(dm._mutant_names[i]) for i in dm._test_idx_real]
            for idx in range(len(test_dataset)):
                _, P, _ = test_dataset[idx]
                name = real_names[idx] if idx < len(real_names) else f"Real_{idx}"
                real_params[name] = P.cpu().numpy()

        # 候选池参数 (归一化空间, 用于距离计算)
        pool_P_np = self._pool_P.cpu().numpy() if hasattr(self._pool_P, 'cpu') else np.array(self._pool_P)

        for i, name in enumerate(self._pool_mutant_names):
            family = str(name).split('_LHS_')[0]
            # 找最匹配的 real mutant (优先取 MAE 最大的匹配)
            best_mae = 0.0
            best_name = None
            for m_name, mae in sorted_maes:
                if m_name == family or m_name.startswith(family) or family.startswith(m_name):
                    if mae > best_mae:
                        best_mae = mae
                        best_name = m_name

            # 维度 1: MAE 百分位排名 [0, 1]
            base_priority = mae_percentile.get(best_name, 0.0) if best_name else 0.0

            # 维度 2: 家族覆盖度 [0, 1] (样本越少, priority 越高)
            family_count = family_counts.get(family, 1)
            coverage_factor = 1.0 - (family_count / max_family_count)

            # 维度 3: 参数空间距离 [0, 1] (距离越近, priority 越高)
            distance_factor = 0.5  # 默认值 (无匹配 Real Mutant 时)
            if best_name and best_name in real_params:
                cand_p = pool_P_np[i]
                real_p = real_params[best_name]
                dist = np.linalg.norm(cand_p - real_p)
                distance_factor = 1.0 / (1.0 + dist)

            # 综合 priority: 0.5 * 百分位 + 0.3 * 距离 + 0.2 * 覆盖度
            priorities[i] = 0.5 * base_priority + 0.3 * distance_factor + 0.2 * coverage_factor

        # 打印摘要
        top_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"[MAAL] target_priority 改进版: 0.5*MAE百分位 + 0.3*参数距离 + 0.2*覆盖度")
        print(f"[MAAL] priority range: {priorities.min():.4f}~{priorities.max():.4f}, "
              f"mean={priorities.mean():.4f}")
        print(f"[MAAL] Top-5 家族覆盖度: {top_families}")
        return priorities

    def _compute_train_family_uncertainty(self, pl_module, dm, device, n_mc_samples=5, max_train_subset=2000):
        """对训练集做 MC dropout, 按家族分组计算平均不确定性 (完全不访问测试集)

        数据隔离原则: 仅使用 dm.train_dataset_lhs, 不访问 dm.test_dataset_real。

        Args:
            pl_module: Lightning 模型
            dm: 数据模块
            device: 计算设备
            n_mc_samples: MC 采样次数 (默认 5)
            max_train_subset: 训练集子集上限 (加速, 默认 2000)

        Returns:
            family_uncertainty: {family_name: mean_uncertainty}
        """
        train_X, train_P, _ = dm.train_dataset_lhs.tensors
        n_train = len(train_P)

        # 获取训练集 mutant_names (通过 dm._train_idx_lhs 映射)
        if hasattr(dm, '_mutant_names') and hasattr(dm, '_train_idx_lhs'):
            train_names = [str(dm._mutant_names[i]) for i in dm._train_idx_lhs]
        else:
            train_names = [f"Train_{i}" for i in range(n_train)]

        # 采样训练集子集 (加速 MC dropout)
        if n_train > max_train_subset:
            subset_idx = np.random.choice(n_train, max_train_subset, replace=False)
            train_X_sub = train_X[subset_idx]
            train_P_sub = train_P[subset_idx]
            train_names_sub = [train_names[i] for i in subset_idx]
        else:
            train_X_sub = train_X
            train_P_sub = train_P
            train_names_sub = train_names

        # MC dropout: 对参数加噪声 forward K 次, 计算预测方差
        pl_module.eval()
        batch_size = 500
        all_unc = []
        with torch.no_grad():
            for i in range(0, len(train_P_sub), batch_size):
                end_idx = min(i + batch_size, len(train_P_sub))
                X_batch = train_X_sub[i:end_idx].to(device)
                P_batch = train_P_sub[i:end_idx].to(device)

                preds_list = []
                for _ in range(n_mc_samples):
                    noise = torch.randn_like(P_batch) * 0.02
                    P_perturbed = P_batch + noise
                    preds = pl_module.forward_ic_time(X_batch, P_perturbed)
                    preds_list.append(preds)

                preds_stack = torch.stack(preds_list, dim=0)
                variance = torch.var(preds_stack, dim=0)
                uncertainty = variance.mean(dim=(1, 2))
                all_unc.append(uncertainty.cpu().numpy())
        pl_module.train()
        uncertainties = np.concatenate(all_unc)

        # 按家族分组计算平均不确定性
        family_unc_list = {}
        for i, name in enumerate(train_names_sub):
            family = str(name).split('_LHS_')[0]
            if family not in family_unc_list:
                family_unc_list[family] = []
            family_unc_list[family].append(uncertainties[i])

        family_uncertainty = {f: float(np.mean(v)) for f, v in family_unc_list.items()}

        print(f"[MAAL] 训练集家族不确定性: {len(family_uncertainty)} 个家族 | "
              f"unc range: {min(family_uncertainty.values()):.4f}~{max(family_uncertainty.values()):.4f} | "
              f"子集大小: {len(train_P_sub)}/{n_train}")
        return family_uncertainty

    def _map_family_priority_train_based(self, family_uncertainty, dm):
        """将训练集家族不确定性映射到候选池样本的 target_priority (完全基于训练集, 零泄露)

        target_priority = 0.5*同家族不确定性百分位 + 0.3*候选到训练集同家族参数距离 + 0.2*家族覆盖度

        数据隔离: 不访问 dm.test_dataset_real, 仅用 dm.train_dataset_lhs。
        - 维度1 (0.5): 训练集中同家族样本的 MC dropout 不确定性 (模型对该家族不自信 → 优先)
        - 维度2 (0.3): 候选样本到训练集同家族参数中心的距离 (越远 → 训练集覆盖不足 → 优先)
        - 维度3 (0.2): 候选池中同家族样本数 (越少 → 覆盖不足 → 优先)

        Args:
            family_uncertainty: {family_name: mean_uncertainty} 来自训练集
            dm: 数据模块 (仅用 train_dataset_lhs)

        Returns:
            priorities: (N,) numpy array, 每个候选样本的 target_priority
        """
        priorities = np.zeros(self._pool_size)

        # 1. 不确定性百分位排名 [0, 1]
        unc_values = np.array(list(family_uncertainty.values()))
        unc_percentile = {}
        for name, unc in family_uncertainty.items():
            unc_percentile[name] = float((unc_values <= unc).mean())

        # 2. 家族覆盖度统计 (候选池中每个家族的样本数)
        family_counts = {}
        for name in self._pool_mutant_names:
            family = str(name).split('_LHS_')[0]
            family_counts[family] = family_counts.get(family, 0) + 1
        max_family_count = max(family_counts.values()) if family_counts else 1

        # 3. 训练集参数按家族分组 (用于参数距离计算) — 完全来自训练集
        train_P_np = dm.train_dataset_lhs.tensors[1].cpu().numpy()
        if hasattr(dm, '_mutant_names') and hasattr(dm, '_train_idx_lhs'):
            train_names = [str(dm._mutant_names[i]) for i in dm._train_idx_lhs]
        else:
            train_names = [f"Train_{i}" for i in range(len(train_P_np))]

        train_family_params = {}
        for i, name in enumerate(train_names):
            family = str(name).split('_LHS_')[0]
            if family not in train_family_params:
                train_family_params[family] = []
            train_family_params[family].append(train_P_np[i])
        # 每个家族取平均参数向量 (代表该家族的参数中心)
        train_family_center = {f: np.mean(params, axis=0) for f, params in train_family_params.items()}

        # 候选池参数 (归一化空间)
        pool_P_np = self._pool_P.cpu().numpy() if hasattr(self._pool_P, 'cpu') else np.array(self._pool_P)

        for i, name in enumerate(self._pool_mutant_names):
            family = str(name).split('_LHS_')[0]

            # 维度 1: 同家族训练集不确定性百分位 [0, 1] (不确定性越高 → priority 越高)
            base_priority = unc_percentile.get(family, 0.5)  # 默认中性值

            # 维度 2: 候选到训练集同家族参数中心的距离 [0, 1] (距离越远 → 覆盖不足 → priority 越高)
            distance_factor = 0.5  # 默认值 (训练集中无该家族时)
            if family in train_family_center:
                cand_p = pool_P_np[i]
                train_center = train_family_center[family]
                dist = np.linalg.norm(cand_p - train_center)
                # dist/(1+dist): dist 越大越接近 1, 鼓励探索训练集未覆盖的参数区域
                distance_factor = dist / (1.0 + dist)

            # 维度 3: 家族覆盖度 [0, 1] (候选池中样本越少 → priority 越高)
            family_count = family_counts.get(family, 1)
            coverage_factor = 1.0 - (family_count / max_family_count)

            # 综合 priority: 0.5*不确定性 + 0.3*参数距离 + 0.2*覆盖度
            priorities[i] = 0.5 * base_priority + 0.3 * distance_factor + 0.2 * coverage_factor

        print(f"[MAAL] target_priority 训练集版: 0.5*同家族MC不确定性 + 0.3*参数距离 + 0.2*覆盖度")
        print(f"[MAAL] priority range: {priorities.min():.4f}~{priorities.max():.4f}, "
              f"mean={priorities.mean():.4f}")
        return priorities

    def _normalize_array(self, arr):
        """归一化数组到 [0, 1]"""
        arr = np.asarray(arr, dtype=np.float64)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def _build_broadcast_from_pool(self, sel_idx, dm):
        """从候选池选样本后构建 broadcast_data (参数候选池版)

        al4pde 原始项目机制: 选中参数后 ODE 仿真生成真实轨迹, 再加入训练集。
        1. 从候选池提取选中样本的未归一化参数
        2. 调用 solve_single_mutant_bridge ODE 仿真生成轨迹
        3. 动态浓度剔除 (超训练集 max×2 视为 monster)
        4. 归一化 Y/P, 从 Y_ic 构造 X
        5. 从候选池移除已选样本
        """
        sel_idx = np.array(sel_idx, dtype=np.int64)
        n_sel = len(sel_idx)

        # 从候选池提取选中样本的参数
        sel_P_real = self._pool_P_real[sel_idx].numpy()   # (N, 141) 未归一化
        sel_names = self._pool_mutant_names[sel_idx]

        # ODE 仿真
        print(f"[al4pde] ODE 仿真生成 {n_sel} 个选中样本的轨迹...")
        sim_results = [None] * n_sel
        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = {i: executor.submit(solve_single_mutant_bridge,
                                          str(sel_names[i]), sel_P_real[i])
                       for i in range(n_sel)}
            done = 0
            for i, f in futures.items():
                try:
                    sim_results[i] = f.result(timeout=SIM_TIMEOUT + 30)
                except FutureTimeoutError:
                    print(f"  ⚠️ 选中样本 {i} ({sel_names[i]}) 进程级超时,跳过")
                    sim_results[i] = None
                done += 1
                if done % max(1, n_sel // 10) == 0 or done == n_sel:
                    print(f"  ODE 仿真进度: {done}/{n_sel} ({100*done//n_sel}%)")

        # 动态浓度剔除 (阈值参考 real mutants, 复用 _physics_validate_and_prune 的阈值逻辑)
        Y_real_norm = dm.test_dataset_real.tensors[2].cpu()
        std_y_cpu = dm.std_y.cpu()
        mean_y_cpu = dm.mean_y.cpu()
        Y_real_raw = Y_real_norm * std_y_cpu + mean_y_cpu
        real_envelope = Y_real_raw.max(dim=-1)[0].max(dim=0)[0].numpy()
        dynamic_thresholds = np.maximum(real_envelope * 2.0, 2.0)

        valid_positions = []
        valid_Y_real = []
        valid_P_real = []
        valid_names = []
        valid_labels = []

        for i in range(n_sel):
            res = sim_results[i]
            if res is None:
                continue
            _, y_uniform, p_vec, pattern_label = res

            # 浓度剔除: 任一变量超过 real mutants max×2 视为 monster
            lhs_max_vals = np.max(y_uniform, axis=1)
            is_monster = any(lhs_max_vals[v] > dynamic_thresholds[v]
                             for v in range(1, 39))
            if is_monster:
                continue

            valid_positions.append(i)
            valid_Y_real.append(y_uniform)
            valid_P_real.append(p_vec)
            valid_names.append(str(sel_names[i]))
            valid_labels.append(pattern_label)

        # 在移除前获取通过验证的样本的 X (从候选池 Y_ic 构造, 已归一化)
        if valid_positions:
            valid_sel_idx = sel_idx[valid_positions]
            X_tensor = self._get_pool_X_batch(valid_sel_idx).clone()  # (M, 39, 2, T)

        # 从候选池移除已选样本 (无论是否通过验证)
        mask = np.ones(self._pool_size, dtype=bool)
        mask[sel_idx] = False
        if self._pool_X is not None:
            self._pool_X = self._pool_X[mask]
        self._pool_P = self._pool_P[mask]
        if self._pool_Y is not None:
            self._pool_Y = self._pool_Y[mask]
        if self._pool_Y_ic is not None:
            self._pool_Y_ic = self._pool_Y_ic[mask]
        if self._pool_mutant_names is not None:
            self._pool_mutant_names = self._pool_mutant_names[mask]
        if self._pool_P_real is not None:
            self._pool_P_real = self._pool_P_real[mask]
        if self._pool_shape_features_cached is not None:
            self._pool_shape_features_cached = self._pool_shape_features_cached[mask]
        if self._al4pde_pool_idx is not None:
            self._al4pde_pool_idx = self._al4pde_pool_idx[mask]
        self._pool_size = mask.sum()

        if not valid_Y_real:
            print(f"[al4pde] 所有 {n_sel} 个选中样本均未通过物理验证, 本轮无样本加入训练集")
            print(f"[al4pde] 候选池剩余: {self._pool_size} 样本")
            return None

        # 归一化 Y (仿真输出为真实值空间, 需归一化以匹配训练集)
        mean_y = dm.mean_y  # (1, 39, 1)
        std_y = dm.std_y    # (1, 39, 1)
        p_mean = dm.p_mean  # (1, 141)
        p_std = dm.p_std    # (1, 141)

        Y_list = []
        for y_real in valid_Y_real:
            y_t = torch.tensor(y_real, dtype=torch.float32)  # (39, 500)
            y_norm = (y_t - mean_y.squeeze(0)) / std_y.squeeze(0)
            Y_list.append(y_norm)

        P_list = []
        for p_real in valid_P_real:
            p_t = torch.tensor(p_real, dtype=torch.float32)  # (141,)
            p_norm = (p_t - p_mean.squeeze(0)) / p_std.squeeze(0)
            p_norm = torch.nan_to_num(p_norm, nan=0.0, posinf=10.0, neginf=-10.0)
            P_list.append(p_norm)

        print(f"[al4pde] 仿真完成: {len(valid_Y_real)}/{n_sel} 通过验证 | "
              f"候选池剩余: {self._pool_size}")

        return {
            'X': X_tensor,
            'Y': torch.stack(Y_list),
            'P': torch.stack(P_list),
            'names': valid_names,
            'labels': valid_labels,
            'sample_plots': {}
        }
