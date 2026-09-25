import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_current_dir))
for _p in [_repo_root, os.path.dirname(_current_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
if 'PYTHONPATH' in os.environ:
    if _repo_root not in os.environ['PYTHONPATH'].split(os.pathsep):
        os.environ['PYTHONPATH'] = f"{_repo_root}{os.pathsep}{os.environ['PYTHONPATH']}"
else:
    os.environ['PYTHONPATH'] = _repo_root

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
# 1. 导入酵母 ODE 方程与参数提取逻辑 (供 PIAL 残差计算)
# ==============================================================
from budding_yeast_v2.data.LHS.lhs_v1_origin import (
    get_default_params,
    eqns as _yeast_ode_eqns,
)
from budding_yeast_v2.data.mass_dataset_generator import (
    simulate_mutant as simulate_mutant_mass,
    is_duplicate_real_mutant,
    initial_conds as initial_conds_mass,
    get_default_params as get_default_params_mass,
    BINDING_GROUPS as MASS_BINDING_GROUPS,
    # y0 变量索引 (用于把 ovr 中全部 init_* 键写入初始条件, 与 mass 版 simulate_mutant 对齐)
    nCDH1T as _nCDH1T, nCDH1 as _nCDH1, nBUB2 as _nBUB2,
    nCLN2 as _nCLN2, nCLB2 as _nCLB2, nCLB5 as _nCLB5,
    nSIC1 as _nSIC1, nCDC6 as _nCDC6, nCDC14 as _nCDC14, nNET1 as _nNET1,
    nCDC20 as _nCDC20, nTEM1 as _nTEM1, nCDC15 as _nCDC15,
    nPPX as _nPPX, nPDS1 as _nPDS1, nESP1 as _nESP1,
)
# v2.6: 全 al4pde 策略统一使用 random-only 采样器 (无 rescue motif 定向采样, 保留 Level 分层)
from budding_yeast_v2.data.mass_random_sampler import sample_mass_random

# MASS 默认参数向量 (与 mass_dataset_generator.generate_mass_dataset 完全一致)
# 注: 与 lhs_v1_origin 默认值仅 ef6n3 不同 (0.3 vs 0.0), 候选池统一以 MASS 默认值为准
_MASS_DEFAULTS = get_default_params_mass()
_MASS_DEFAULTS.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

# ovr 中 init_* 键 → y0 变量索引 (对齐 mass_dataset_generator.simulate_mutant L224-240)
_MASS_INIT_Y0_IDX = {
    'init_CDH1T': _nCDH1T, 'init_CDH1': _nCDH1, 'init_BUB2': _nBUB2,
    'init_CLN2': _nCLN2, 'init_CLB2': _nCLB2, 'init_CLB5': _nCLB5,
    'init_SIC1': _nSIC1, 'init_CDC6': _nCDC6, 'init_CDC14': _nCDC14,
    'init_NET1': _nNET1, 'init_CDC20': _nCDC20, 'init_Tem1': _nTEM1,
    'init_Cdc15': _nCDC15, 'init_PPX': _nPPX, 'init_PDS1': _nPDS1,
    'init_ESP1': _nESP1,
}
# mass 版仿真的硬超时 (对齐 generate_mass_dataset 的 future.result(timeout=15))
MASS_SIM_TIMEOUT = 15.0

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


def _parse_mass_gene_modes(desc_str):
    """解析 MASS 采样描述串为 (gene, mode) 对列表。

    desc_str 格式 (sample_mass_mutant 返回):
      - 单基因: 'CLB2[KO]'
      - 复合机制: 'CLB2[GAL-OE+db]'
      - 多基因: 'CLB2[KO]+CLN2[GAL-OE]'

    Returns:
        [(gene, mode), ...] 复合机制展开为多个 (gene, mode) 对
    """
    import re
    pairs = []
    for m in re.finditer(r'([A-Za-z0-9_]+)\[([^\]]+)\]', str(desc_str)):
        gene = m.group(1)
        for mode in m.group(2).split('+'):
            mode = mode.strip()
            if mode:
                pairs.append((gene, mode))
    return pairs


def _mass_sample_genes(name):
    """从 MASS 样本名提取全部突变基因 (去重保序)。

    'MASS_RAND_CLB2[KO]+CLN2[GAL-OE]_L1_123' → ['CLB2', 'CLN2']
    'MASS_RAND_CLB2[GAL-OE+db]_L2_7'          → ['CLB2'] (复合机制同基因去重)
    无突变基因 (纯碳源 mdt 变化) → []

    v2.7 多基因家族聚合使用: 一个样本的家族信号应归属其全部突变基因
    (CLB2[KO]+CLN2[GAL-OE] 同时携带 CLB2 与 CLN2 两个家族的信息,
    仅看首个基因会丢弃其余家族信号)。
    """
    desc = _mass_desc_of(name)
    if desc is None:
        desc = str(name)
    genes = []
    for gene, _mode in _parse_mass_gene_modes(desc):
        if gene not in genes:
            genes.append(gene)
    return genes


def _mass_desc_of(name):
    """从 MASS 样本名提取突变描述串 (供 _parse_mass_gene_modes 解析 (gene, mode) 对)。

    'MASS_RAND_CLB2[KO]+CLN2[GAL-OE]_L1_123' → 'CLB2[KO]+CLN2[GAL-OE]'
    无法解析时返回 None。
    """
    import re
    m = re.match(r'MASS(?:_AL|_RAND)?_(.+)_L\d+_\d+$', str(name))
    return m.group(1) if m else None



def solve_single_mass_bridge(name, ovr):
    """MASS 版 ODE 仿真桥: 用完整 ovr 字典调用 mass_dataset_generator.simulate_mutant。

    与 solve_single_mutant_bridge (lhs 版) 的关键差异:
    1. 接收完整 ovr 字典 (含 init_Tem1/init_Cdc15/Vi20/kbub2 等 lhs 版不认识的键),
       保证 TEM1-KO / CDC15-KO 等突变的初始条件覆盖与数据集生成时完全一致;
    2. mass 版引擎自带 MAD2-CSL / BUB2-CKO 的特殊处理 (is_mad2_ko / is_bub2_ko);
    3. mass 版引擎自带 mass>10 过滤 (返回 None + "Discarded_Mass");
    4. 补 SIGALRM 硬超时保护 (mass 版引擎无内建超时, 防止 BDF 卡死进程池 worker)。

    Args:
        name: 样本名 (仅用于日志/记录, mass 引擎不查表)
        ovr: sample_mass_mutant 返回的完整参数覆盖字典

    Returns:
        (X_dummy, y_uniform, p_vec, pattern_label) 或 None (仿真失败/被过滤)
    """
    import signal

    def _sigalrm_handler(signum, frame):
        raise TimeoutError("MASS ODE BDF Timeout (SIGALRM)")

    old_handler = signal.signal(signal.SIGALRM, _sigalrm_handler)
    signal.alarm(max(1, int(MASS_SIM_TIMEOUT)))
    try:
        m_name, y_uniform, bio_label, pattern_label = simulate_mutant_mass(
            name, ovr, 210, 500)
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if y_uniform is None:
        return None

    # ovr → 141 维参数向量 (仅取 PARAM_KEYS 中的键, 与数据集 npz 导出逻辑一致)
    p_vec = np.array([ovr.get(k, _MASS_DEFAULTS[k]) for k in PARAM_KEYS],
                     dtype=np.float64)
    X_dummy = np.zeros((39, 2, 500))
    return (X_dummy, y_uniform, p_vec, pattern_label)


def _solve_single_mass_task_wrapper(task_tuple):
    """顶级函数包装器，供 ProcessPoolExecutor(mp_context='spawn') 进程池调用"""
    name, ovr = task_tuple
    try:
        return solve_single_mass_bridge(name, ovr)
    except Exception:
        return None


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
                 # ====== 兼容参数 ======
                 atlas_csv_path=None,
                 # ====== al4pde 框架专属参数 ======
                 al_pool_dataset=None,
                 target_weight=0.4,
                 mass_weight=0.5,
                 # ====== al4pde_novelty 策略专属参数 ======
                 novelty_weight=0.6,
                 family_weight=0.2):
        """
        标准主动学习 (Active Learning) 与 al4pde 外层重训框架回调模块。

        支持两大 AL 范式：
        1. **al4pde 框架策略 (推荐)**：'al4pde_random', 'al4pde_uncertainty', 'al4pde_diversity',
           'maal', 'al4pde_mass', 'al4pde_novelty'。
           候选池为纯参数空间采样，选中 Top-K 后通过 MASS 物理引擎在线 ODE 仿真打标加入合成训练集。
        2. **标准池化 AL (Pool-Based)**：'us', 'is', 'random', 'wrs', 'vessal', 'hggs'。
           从已有的合成数据池中选取样本加入训练集。

        注: 查询合成策略 ('rgs', 'mutant_aware') 因存在测试集数据泄露风险已被彻底废弃并移除。
        """
        self.trigger_every_n_epochs = trigger_every_n_epochs
        self.strategy = strategy.lower()
        if self.strategy in ('rgs', 'mutant_aware'):
            raise ValueError(
                f"Strategy '{self.strategy}' (Query Synthesis) has been permanently removed "
                f"due to test-set data leakage. Please use al4pde strategies (e.g. 'al4pde_random', "
                f"'al4pde_uncertainty', 'al4pde_diversity', 'maal', 'al4pde_mass', 'al4pde_novelty') "
                f"or standard pool-based strategies ('us', 'is')."
            )

        self.num_add = num_add
        self.perturbation = perturbation
        self.mae_threshold = mae_threshold
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

        # ====== al4pde 框架状态 ======
        self.al_pool_dataset = al_pool_dataset
        self.target_weight = target_weight
        # al4pde_mass 策略: MASS 生物学分层优先级权重 (diversity 权重 = 1 - mass_weight)
        self.mass_weight = mass_weight
        # al4pde_novelty 策略权重: acquisition = w_n*novelty + w_f*family_unc + w_d*diversity
        #   (w_d = 1 - novelty_weight - family_weight, 自动归一化)
        self.novelty_weight = novelty_weight
        self.family_weight = family_weight
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
        self._pool_shape_features_cached = None  # 形状特征缓存 (diversity/maal 用)
        # MASS 候选池专用状态:
        self._pool_ovr = None           # 每个候选的完整 ovr 字典 (含 init_Tem1 等额外键, 供 mass 版仿真)
        self._pool_mass_meta = None     # 每个候选的采样元数据 [{'genes': [(gene, mode)], 'level': int}, ...]

        if self.strategy in ('al4pde_random', 'al4pde_uncertainty', 'al4pde_diversity',
                             'maal', 'al4pde_mass', 'al4pde_novelty'):
            self.initial_train_size = initial_train_size
            self.trigger_every_n_epochs = 4  # 保留字段, 外层重训模式下不生效
            self.num_add = num_add            # 保留字段, 外层重训模式下不生效
            self._al4pde_num_al_iter = 3      # 3 轮 AL (Iter 0: 10k -> Iter 1: 20k -> Iter 2: 40k)
            self._al4pde_per_round_add = 10000
            print(f"[AL Manager] al4pde 设置: initial_train_size={self.initial_train_size}, "
                  f"{self._al4pde_num_al_iter} 轮指数倍增 (10k -> 20k -> 40k)")
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
        标准主动学习初始化：
        - 池化策略 (us/is/random/wrs/vessal/hggs)：通过 K-Means 聚类选出初始训练子集,
          剩余样本构成未标注池 (unlabeled pool), 后续从池中查询。
        - al4pde 策略：初始化由 train.py 外层循环调用的 al4pde_setup 负责。
        """
        # 所有 AL 策略都执行初始化 (none 已在外层过滤)
        if self.strategy == 'none':
            return

        # 仅在 global zero 进程执行初始化
        if not trainer.is_global_zero:
            return

        dm = trainer.datamodule
        if dm is None or not hasattr(dm, 'train_dataset_synth'):
            print("[AL Manager] 警告: 未找到 train_dataset_synth，跳过标准 AL 初始化")
            return

        # 获取全量合成训练集
        full_tensors = dm.train_dataset_synth.tensors
        full_X, full_P, full_Y = full_tensors[0], full_tensors[1], full_tensors[2]
        full_size = full_X.shape[0]

        print(f"\n[AL Manager] ===== 标准主动学习初始化 =====")
        print(f"[AL Manager] 策略: {self.strategy.upper()} | 全量合成训练集: {full_size}")

        # ====== al4pde 框架: 初始化由 train.py 外层循环调用的 al4pde_setup 负责 ======
        if self._is_al4pde_strategy():
            if not self._al_initialized:
                print("[AL Manager] al4pde 策略: 初始化由 train.py 外层循环的 al4pde_setup 负责 "
                      "(随机候选池 + 随机初始训练集), on_train_start 跳过 K-Means 路径")
            return

        # ====== 池化策略 (us/is/random/wrs/vessal/hggs): K-Means 初始化 ======
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
        dm.train_dataset_synth = TensorDataset(
            full_X[selected_idx],
            full_P[selected_idx],
            full_Y[selected_idx]
        )
        # 同步更新 _train_idx_synth
        if hasattr(dm, '_train_idx_synth'):
            dm._train_idx_synth = np.asarray(dm._train_idx_synth)[selected_idx]

        # 标准池化策略: 保存未标注池
        self._pool_X = full_X[pool_idx]
        self._pool_P = full_P[pool_idx]
        self._pool_Y = full_Y[pool_idx]
        self._pool_size = self._pool_X.shape[0]

        self._al_initialized = True
        self._train_size_history.append(initial_size)

        print(f"[AL Manager] 初始训练集: {initial_size} 样本 (K-Means={self.initial_selection=='kmeans'})")
        print(f"[AL Manager] 未标注池: {self._pool_size} 样本 | 策略: {self.strategy}")
        print(f"[AL Manager] 验证集/测试集保持不变")
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

    def al4pde_setup(self, dm, device='cuda', exp_dir=None):
        """v2.3 外层重训式: 一次性初始化 (加载候选池 + 确定初始训练集 + 累积数据缓存)。
        供 train.py 在外层循环开始前调用一次。"""
        from torch.utils.data import TensorDataset

        self._generate_param_only_pool(dm)  # 加载预存候选池 (如 pure_random_1m_pool.npz)

        # 缓存累积的物理空间数据，用于后续各轮次主动学习训练集落盘 (.npz)
        self._cum_synth_data = []
        self._cum_synth_params = []
        self._cum_synth_names = []
        self._cum_synth_labels = []
        self._benchmark_real_data = []
        self._benchmark_real_params = []
        self._benchmark_real_names = []
        self._benchmark_real_labels = []
        self._param_names_cached = np.array(PARAM_KEYS)

        dataset_path = getattr(dm, 'dataset_path', None)
        if dataset_path and os.path.exists(dataset_path):
            init_npz = np.load(dataset_path, allow_pickle=True)
            if 'data' in init_npz and 'params' in init_npz:
                data_arr = init_npz['data']
                params_arr = init_npz['params']
                mutant_names_arr = init_npz['mutant_names'] if 'mutant_names' in init_npz else None
                pattern_labels_arr = init_npz['pattern_labels'] if 'pattern_labels' in init_npz else None

                if hasattr(dm, '_train_idx_synth') and len(dm._train_idx_synth) > 0:
                    idx = dm._train_idx_synth
                    self._cum_synth_data = list(data_arr[idx])
                    self._cum_synth_params = list(params_arr[idx])
                    if mutant_names_arr is not None:
                        self._cum_synth_names = [str(n) for n in mutant_names_arr[idx]]
                    else:
                        self._cum_synth_names = [f"Train_{i}" for i in range(len(idx))]
                    if pattern_labels_arr is not None:
                        self._cum_synth_labels = [str(l) for l in pattern_labels_arr[idx]]
                    else:
                        self._cum_synth_labels = ['Pattern_B'] * len(self._cum_synth_names)

                if hasattr(dm, '_test_idx_real') and len(dm._test_idx_real) > 0:
                    idx = dm._test_idx_real
                    self._benchmark_real_data = list(data_arr[idx])
                    self._benchmark_real_params = list(params_arr[idx])
                    if mutant_names_arr is not None:
                        self._benchmark_real_names = [str(n) for n in mutant_names_arr[idx]]
                    else:
                        self._benchmark_real_names = [f"Real_{i}" for i in range(len(idx))]
                    if pattern_labels_arr is not None:
                        self._benchmark_real_labels = [str(l) for l in pattern_labels_arr[idx]]
                    else:
                        self._benchmark_real_labels = ['Pattern_B'] * len(self._benchmark_real_names)

                if 'param_names' in init_npz:
                    self._param_names_cached = init_npz['param_names']

        # 检查是否已直接通过 --dataset_name 加载了预生成的初始训练集 (如 al_iter0_10000_210m_500s.npz)
        if hasattr(dm, 'train_dataset_synth') and len(dm.train_dataset_synth) > 0:
            print(f"[al4pde] ✅ 直接复用预生成的初始训练集: {len(dm.train_dataset_synth)} 样本 (无须重复进行初始 ODE 仿真)")
            if hasattr(dm, '_mutant_names'):
                existing_names = set(str(n) for n in dm._mutant_names)
                mask = np.isin(self._pool_mutant_names, list(existing_names), invert=True)
                if mask.sum() < len(mask):
                    n_removed = len(mask) - mask.sum()
                    print(f"[al4pde] 从候选池中剔除初始训练集已占用的 {n_removed} 个样本")
                    self._pool_P = self._pool_P[mask]
                    if self._pool_X is not None:
                        self._pool_X = self._pool_X[mask]
                    if self._pool_Y is not None:
                        self._pool_Y = self._pool_Y[mask]
                    if self._pool_Y_ic is not None:
                        self._pool_Y_ic = self._pool_Y_ic[mask]
                    self._pool_mutant_names = self._pool_mutant_names[mask]
                    if self._pool_P_real is not None:
                        self._pool_P_real = self._pool_P_real[mask]
                    if self._pool_shape_features_cached is not None:
                        self._pool_shape_features_cached = self._pool_shape_features_cached[mask]
                    if self._al4pde_pool_idx is not None:
                        self._al4pde_pool_idx = self._al4pde_pool_idx[mask]
                    if self._pool_ovr is not None:
                        self._pool_ovr = [o for o, m in zip(self._pool_ovr, mask) if m]
                    self._pool_size = mask.sum()
        else:
            initial_size = min(self.initial_train_size, self._pool_size)
            rng = np.random.RandomState(self.random_seed + 1)
            sel_idx = rng.permutation(self._pool_size)[:initial_size]
            print(f"[al4pde] 从候选池完全随机抽取初始集: {initial_size}/{self._pool_size} (seed={self.random_seed + 1})")
            bd = self._build_broadcast_from_pool(sel_idx, dm)
            if bd is None or bd['X'].shape[0] == 0:
                raise RuntimeError("[al4pde] 初始训练集仿真全部失败, 无法继续")
            dm.train_dataset_synth = TensorDataset(bd['X'], bd['P'], bd['Y'])
            if hasattr(dm, '_mutant_names'):
                n_existing = len(dm._mutant_names)
                dm._mutant_names = list(dm._mutant_names) + list(bd['names'])
                dm._train_idx_synth = np.arange(n_existing, n_existing + len(bd['names']))
            if 'Y_real' in bd:
                self._cum_synth_data.extend(bd['Y_real'])
                self._cum_synth_params.extend(bd['P_real'])
                self._cum_synth_names.extend(bd['names'])
                self._cum_synth_labels.extend(bd['labels'])

        self._al_initialized = True
        self._al4pde_al_iter = 0
        self._train_size_history.append(len(dm.train_dataset_synth))

        # 保存 Round 0 初始训练集到实验目录 datasets/
        if exp_dir and len(self._cum_synth_data) > 0:
            ds_dir = os.path.join(exp_dir, 'datasets')
            os.makedirs(ds_dir, exist_ok=True)
            r0_path = os.path.join(ds_dir, f'al_iter0_train{len(self._cum_synth_data)}.npz')
            if not os.path.exists(r0_path):
                np.savez_compressed(
                    r0_path,
                    data=np.array(self._benchmark_real_data + self._cum_synth_data, dtype=np.float32),
                    params=np.array(self._benchmark_real_params + self._cum_synth_params, dtype=np.float64),
                    mutant_names=np.array(self._benchmark_real_names + self._cum_synth_names),
                    pattern_labels=np.array(self._benchmark_real_labels + self._cum_synth_labels),
                    param_names=self._param_names_cached
                )
                print(f"💾 [al4pde] 已将 Round 0 初始数据集落盘保存至: {r0_path}")

        print(f"[al4pde] 初始训练集就绪: {len(dm.train_dataset_synth)} 样本 | 候选池剩余: {self._pool_size} | 策略: {self.strategy}")

    def al4pde_select_and_append(self, pl_module, dm, device='cuda', exp_dir=None):
        """v2.3 外层重训式: 用当前(上一轮训练完的)模型选样并追加到 train_dataset_synth。
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
        elif self.strategy == 'maal':
            bd = self._maal_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_mass':
            bd = self._al4pde_mass_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_novelty':
            bd = self._al4pde_novelty_sample(None, pl_module, dm, device, 0)
        else:
            bd = None
        if bd is None:
            print("[al4pde v2.3] 本轮候选池为空, 未追加样本")
            if pl_module is not None:
                pl_module.train()
            return 0
        old_tensors = dm.train_dataset_synth.tensors
        old_X, old_P, old_Y = old_tensors[0], old_tensors[1], old_tensors[2]
        if len(old_tensors) >= 4:
            pattern_ids = torch.cat([old_tensors[3], torch.zeros(bd['X'].shape[0], dtype=old_tensors[3].dtype, device=old_tensors[3].device)], dim=0)
            dm.train_dataset_synth = TensorDataset(
                torch.cat([old_X, bd['X'].to(old_X.device)], dim=0),
                torch.cat([old_P, bd['P'].to(old_X.device)], dim=0),
                torch.cat([old_Y, bd['Y'].to(old_Y.device)], dim=0),
                pattern_ids
            )
        else:
            dm.train_dataset_synth = TensorDataset(
                torch.cat([old_X, bd['X'].to(old_X.device)], dim=0),
                torch.cat([old_P, bd['P'].to(old_X.device)], dim=0),
                torch.cat([old_Y, bd['Y'].to(old_Y.device)], dim=0),
            )
        if 'names' in bd and hasattr(dm, '_mutant_names'):
            new_names = list(bd['names'])
            n_existing = len(dm._mutant_names)
            dm._mutant_names = list(dm._mutant_names) + new_names
            new_indices = np.arange(n_existing, n_existing + len(new_names))
            if hasattr(dm, '_train_idx_synth'):
                dm._train_idx_synth = np.concatenate([np.asarray(dm._train_idx_synth), new_indices])
            else:
                dm._train_idx_synth = new_indices
        self._al4pde_al_iter += 1
        self._train_size_history.append(len(dm.train_dataset_synth))
        n_added = int(bd['X'].shape[0])
        print(f"[al4pde v2.3] 追加 {n_added} 个样本 → 训练集 {len(dm.train_dataset_synth)} "
              f"(al_iter={self._al4pde_al_iter})")

        # 持久化落盘当轮更新后的数据集 .npz 与选拔明细 CSV
        if exp_dir and 'Y_real' in bd and len(bd['Y_real']) > 0:
            ds_dir = os.path.join(exp_dir, 'datasets')
            os.makedirs(ds_dir, exist_ok=True)
            self._cum_synth_data.extend(bd['Y_real'])
            self._cum_synth_params.extend(bd['P_real'])
            self._cum_synth_names.extend(bd['names'])
            self._cum_synth_labels.extend(bd['labels'])

            round_ds_path = os.path.join(ds_dir, f'al_iter{self._al4pde_al_iter}_train{len(self._cum_synth_data)}.npz')
            np.savez_compressed(
                round_ds_path,
                data=np.array(self._benchmark_real_data + self._cum_synth_data, dtype=np.float32),
                params=np.array(self._benchmark_real_params + self._cum_synth_params, dtype=np.float64),
                mutant_names=np.array(self._benchmark_real_names + self._cum_synth_names),
                pattern_labels=np.array(self._benchmark_real_labels + self._cum_synth_labels),
                param_names=self._param_names_cached
            )
            print(f"💾 [al4pde v2.3] 已将 Iteration {self._al4pde_al_iter} 扩充数据集完整保存至: {round_ds_path}")

            # 保存选样明细清单 CSV
            import csv
            sample_csv_path = os.path.join(exp_dir, 'eval', f'selected_samples_round_{self._al4pde_al_iter}.csv')
            os.makedirs(os.path.dirname(sample_csv_path), exist_ok=True)
            with open(sample_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Selection_Index', 'Mutant_Name', 'Pattern_Label', 'AL_Iteration', 'Strategy'])
                for s_i, (s_name, s_lbl) in enumerate(zip(bd['names'], bd['labels'])):
                    writer.writerow([s_i + 1, s_name, s_lbl, self._al4pde_al_iter, self.strategy])
            print(f"📝 [al4pde v2.3] 已将当轮选拔样本清单保存至: {sample_csv_path}")

        if pl_module is not None:
            pl_module.train()
        return n_added
        if pl_module is not None:
            pl_module.train()
        return n_added

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        # v2.3: al4pde 外层重训模式下, 选样由 train.py 外层循环负责, 关闭 in-loop 触发
        if getattr(self, '_outer_loop_mode', False) and self._is_al4pde_strategy():
            return

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
            # ====== al4pde 框架策略 (从候选池查询) ======
            if self.strategy == 'al4pde_random':
                broadcast_data = self._al4pde_random_sample(trainer, pl_module, dm, device, current_epoch)
            elif self.strategy == 'al4pde_uncertainty':
                broadcast_data = self._al4pde_uncertainty_sample(trainer, pl_module, dm, device, current_epoch)
            elif self.strategy == 'al4pde_diversity':
                broadcast_data = self._al4pde_diversity_sample(trainer, pl_module, dm, device, current_epoch)
            elif self.strategy == 'maal':
                broadcast_data = self._maal_sample(trainer, pl_module, dm, device, current_epoch)
            elif self.strategy == 'al4pde_mass':
                broadcast_data = self._al4pde_mass_sample(trainer, pl_module, dm, device, current_epoch)
            elif self.strategy == 'al4pde_novelty':
                broadcast_data = self._al4pde_novelty_sample(trainer, pl_module, dm, device, current_epoch)
            # 所有池化策略 (us/is/random/wrs/vessal/hggs)：统一走标准 AL 入口
            elif self._al_initialized:
                broadcast_data = self._standard_al_sample(trainer, pl_module, dm, device, current_epoch)
            else:
                broadcast_data = None

        if trainer.world_size > 1:
            broadcast_data = trainer.strategy.broadcast(broadcast_data, src=0)

        if broadcast_data is not None:
            from torch.utils.data import TensorDataset

            old_tensors = dm.train_dataset_synth.tensors
            old_X, old_P, old_Y = old_tensors[0], old_tensors[1], old_tensors[2]

            updated_X = torch.cat([old_X, broadcast_data['X'].to(old_X.device)], dim=0)
            updated_P = torch.cat([old_P, broadcast_data['P'].to(old_X.device)], dim=0)
            updated_Y = torch.cat([old_Y, broadcast_data['Y'].to(old_Y.device)], dim=0)

            if len(old_tensors) >= 4:
                old_patterns = old_tensors[3]
                new_patterns = torch.zeros(broadcast_data['X'].shape[0], dtype=old_patterns.dtype, device=old_patterns.device)
                dm.train_dataset_synth = TensorDataset(updated_X, updated_P, updated_Y, torch.cat([old_patterns, new_patterns], dim=0))
            else:
                dm.train_dataset_synth = TensorDataset(updated_X, updated_P, updated_Y)
            self._train_size_history.append(len(dm.train_dataset_synth))
            print(f"[AL Manager] 训练集成功扩容！当前新样本总数: {len(dm.train_dataset_synth)}")

            if 'names' in broadcast_data and hasattr(dm, '_mutant_names'):
                new_names = list(broadcast_data['names'])
                n_existing = len(dm._mutant_names)
                dm._mutant_names = list(dm._mutant_names) + new_names
                new_indices = np.arange(n_existing, n_existing + len(new_names))
                if hasattr(dm, '_train_idx_synth'):
                    dm._train_idx_synth = np.concatenate([np.asarray(dm._train_idx_synth), new_indices])
                else:
                    dm._train_idx_synth = new_indices

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
        train_P = dm.train_dataset_synth.tensors[1].cpu().numpy()
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
        batch_size = 2500  # 针对 100 万候选池显著提升批大小加速前向推理 (RTX A6000 48G 显存)
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
        batch_size = 2500  # 针对 100 万候选池提速
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
        train_X, train_P, train_Y_norm = dm.train_dataset_synth.tensors[:3]
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
        train_X, train_P, train_Y_norm = dm.train_dataset_synth.tensors[:3]
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
    def _generate_param_only_pool(self, dm):
        """生成只含参数的候选池 (实验开始前调用, random-only 采样版)

        v2.6 统一规范 (所有 al4pde 策略, 用户指定: 训练集/候选池完全随机):
        - 采样器: sample_mass_random (mass_random_sampler.py, 基于
          mass_dataset_generator 但移除 rescue motif 定向采样分支),
          Level 分布 L1/L2/L3/L4 = 15%/30%/30%/25% (v2.7 多基因加强,
          数据集原生 25/35/25/15; L3+L4 上位性组合载体 40% → 55%)
          (L1: ki=1 | L2: ki=2 | L3: ki=3~5 | L4: ki=4~6, 任何 Level 均无救援基序)
          随机探索可能偶然撞出 motif 形组合 (随机碰撞, 非定向构造), 属于允许范围
        - 池规模 100000: 初始集 5000 + 4 轮 × 19000 = 81000 全部取自本池,
          需为 ODE 仿真失败/物理过滤预留冗余
        - 样本名前缀 MASS_RAND_ (区分数据集原生 MASS_ 样本)
        - 去重 1 (D_real 隔离): is_duplicate_real_mutant, 防止与 126 个真实突变体参数
          完全重复 (避免测试集参数泄露进训练池)
        - 去重 2 (池内多样性): 相同扰动参数键集合的候选之间, mdt 要求精确相等、其余
          扰动参数相对差全部 < 1% 视为重复丢弃 (v2.8: 8% → 1%, 用户指定)
          背景量化: 8% 时 L1 容量仅 ~5.9K (32 个确定性模式单元值固定无随机性,
          连续模式槽位数 = ln(取值范围)/ln(1.08) ≈ 每单元每碳源 40 个),
          低于 L1 配额 15000 → 指针卡死烧尽尝试预算 (实测池仅 39387/100000);
          1% 使连续模式槽位 ×7.7 (ln1.08/ln1.01), L1 容量 ~45K, 15K 配额可填满,
          实际样本平均间距 ~3-4% (15K/45K 槽占用), 非极端近重复;
          L2-L4 组合空间远大于配额, 阈值变化对其无实际影响
        - 候选池只存参数信息 (P + Y_IC + 完整 ovr + 采样元数据), 不含完整时间序列
        - 选中后通过 mass 版 ODE 引擎仿真生成真实轨迹 (al4pde 原始项目机制)

        内存优化: 只存 P_norm + Y_IC + P_real + ovr + meta, 完整 Y 选中后 ODE 生成
        """
        import math

        # 若提供了预存候选池 (如 pure_random_1m_pool.npz)，直接毫秒级加载
        if self.al_pool_dataset and os.path.exists(self.al_pool_dataset):
            print(f"[al4pde] 🚀 检测到预存候选池文件: {self.al_pool_dataset}，正在快速加载...")
            t_load = time.time()
            data = np.load(self.al_pool_dataset, allow_pickle=True)
            P_tensor = torch.tensor(data['params'], dtype=torch.float32)
            pool_names = list(data['mutant_names'])

            if 'y_init' in data:
                Y_ic_tensor = torch.tensor(data['y_init'], dtype=torch.float32)
            else:
                y0_base = initial_conds_mass()
                Y_ic_tensor = torch.tensor(np.tile(y0_base, (len(P_tensor), 1)), dtype=torch.float32)

            mean_y = dm.mean_y
            std_y = dm.std_y
            p_mean = dm.p_mean.to(P_tensor.device)
            p_std = dm.p_std.to(P_tensor.device)

            if getattr(dm, 'norm_type', 'log1p') == 'asinh_z':
                z = (P_tensor - p_mean) / p_std
                P_norm = torch.asinh(z)
            else:
                P_norm = (P_tensor - p_mean) / p_std
            P_norm = torch.nan_to_num(P_norm, nan=0.0, posinf=10.0, neginf=-10.0)

            if getattr(dm, 'norm_type', 'log1p') in ('log1p', 'asinh_z'):
                Y_ic_trans = torch.log1p(torch.clamp(Y_ic_tensor, min=0.0))
            else:
                Y_ic_trans = Y_ic_tensor
            mean_y_ic = mean_y.squeeze(0).squeeze(-1).to(Y_ic_tensor.device)
            std_y_ic = std_y.squeeze(0).squeeze(-1).to(Y_ic_tensor.device)
            Y_ic_norm = (Y_ic_trans - mean_y_ic) / std_y_ic
            Y_ic_norm = torch.nan_to_num(Y_ic_norm, nan=0.0, posinf=10.0, neginf=-10.0)

            T_STEPS = 500
            t_grid = torch.linspace(0, 1, T_STEPS).reshape(1, 1, T_STEPS)

            self._pool_X = None
            self._pool_P = P_norm
            self._pool_Y = None
            self._pool_Y_ic = Y_ic_norm
            self._al4pde_t_grid = t_grid
            self._pool_mutant_names = np.array(pool_names)
            self._pool_size = len(pool_names)
            self._al4pde_pool_loaded = True
            self._pool_P_real = P_tensor
            self._pool_ovr = None

            pool_metas = []
            for name in pool_names:
                desc_str = _mass_desc_of(name)
                lvl = 1
                if '_L' in str(name):
                    try:
                        lvl = int(str(name).split('_L')[1].split('_')[0])
                    except Exception:
                        pass
                pool_metas.append({
                    'genes': _parse_mass_gene_modes(desc_str),
                    'level': lvl
                })
            self._pool_mass_meta = pool_metas
            self._al4pde_mean_y = mean_y
            self._al4pde_std_y = std_y
            self._al4pde_pool_idx = None

            mem_gb = (P_norm.element_size() * P_norm.nelement() +
                      Y_ic_norm.element_size() * Y_ic_norm.nelement()) / 1e9
            print(f"[al4pde] ✅ 预存候选池加载完成: {self._pool_size} 样本, 内存={mem_gb:.2f} GB "
                  f"(耗时 {time.time() - t_load:.2f}s)")
            return

        N_POOL_TARGET = 100000   # v2.6: 随机池统一 100000 (全 al4pde 策略)
        DEDUP_REL_TOL = 0.01    # v2.8: 池内去重相对阈值 8% → 1% (用户指定, 详见 docstring)
        max_attempts = max(50000, N_POOL_TARGET * 10)
        STALL_LIMIT = 20000     # v2.8: 同一 Level 连续被拒上限 → 判定饱和并跳过其剩余槽位

        print(f"[al4pde] ===== 生成参数候选池 (random-only 采样器: 无定向采样, 保留 Level 分层) =====")
        print(f"[al4pde] 目标规模: {N_POOL_TARGET} | Level 分布 L1/L2/L3/L4 = 15%/30%/30%/25% "
              f"(v2.7 多基因加强, 数据集原生为 25/35/25/15)")
        print(f"[al4pde] 去重: D_real 隔离 (126 real mutants) + 池内 1% 相对阈值 (v2.8)")

        # Level 分配 (v2.7 多基因加强, 用户指定: 候选池多基因 Level 占比稍高)
        # 数据集生成用 25/35/25/15; 候选池改为 15/30/30/25:
        #   - L3+L4 (ki≥3, 含上位性组合的主要载体) 40% → 55%
        #   - L1 单基因降至 15% (单基因覆盖成本低, AL 选样空间重点在多基因组合)
        #   - L2 (ki=2, 二元上位性) 保持 30%
        # 初始训练集从池中均匀随机抽取 → 自动继承此分布
        n_l1 = int(math.floor(0.15 * N_POOL_TARGET))
        n_l2 = int(math.floor(0.30 * N_POOL_TARGET))
        n_l3 = int(math.floor(0.30 * N_POOL_TARGET))
        n_l4 = N_POOL_TARGET - n_l1 - n_l2 - n_l3
        level_assignments = [1] * n_l1 + [2] * n_l2 + [3] * n_l3 + [4] * n_l4
        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(level_assignments)

        # 池内去重分桶: {扰动参数键集合 tuple: [扰动值数组 list]}
        dedup_buckets = {}
        n_dup_real = 0      # 与 real mutant 重复被拦截数
        n_dup_pool = 0      # 池内重复被拦截数

        pool_names = []
        pool_params = []
        pool_ics = []
        pool_ovrs = []
        pool_metas = []

        y0_base = initial_conds_mass()  # (39,) 固定初始条件 (Chen 2004 Table 2B)

        attempts = 0
        n_accepted = 0
        t_start = time.time()
        # v2.8 槽位指针 + 饱和跳过 (修复卡死 bug):
        # 旧版 level = level_assignments[n_accepted] —— 当某 Level 容量耗尽时
        # (如 8% 阈值下 L1 只能收 ~5.9K < 配额 15000), 指针永远卡在该槽位
        # (每次尝试必被拒, n_accepted 不前进), 烧尽尝试预算且后续槽位无法到达。
        # 新版: level_ptr 独立推进; 某 Level 连续 STALL_LIMIT 次被拒 → 判定饱和,
        # 跳过其全部剩余槽位, 其余 Level 正常填充。
        level_ptr = 0
        n_assign = len(level_assignments)
        level_saturated = {1: False, 2: False, 3: False, 4: False}
        level_stall = {1: 0, 2: 0, 3: 0, 4: 0}
        level_accepted = {1: 0, 2: 0, 3: 0, 4: 0}
        while n_accepted < N_POOL_TARGET and attempts < max_attempts:
            # 跳过已饱和 Level 的剩余槽位 (不消耗 attempts)
            while level_ptr < n_assign and level_saturated[level_assignments[level_ptr]]:
                level_ptr += 1
            if level_ptr >= n_assign:
                break  # 全部槽位处理完毕 (部分 Level 饱和被跳过)
            attempts += 1
            level = level_assignments[level_ptr]
            ovr, desc_str = sample_mass_random(level=level)

            rejected = False
            # 去重 1: D_real 隔离 (防测试集真实突变体参数泄露)
            if is_duplicate_real_mutant(ovr, _MASS_DEFAULTS):
                n_dup_real += 1
                rejected = True
            else:
                # 去重 2: 池内 1% 相对阈值 (相同扰动键集合 + mdt 精确相等 + 全扰动参数相对差 < 1%)
                key_sig = tuple(sorted(ovr.keys()))
                if key_sig:
                    pert_vals = np.array([ovr[k] for k in key_sig], dtype=np.float64)
                    is_dup = False
                    for prev_vals in dedup_buckets.get(key_sig, ()):
                        if ovr.get('mdt') != prev_vals.get('mdt'):
                            continue  # 离散开关参数 (碳源) 要求精确相等
                        arr_prev = np.array([prev_vals[k] for k in key_sig], dtype=np.float64)
                        if np.all(np.abs(pert_vals - arr_prev) <=
                                  DEDUP_REL_TOL * np.maximum(np.abs(arr_prev), 1e-12)):
                            is_dup = True
                            break
                    if is_dup:
                        n_dup_pool += 1
                        rejected = True
                    else:
                        dedup_buckets.setdefault(key_sig, []).append(ovr)

            if rejected:
                # 该 Level 连续被拒计数 (接受时清零); 达上限判定采样空间饱和
                level_stall[level] += 1
                if level_stall[level] >= STALL_LIMIT:
                    level_saturated[level] = True
                    print(f"[al4pde] ⚠️ Level {level} 采样空间饱和 (连续 {STALL_LIMIT} 次被拒), "
                          f"跳过其剩余槽位 (该 Level 已收 {level_accepted[level]} 个)")
                continue

            level_stall[level] = 0
            level_ptr += 1
            n_accepted += 1
            level_accepted[level] += 1
            # 命名对齐数据集格式, 加 AL 随机池标识区分数据集原生样本:
            # 数据集: MASS_{desc}_L{level}_{idx} | 随机候选池: MASS_RAND_{desc}_L{level}_{idx}
            sample_name = f"MASS_RAND_{desc_str}_L{level}_{n_accepted}".replace(" ", "_")
            pool_names.append(sample_name)
            pool_ovrs.append(dict(ovr))
            # 采样元数据: 从 desc_str 解析 (gene, mode) 对, 供 maal/mass/novelty 策略使用
            pool_metas.append({
                'genes': _parse_mass_gene_modes(desc_str),
                'level': int(level),
            })

            # 参数向量 (141 维, 仅 PARAM_KEYS 键, 与数据集 npz 导出一致)
            p_vec = [ovr.get(k, _MASS_DEFAULTS[k]) for k in PARAM_KEYS]
            pool_params.append(p_vec)

            # 构造 IC: 用 initial_conds 覆盖 ovr 中全部 init_* 键
            # (与 mass 版 simulate_mutant 的 y0 写入逻辑完全一致, 支持 16 个 init 键)
            y0 = y0_base.copy()
            for k, idx in _MASS_INIT_Y0_IDX.items():
                if k in ovr:
                    y0[idx] = ovr[k]
            pool_ics.append(y0)

            if n_accepted % max(1, N_POOL_TARGET // 10) == 0:
                print(f"[al4pde] MASS 采样进度: {n_accepted}/{N_POOL_TARGET} "
                      f"({100 * n_accepted // N_POOL_TARGET}%) | 尝试 {attempts} 次 | "
                      f"拦截: real重复={n_dup_real}, 池内重复={n_dup_pool}")

        elapsed = time.time() - t_start
        print(f"[al4pde] MASS 采样完成: {n_accepted} 个候选 ({attempts} 次尝试, "
              f"耗时 {elapsed:.0f}s) | 拦截: real重复={n_dup_real}, 池内重复={n_dup_pool}")
        print(f"[al4pde] Level 实收: L1={level_accepted[1]}, L2={level_accepted[2]}, "
              f"L3={level_accepted[3]}, L4={level_accepted[4]}")

        # v2.8 池规模护栏: 初始集 5000 + 4 轮 × 19000 = 81000 全部取自本池,
        # ODE 仿真失败/极端值过滤还需冗余。低于硬性需求直接报错终止
        # (旧版静默带残池继续 → 5 轮实验设计全部失真, 实测 39387 池事故)
        POOL_REQUIRED = 86000   # 81000 + ~6% 过滤冗余
        if n_accepted < POOL_REQUIRED:
            raise RuntimeError(
                f"[al4pde] 候选池仅 {n_accepted} < 最低需求 {POOL_REQUIRED} "
                f"(初始集 5000 + 4 轮×19000 + 过滤冗余)。原因: Level 配额超过采样空间"
                f"容量 (确定性模式值固定 + 去重阈值过严)。请减小 DEDUP_REL_TOL "
                f"或调整 Level 配额后重跑")
        if n_accepted < N_POOL_TARGET:
            print(f"[al4pde] ⚠️ 候选池 {n_accepted} < 目标 {N_POOL_TARGET} "
                  f"(部分 Level 饱和被跳过), 继续运行")

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

        # 统计 Level 分布 (仅用于打印)
        level_counts = {}
        for meta in pool_metas:
            level_counts[meta['level']] = level_counts.get(meta['level'], 0) + 1

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

        # MASS 专用: 完整 ovr (供 mass 版仿真) + 采样元数据 (供 boundary/maal/mass 策略)
        self._pool_ovr = pool_ovrs
        self._pool_mass_meta = pool_metas

        # 保存归一化统计量 (用于仿真后归一化 Y)
        self._al4pde_mean_y = mean_y
        self._al4pde_std_y = std_y

        # 不再需要磁盘加载相关变量
        self._al4pde_pool_idx = None
        self._al4pde_pool_path = None

        mem_gb = (P_norm.nelement() * 4 + Y_ic_norm.nelement() * 4 +
                  P_tensor.nelement() * 4) / (1024**3)
        level_str = ", ".join(f"L{k}={level_counts.get(k, 0)}"
                              for k in sorted(level_counts))
        print(f"[al4pde] 候选池构建完成: {self._pool_size} 样本 ({level_str}) | "
              f"P_norm(mean={P_norm.mean():.3f}, std={P_norm.std():.3f}), "
              f"Y_ic_norm(mean={Y_ic_norm.mean():.3f}, std={Y_ic_norm.std():.3f}) | "
              f"内存={mem_gb:.2f} GB (只存 P + Y_IC + P_real + ovr + meta, "
              f"完整 Y 选中后 ODE 生成)")

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

    def _is_al4pde_strategy(self):
        """判断是否为 al4pde 框架策略 (使用 ExponentialSchedule 数量机制 + 末轮不选)"""
        return self.strategy in ('al4pde_random', 'al4pde_uncertainty',
                                 'al4pde_diversity', 'maal',
                                 'al4pde_mass', 'al4pde_novelty')

    def _al4pde_scheduled_num_add(self):
        """al4pde 指数递增调度: Iter 0 添加 initial_train_size (如 10k), Iter 1 添加 2x (如 20k)"""
        iter_idx = getattr(self, '_al4pde_al_iter', 0)
        base_add = getattr(self, 'initial_train_size', 10000)
        scheduled = base_add * (2 ** iter_idx)
        return min(scheduled, self._pool_size)

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

        # Step 1: 计算轨迹形状感知特征 (与 diversity/maal 一致)
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
        """计算候选池每个样本的 MASS 生物学分层优先级 (真实机制元数据版)。

        MASS 候选池在采样时已记录每个样本的真实 (gene, mode) 元数据
        (self._pool_mass_meta, 由 _parse_mass_gene_modes 从 desc_str 解析),
        直接查 mass_dataset_generator.BINDING_GROUPS 的生成先验打分:
          score = mean over (gene, mode) 对 of [ P(group) * P(mode|group) ]
          - P(group):    BINDING_GROUPS[gid]['P_bio_G']  (通路频率, 已归一化)
          - P(mode|group): BINDING_GROUPS[gid]['modes'][mode] (机制频率)
        高频通路 + 高频机制 → 高分, 与数据集生成先验一致。

        对比旧版 (从参数值相对 WT 默认值偏移推断 loss/gain/partial 粗机制):
        无法区分 db/PM/CA/APC-A/TAB6-1/CSL 等结构化模式, 且对 init_*/离散开关
        参数 (mdt/Tem1T/B0) 误判; 真实元数据版完全无此问题。

        数据隔离: 仅访问采样元数据 + 静态先验表, 不访问测试集。
        """
        # gene → group 映射 (BINDING_GROUPS 的 name 字段, G1-G27 无重名)
        gene_to_group = {}
        for g_info in MASS_BINDING_GROUPS.values():
            gene_to_group[g_info['name']] = g_info
        max_p_g = max((g['P_bio_G'] for g in MASS_BINDING_GROUPS.values()), default=1.0)

        # 救援基序中未登记进 modes 的 mode (如 ANTAGONISTIC_TUG 的 'pDEL') 给低默认概率
        DEFAULT_MECH_P = 0.02

        scores = np.zeros(self._pool_size)
        n_empty = 0
        for i in range(self._pool_size):
            meta = self._pool_mass_meta[i] if i < len(self._pool_mass_meta) else None
            pairs = meta.get('genes', []) if meta else []
            if not pairs:
                n_empty += 1          # 纯碳源样本 (无突变基因) → score 0
                continue
            contribs = []
            for gene, mode in pairs:
                g_info = gene_to_group.get(gene)
                if g_info is None:
                    contribs.append(0.01)
                    continue
                mech_p = g_info['modes'].get(mode, DEFAULT_MECH_P)
                p_g = g_info['P_bio_G'] / max_p_g     # 通路频率归一到 [0, 1]
                contribs.append(p_g * mech_p)
            scores[i] = float(np.mean(contribs))

        n_scored = self._pool_size - n_empty
        print(f"[al4pde_mass] mass priority (真实机制元数据): scored {n_scored}/"
              f"{self._pool_size} 候选 | range: {scores.min():.4f}~{scores.max():.4f}, "
              f"mean={scores.mean():.4f}")
        return scores

    # ====== al4pde_novelty 策略 (纯随机候选池 + 组合新颖性主动选择) ======
    def _al4pde_novelty_sample(self, trainer, pl_module, dm, device, current_epoch):
        """al4pde_novelty: 从纯随机候选池中主动挖掘"定向采样等效价值"样本

        研究背景: 定向采样 (rescue motif) 训练集让模型在 real mutants 测试集上
        表现更好, 因其覆盖了 "驱动突变+抑制突变" 的上位性组合。本策略在完全
        随机的候选池上 (无任何定向样本), 通过三个零泄露信号逼近该价值:

        acquisition = w_n * novelty + w_f * family_unc + w_d * diversity
        - novelty (组合新颖性): 候选的突变单元上位性组合在训练集中的覆盖稀缺度
          (单元=(基因,突变类型); 全子集评估: 单元/对/三元/.../完整组合, 尺寸加权
          w_s=2^-|s-2|, 详见 _compute_combination_novelty_scores),
          优先补齐训练集未覆盖的组合 (rescue motif 价值的结构化代理)
        - family_unc (家族不确定性): 训练集 MC dropout 按首基因家族分组的
          平均不确定性, 模型不自信的家族优先补充 (hard family 信号)
        - diversity (轨迹多样性): ODE 真实轨迹形状特征到训练集的最小距离,
          保证参数空间 + mass pattern 双重覆盖 (参数距离与 pattern 一起考量)

        数据隔离 (零泄露): 三个信号仅使用 dm.train_dataset_synth (当前训练集) 与
        self._pool_* (随机候选池自带元数据), 不访问 real mutants 的参数/轨迹/
        性能, 训练过程对 126 个真实突变体完全盲。

        权重: --al_novelty_weight (默认 0.6, novelty 是 rescue motif 价值的主要
        结构化代理, 主导选样) + --al_family_weight (默认 0.2),
        diversity 权重 = 1 - novelty_weight - family_weight 自动归一化。
        """
        if self._pool_size == 0:
            print("[al4pde_novelty] 候选池已空, 跳过本轮 AL")
            return None

        # Step 1: 组合新颖性 (候选 vs 训练集的突变单元全子集覆盖稀缺度)
        print(f"[al4pde_novelty] Step 1: 计算组合新颖性 (单元级全子集覆盖, 池: {self._pool_size})...")
        novelty_scores = self._compute_combination_novelty_scores(dm)
        novelty_scores = self._normalize_array(novelty_scores)

        # Step 2: 家族不确定性 (训练集 MC dropout → 家族分组 → 映射到候选)
        print("[al4pde_novelty] Step 2: 计算家族不确定性 (训练集 MC dropout)...")
        family_unc = self._compute_train_family_uncertainty(pl_module, dm, device)
        family_scores = self._map_family_unc_to_pool(family_unc)
        family_scores = self._normalize_array(family_scores)

        # Step 3: 轨迹多样性 (ODE 真实轨迹形状特征 min-dist, 与 diversity 策略一致)
        print("[al4pde_novelty] Step 3: 计算轨迹多样性 (ODE 形状特征 min-dist)...")
        pool_features = self._compute_pool_features_shape_aware(pl_module, dm, device)
        train_features = self._compute_train_features_shape_aware(pl_module, dm, device)
        pool_features, train_features = self._standardize_shape_features(pool_features, train_features)
        diversity_scores = self._compute_diversity_scores(pool_features, train_features)
        diversity_scores = self._normalize_array(diversity_scores)

        # Step 4: 加权 acquisition (权重自动归一化: w_d = 1 - w_n - w_f)
        w_n = float(np.clip(self.novelty_weight, 0.0, 1.0))
        w_f = float(np.clip(self.family_weight, 0.0, 1.0 - w_n))
        w_d = max(0.0, 1.0 - w_n - w_f)
        acquisition = w_n * novelty_scores + w_f * family_scores + w_d * diversity_scores
        print(f"[al4pde_novelty] Step 4: acquisition = {w_n:.2f}*novelty + {w_f:.2f}*family "
              f"+ {w_d:.2f}*diversity | range: {acquisition.min():.4f}~{acquisition.max():.4f}")

        # Step 5: top_k 选取
        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        sel_idx = np.argsort(acquisition)[-n_select:]
        print(f"[al4pde_novelty] Step 5: top_k 选取 {n_select} 个样本")
        return self._build_broadcast_from_pool(sel_idx, dm)

    def _compute_combination_novelty_scores(self, dm):
        """计算候选池每个样本的组合新颖性 (v2: 突变单元级全子集覆盖稀缺度)

        生物学动机: rescue motif 定向采样的价值在于覆盖 "驱动突变+抑制突变"
        的特定上位性组合; real mutants 大多为这类多基因上位性组合。本函数在
        不访问 real mutants 的前提下, 度量每个随机候选能带来的"新组合"。

        v2 升级 (相对 v1 "基因对覆盖" 的三点改进):
        1. 突变单元细化为 (基因, 突变类型): CLB2[KO] 与 CLB2[GAL-OE] 是不同
           单元, 覆盖判断按单元进行 —— 不再按基因名合并, 避免训练集见过
           CLB2[KO] 就把候选的 CLB2[GAL-OE] 误判为已覆盖
        2. 组合空间从仅基因对扩展到全子集: 对候选的 k 个单元, 评估
           s=1..k 全部尺寸的组合 (单元/对/三元/.../完整组合), 覆盖完整
           上位性组合结构
        3. 尺寸加权 w_s = 2^{-|s-2|}: 二元上位性 (s=2) 是 rescue motif 价值
           核心取权重 1; 单突变类型 (s=1) 与三元组 (s=3) 取 0.5, 更高阶逐级
           减半 —— 防止必然未见的高阶组合饱和分数、挤掉低阶判别信号

        公式: novelty = Σ_{s=1}^{k} w_s · u_s / C(k,s)  /  Σ_{s=1}^{k} w_s
          k     = 候选的不同突变单元数
          u_s   = 候选的 s 元单元子集中训练集未出现过的数量
          C(k,s)= 候选 s 元子集总数 (u_s/C(k,s) 即该阶"未见密度")
          → 各阶未见密度的加权和归一, 范围 [0,1], 不同 k 候选可公平比较

        特殊情况:
          - 纯碳源候选 (无突变单元): novelty = 0
          - 复合机制 CLB2[GAL-OE+db] 展开为两个单元 (GAL-OE, db), 同基因
            机制组合同样计入上位性组合
          - k ≤ 12 全子集枚举 (实际上限 k=11: 6 绑定组 + 5 复合机制);
            k > 12 仅评估 1..4 阶 + 完整组合 (防御性兜底, 理论不可达)

        数据隔离: 仅解析 dm.train_dataset_synth 样本名 + self._pool_mass_meta,
        不访问测试集。
        """
        from itertools import combinations

        # ---------- Step 1: 训练集已覆盖的突变单元子集 (bitmask 整数键, 省内存) ----------
        if hasattr(dm, '_mutant_names') and hasattr(dm, '_train_idx_synth'):
            train_names = [str(dm._mutant_names[i]) for i in dm._train_idx_synth]
        else:
            train_names = [f"Train_{i}" for i in range(len(dm.train_dataset_synth))]

        unit_ids = {}   # (gene, mode) -> int id (词表约百余个, 子集压成 bitmask int)
        def _uid(gm):
            if gm not in unit_ids:
                unit_ids[gm] = len(unit_ids)
            return unit_ids[gm]

        MAX_FULL_ENUM_UNITS = 12

        def _sizes_of(k):
            # 评估的尺寸集: k≤12 全枚举 1..k; 超限仅 1..4 阶 + 完整组合
            return range(1, k + 1) if k <= MAX_FULL_ENUM_UNITS else list(range(1, 5)) + [k]

        train_combos = {}   # {size: set(bitmask)}: 训练集出现过的该尺寸全部子集
        for name in train_names:
            desc = _mass_desc_of(name)
            if desc is None:
                continue
            units = sorted({_uid(gm) for gm in _parse_mass_gene_modes(desc)})
            if not units:
                continue
            for s in _sizes_of(len(units)):
                bucket = train_combos.setdefault(s, set())
                for combo in combinations(units, s):
                    mask = 0
                    for u in combo:
                        mask |= 1 << u
                    bucket.add(mask)

        # ---------- Step 2: 候选逐阶加权评分 ----------
        def _size_weight(s):
            # w_1=0.5, w_2=1.0 (核心: 二元上位性), w_3=0.5, 更高阶逐级减半
            return 2.0 ** (-abs(s - 2))

        scores = np.zeros(self._pool_size)
        n_empty = 0
        for i in range(self._pool_size):
            meta = self._pool_mass_meta[i] if i < len(self._pool_mass_meta) else None
            pairs = meta.get('genes', []) if meta else []
            units = sorted({_uid(gm) for gm in pairs})
            if not units:
                n_empty += 1          # 纯碳源样本 → 无组合新颖性
                continue
            w_sum = 0.0
            nov_sum = 0.0
            for s in _sizes_of(len(units)):
                seen_set = train_combos.get(s)
                total = 0
                unseen = 0
                for combo in combinations(units, s):
                    mask = 0
                    for u in combo:
                        mask |= 1 << u
                    total += 1
                    if seen_set is None or mask not in seen_set:
                        unseen += 1
                w = _size_weight(s)
                nov_sum += w * (unseen / total)
                w_sum += w
            scores[i] = nov_sum / w_sum

        n_scored = self._pool_size - n_empty
        n_units_seen = len(train_combos.get(1, ()))
        n_unit_pairs = len(train_combos.get(2, ()))
        print(f"[al4pde_novelty] 组合新颖性 v2 (单元级全子集, w_s=2^-|s-2|): 训练集覆盖 "
              f"{n_units_seen} 突变单元 / {n_unit_pairs} 单元对 | scored {n_scored}/{self._pool_size} 候选 | "
              f"range: {scores.min():.4f}~{scores.max():.4f}, mean={scores.mean():.4f}")
        return scores

    def _map_family_unc_to_pool(self, family_uncertainty):
        """将训练集家族不确定性映射到候选池样本 (v2.7 多基因版)

        候选的全部突变基因各自查家族不确定性百分位 [0,1], 取均值:
        - 候选 CLB2[KO]+CLN2[GAL-OE] → mean(pct[CLB2], pct[CLN2])
          (样本同时携带两个家族的信息, 补充任一家族均有价值)
        - 训练集中未出现的基因 → 该基因计 0.5 (中性, 由 novelty 主导其排序)
        - 纯碳源候选 → 0 (无家族信号)
        数据隔离: 输入 family_uncertainty 仅来自训练集参数扰动集成。
        """
        fams = sorted(family_uncertainty.keys())
        if len(fams) == 0:
            return np.zeros(self._pool_size)
        vals = np.array([family_uncertainty[f] for f in fams], dtype=np.float64)
        # 家族不确定性 → 百分位排名 [0,1] (消除量纲, 最不确定的家族 → 1.0)
        order = np.argsort(np.argsort(vals))
        ranks = order / max(1, len(fams) - 1)
        fam_pct = dict(zip(fams, ranks))

        scores = np.zeros(self._pool_size)
        for i in range(self._pool_size):
            meta = self._pool_mass_meta[i] if i < len(self._pool_mass_meta) else None
            pairs = meta.get('genes', []) if meta else []
            if pairs:
                # 全部突变基因 (去重) 的家族百分位均值 (v2.7: 不再只看首个基因)
                genes = []
                for gene, _mode in pairs:
                    if gene not in genes:
                        genes.append(gene)
                scores[i] = float(np.mean([fam_pct.get(g, 0.5) for g in genes]))
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
        train_X, train_P, _ = dm.train_dataset_synth.tensors[:3]
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

    # ====== v2.2 方案 A: 分组特征权重 (al4pde_diversity/maal/novelty 共用) ======
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

    def _compute_pool_features_shape_aware(self, pl_module, dm, device, batch_size=2500):
        """返回候选池的轨迹形状感知特征 (raw, 未标准化)
        通过模型 forward_ic_time 快速批量预测生成轨迹特征 (无需前端耗时 ODE 求解)。
        """
        if self._pool_shape_features_cached is not None:
            return self._pool_shape_features_cached

        pl_module.eval()
        out_features = None
        with torch.no_grad():
            for i in range(0, self._pool_size, batch_size):
                end_idx = min(i + batch_size, self._pool_size)
                X_batch = self._get_pool_X_batch(slice(i, end_idx)).to(device)
                P_batch = self._pool_P[i:end_idx].to(device)
                preds = pl_module.forward_ic_time(X_batch, P_batch)
                feat = self._extract_shape_features(preds, P_batch).cpu()
                del X_batch, P_batch, preds
                if out_features is None:
                    out_features = torch.empty((self._pool_size, feat.shape[1]), dtype=torch.float32)
                out_features[i:end_idx] = feat
                del feat
        pl_module.train()
        return out_features

    def _compute_train_features_shape_aware(self, pl_module, dm, device, batch_size=200):
        """从训练集真实 Y 轨迹提取形状特征 (raw, 未标准化)

        改进: 训练集已有真实轨迹 (数据集中的 Y), 直接用, 不需要模型预测。
        """
        _, train_P, train_Y = dm.train_dataset_synth.tensors[:3]
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
        """原生 PyTorch 向量化 MaxDist 贪心最大距离选样 (等价于 bmdal_reg, 耗时由几百秒降至几秒, 兼具 NumPy 2 兼容性)"""
        n_pool = pool_features.shape[0]
        n_select = min(n_select, n_pool)
        if n_select <= 0:
            return np.array([], dtype=int)

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pool_feat = pool_features.to(device)
            train_feat = train_features.to(device)
        except RuntimeError:
            device = torch.device('cpu')
            pool_feat = pool_features.cpu()
            train_feat = train_features.cpu()

        # 1. 计算每个候选样本到当前训练集的最小欧氏距离
        chunk_size = 50000
        min_dist_to_train = torch.full((n_pool,), float('inf'), device=device)
        with torch.no_grad():
            for i in range(0, n_pool, chunk_size):
                end_c = min(i + chunk_size, n_pool)
                p_chunk = pool_feat[i:end_c]
                dists = torch.cdist(p_chunk, train_feat)  # (chunk, M)
                min_dist_to_train[i:end_c] = dists.min(dim=1)[0]
                del dists, p_chunk

            # 2. 贪心最大距离选择
            selected_indices = []
            curr_min_dist = min_dist_to_train.clone()

            update_chunk = 100000
            for step in range(n_select):
                best_idx = torch.argmax(curr_min_dist).item()
                selected_indices.append(best_idx)
                curr_min_dist[best_idx] = -float('inf')
                if step == n_select - 1:
                    break
                best_point = pool_feat[best_idx:best_idx + 1]  # (1, D)
                for i in range(0, n_pool, update_chunk):
                    end_u = min(i + update_chunk, n_pool)
                    p_chunk = pool_feat[i:end_u]
                    dist_to_new = torch.norm(p_chunk - best_point, dim=1)
                    curr_min_dist[i:end_u] = torch.minimum(curr_min_dist[i:end_u], dist_to_new)

        return np.array(selected_indices, dtype=int)

    def _compute_diversity_scores(self, pool_features, train_features, chunk=50000):
        """计算每个候选样本到训练集的最小欧氏距离 (diversity score), 支持 GPU 高速批处理 (流式传输, 显存 < 100MB)"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            train_feat = train_features.to(device)
        except RuntimeError:
            device = torch.device('cpu')
            train_feat = train_features.cpu()

        n_pool = pool_features.shape[0]
        min_dists = torch.empty((n_pool,), dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, n_pool, chunk):
                end_i = min(i + chunk, n_pool)
                p_chunk = pool_features[i:end_i].to(device)
                dist = torch.cdist(p_chunk, train_feat)
                min_dists[i:end_i] = dist.min(dim=1)[0].cpu()
                del dist, p_chunk
        return min_dists.numpy()

    def _compute_train_family_uncertainty(self, pl_module, dm, device, n_mc_samples=5, max_train_subset=2000):
        """对训练集做参数扰动集成 (pseudo MC dropout), 按家族分组计算平均不确定性

        机制 (三步):
        1. 从当前训练集随机抽最多 max_train_subset 个样本
        2. 每个样本 forward n_mc_samples 次, 每次给归一化参数向量 P 加 2% 高斯
           噪声 (输入扰动, 模型保持 eval), 取 n_mc_samples 次预测的方差
           (对时间维+变量维取 mean) → 单样本不确定性
           直觉: 模型在某参数邻域预测对微小扰动剧烈抖动 → 该区域欠拟合/外推
        3. 按基因家族分组取平均 → {gene: mean_uncertainty} (v2.7 多基因版:
           每个样本归属其全部突变基因, 纯碳源样本归入 'CARBON' 伪家族)

        注意: 实现是输入参数扰动集成 (局部敏感性分析), 不是真正开启 dropout 层;
        效果等价于度量模型对参数空间各区域的预测稳定度。

        数据隔离原则: 仅使用 dm.train_dataset_synth, 不访问 dm.test_dataset_real。

        Args:
            pl_module: Lightning 模型
            dm: 数据模块
            device: 计算设备
            n_mc_samples: MC 采样次数 (默认 5)
            max_train_subset: 训练集子集上限 (加速, 默认 2000)

        Returns:
            family_uncertainty: {family_name: mean_uncertainty}
        """
        train_X, train_P, _ = dm.train_dataset_synth.tensors[:3]
        n_train = len(train_P)

        # 获取训练集 mutant_names (通过 dm._train_idx_synth 映射)
        if hasattr(dm, '_mutant_names') and hasattr(dm, '_train_idx_synth'):
            train_names = [str(dm._mutant_names[i]) for i in dm._train_idx_synth]
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

        # 按基因家族分组计算平均不确定性 (v2.7 多基因版)
        # 每个训练样本的不确定性归属其全部突变基因:
        #   'MASS_RAND_CLB2[KO]+CLN2[GAL-OE]_L1_123' 同时计入 CLB2 与 CLN2 两个家族
        # (旧版仅取首个基因, 多基因样本的其余家族信号被丢弃)
        # 纯碳源样本 (无突变基因) 归入 'CARBON' 伪家族。
        family_unc_list = {}
        for i, name in enumerate(train_names_sub):
            genes = _mass_sample_genes(name)
            fams = genes if genes else ['CARBON']
            for fam in fams:
                family_unc_list.setdefault(fam, []).append(uncertainties[i])

        family_uncertainty = {f: float(np.mean(v)) for f, v in family_unc_list.items()}

        print(f"[MAAL] 训练集家族不确定性: {len(family_uncertainty)} 个家族 | "
              f"unc range: {min(family_uncertainty.values()):.4f}~{max(family_uncertainty.values()):.4f} | "
              f"子集大小: {len(train_P_sub)}/{n_train}")
        return family_uncertainty

    def _map_family_priority_train_based(self, family_uncertainty, dm):
        """将训练集家族不确定性映射到候选池样本的 target_priority (完全基于训练集, 零泄露)

        v2.7 多基因版: 候选的全部突变基因各自计算三维信号后取均值
        (候选 CLB2[KO]+CLN2[GAL-OE] 同时按 CLB2 与 CLN2 两个家族评估)。

        target_priority = 0.5*同家族不确定性百分位 + 0.3*候选到训练集同家族参数距离 + 0.2*家族覆盖度
        (三个维度均为: 候选全部突变基因的信号均值; 纯碳源候选按 'CARBON' 伪家族评估)

        数据隔离: 不访问 dm.test_dataset_real, 仅用 dm.train_dataset_synth。
        - 维度1 (0.5): 训练集中同家族样本的不确定性 (模型对该家族不自信 → 优先)
        - 维度2 (0.3): 候选样本到训练集同家族参数中心的距离 (越远 → 训练集覆盖不足 → 优先)
        - 维度3 (0.2): 候选池中同家族样本数 (越少 → 覆盖不足 → 优先)

        Args:
            family_uncertainty: {gene: mean_uncertainty} 来自训练集 (多基因聚合)
            dm: 数据模块 (仅用 train_dataset_synth)

        Returns:
            priorities: (N,) numpy array, 每个候选样本的 target_priority
        """
        priorities = np.zeros(self._pool_size)

        # 1. 不确定性百分位排名 [0, 1]
        unc_values = np.array(list(family_uncertainty.values()))
        unc_percentile = {}
        for name, unc in family_uncertainty.items():
            unc_percentile[name] = float((unc_values <= unc).mean())

        # 2. 家族覆盖度统计 (候选池中每个基因家族的样本数; v2.7: 每个候选
        #    为其全部突变基因各计 1 次, 纯碳源候选计入 'CARBON')
        family_counts = {}
        for name in self._pool_mutant_names:
            genes = _mass_sample_genes(name)
            fams = genes if genes else ['CARBON']
            for fam in fams:
                family_counts[fam] = family_counts.get(fam, 0) + 1
        max_family_count = max(family_counts.values()) if family_counts else 1

        # 3. 训练集参数按基因家族分组 (用于参数距离计算) — 完全来自训练集
        #    v2.7: 每个训练样本的参数归属其全部突变基因 (多基因样本计入多个家族)
        train_P_np = dm.train_dataset_synth.tensors[1].cpu().numpy()
        if hasattr(dm, '_mutant_names') and hasattr(dm, '_train_idx_synth'):
            train_names = [str(dm._mutant_names[i]) for i in dm._train_idx_synth]
        else:
            train_names = [f"Train_{i}" for i in range(len(train_P_np))]

        train_family_params = {}
        for i, name in enumerate(train_names):
            genes = _mass_sample_genes(name)
            fams = genes if genes else ['CARBON']
            for fam in fams:
                train_family_params.setdefault(fam, []).append(train_P_np[i])
        # 每个家族取平均参数向量 (代表该家族的参数中心)
        train_family_center = {f: np.mean(params, axis=0) for f, params in train_family_params.items()}

        # 候选池参数 (归一化空间)
        pool_P_np = self._pool_P.cpu().numpy() if hasattr(self._pool_P, 'cpu') else np.array(self._pool_P)

        for i, name in enumerate(self._pool_mutant_names):
            genes = _mass_sample_genes(name)
            fams = genes if genes else ['CARBON']

            # 维度 1: 全部突变基因的同家族不确定性百分位均值 [0, 1]
            # (基因在训练集中未出现 → 该基因计 0.5 中性值)
            base_priority = float(np.mean([unc_percentile.get(f, 0.5) for f in fams]))

            # 维度 2: 全部突变基因的参数中心距离均值 [0, 1] (越远 → 覆盖不足 → 优先)
            # (基因在训练集中无样本 → 该基因计 0.5 默认值)
            dist_factors = []
            for f in fams:
                if f in train_family_center:
                    dist = np.linalg.norm(pool_P_np[i] - train_family_center[f])
                    # dist/(1+dist): dist 越大越接近 1, 鼓励探索训练集未覆盖的参数区域
                    dist_factors.append(dist / (1.0 + dist))
                else:
                    dist_factors.append(0.5)
            distance_factor = float(np.mean(dist_factors))

            # 维度 3: 全部突变基因的家族覆盖度均值 [0, 1] (池中样本越少 → 优先)
            coverage_factor = float(np.mean(
                [1.0 - family_counts.get(f, 1) / max_family_count for f in fams]))

            # 综合 priority: 0.5*不确定性 + 0.3*参数距离 + 0.2*覆盖度
            priorities[i] = 0.5 * base_priority + 0.3 * distance_factor + 0.2 * coverage_factor

        print(f"[MAAL] target_priority 训练集版 (v2.7 多基因聚合): "
              f"0.5*同家族不确定性 + 0.3*参数距离 + 0.2*覆盖度")
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
        1. 从候选池提取选中样本的完整 ovr (MASS 候选池) 或未归一化参数
        2. 调用 solve_single_mass_bridge ODE 仿真生成轨迹 (MASS 引擎, 支持全部 init_* 键)
        3. 动态浓度剔除 (超训练集 max×2 视为 monster)
        4. 归一化 Y/P, 从 Y_ic 构造 X
        5. 从候选池移除已选样本
        """
        sel_idx = np.array(sel_idx, dtype=np.int64)
        n_sel = len(sel_idx)

        # 从候选池提取选中样本的参数
        sel_P_real = self._pool_P_real[sel_idx].numpy()   # (N, 141) 未归一化
        sel_names = self._pool_mutant_names[sel_idx]
        # MASS 候选池: 优先用完整 ovr 仿真 (含 init_Tem1 等 lhs 引擎不认识的键)
        sel_ovrs = None
        if self._pool_ovr is not None:
            sel_ovrs = [dict(self._pool_ovr[i]) for i in sel_idx]

        # ODE 仿真
        print(f"[al4pde] ODE 仿真生成 {n_sel} 个选中样本的轨迹 (MASS 引擎)...")
        tasks = []
        for i in range(n_sel):
            ovr = sel_ovrs[i] if sel_ovrs is not None else {k: v for k, v in zip(PARAM_KEYS, sel_P_real[i].tolist())}
            tasks.append((str(sel_names[i]), ovr))

        sim_results = [None] * n_sel
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        workers = min(6, os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            done = 0
            for idx, res in enumerate(executor.map(_solve_single_mass_task_wrapper, tasks, chunksize=50)):
                sim_results[idx] = res
                done += 1
                if done % max(1, n_sel // 10) == 0 or done == n_sel:
                    print(f"  ODE 仿真进度: {done}/{n_sel} ({100*done//n_sel}%)")

        # 动态浓度剔除 (阈值参考 real mutants, 复用物理空间的 real mutants 最大值)
        if hasattr(dm, '_Y') and hasattr(dm, '_test_idx_real'):
            real_envelope = dm._Y[dm._test_idx_real].max(dim=-1)[0].max(dim=0)[0].cpu().numpy()
        else:
            Y_real_phys = dm.denormalize_y(dm.test_dataset_real.tensors[2].cpu())
            real_envelope = Y_real_phys.max(dim=-1)[0].max(dim=0)[0].cpu().numpy()
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
        if self._pool_ovr is not None:
            self._pool_ovr = [o for o, m in zip(self._pool_ovr, mask) if m]
        if self._pool_mass_meta is not None:
            self._pool_mass_meta = [mt for mt, m in zip(self._pool_mass_meta, mask) if m]
        self._pool_size = mask.sum()

        if not valid_Y_real:
            print(f"[al4pde] 所有 {n_sel} 个选中样本均未通过物理验证, 本轮无样本加入训练集")
            print(f"[al4pde] 候选池剩余: {self._pool_size} 样本")
            return None

        # 归一化 Y 和 P (仿真输出为真实物理值空间, 需按照 dm.norm_type 归一化以匹配训练集)
        norm_type = getattr(dm, 'norm_type', 'log1p')
        mean_y = dm.mean_y.cpu()  # (1, 39, 1)
        std_y = dm.std_y.cpu()    # (1, 39, 1)
        p_mean = dm.p_mean.cpu()  # (1, 141)
        p_std = dm.p_std.cpu()    # (1, 141)
        p_std_safe = torch.where(p_std > 1e-5, p_std, torch.ones_like(p_std))

        Y_list = []
        for y_real in valid_Y_real:
            y_t = torch.tensor(y_real, dtype=torch.float32)  # (39, 500)
            if norm_type in ('log1p', 'asinh_z'):
                y_trans = torch.log1p(torch.clamp(y_t, min=0.0))
            else:
                y_trans = y_t
            y_norm = (y_trans - mean_y.squeeze(0)) / std_y.squeeze(0)
            y_norm = torch.nan_to_num(y_norm, nan=0.0, posinf=10.0, neginf=-10.0)
            Y_list.append(y_norm)

        P_list = []
        for p_real in valid_P_real:
            p_t = torch.tensor(p_real, dtype=torch.float32).unsqueeze(0)  # (1, 141)
            if norm_type == 'asinh_z':
                z = (p_t - p_mean) / p_std_safe
                p_norm = torch.asinh(z).squeeze(0)
            else:
                p_norm = ((p_t - p_mean) / p_std_safe).squeeze(0)
            p_norm = torch.nan_to_num(p_norm, nan=0.0, posinf=10.0, neginf=-10.0)
            P_list.append(p_norm)

        print(f"[al4pde] 仿真完成: {len(valid_Y_real)}/{n_sel} 通过验证 | "
              f"候选池剩余: {self._pool_size}")

        return {
            'X': X_tensor,
            'Y': torch.stack(Y_list),
            'P': torch.stack(P_list),
            'Y_real': valid_Y_real,
            'P_real': valid_P_real,
            'names': valid_names,
            'labels': valid_labels,
            'sample_plots': {}
        }
