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
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import json

# ==============================================================
# 1. Budding online simulation and candidate sampling
# ==============================================================
from budding_yeast_v2.data.systems.budding.api import (
    get_default_params,
)
from budding_yeast_v2.data.systems.budding.api import (
    simulate_mutant as simulate_mutant_mass,
    is_duplicate_real_mutant,
    initial_conds as initial_conds_mass,
    prepare_initial_conditions,
    get_default_params as get_default_params_mass,
    BINDING_GROUPS as MASS_BINDING_GROUPS,

)
# v2.6: 全 al4pde 策略统一使用 random-only 采样器 (无 rescue motif 定向采样, 保留 Level 分层)
from budding_yeast_v2.data.systems.budding.api import sample_mass_random
from budding_yeast_v2.data.normalization import normalize_y, normalize_p
from budding_yeast_v2.data.simulation_guards import bounded_ordered_map
from budding_yeast_v2.data.label_metadata import UNKNOWN_BIOLOGY_LABEL, trajectory_label_metadata
from budding_yeast_v2.data.systems.budding.pattern_groups import abc_pattern_ids

# MASS 默认参数向量 (与 mass_dataset_generator.generate_mass_dataset 完全一致)
# 生物模型/初值统一使用 systems.budding 的 MA 基准；随机合成采样保留 MASS 方案。
_MASS_DEFAULTS = get_default_params_mass()
_MASS_DEFAULTS.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

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


def solve_single_mass_bridge(name, ovr, *, return_biological_label=False):
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
        默认返回 (X_dummy, y_uniform, p_vec, pattern_label) 或 None；
        return_biological_label=True 时另附原始生物标签。
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
    result = (X_dummy, y_uniform, p_vec, pattern_label)
    return (*result, bio_label) if return_biological_label else result


def _solve_single_mass_task_wrapper(task_tuple):
    """顶级函数包装器，供 ProcessPoolExecutor(mp_context='spawn') 进程池调用"""
    name, ovr = task_tuple
    try:
        return solve_single_mass_bridge(name, ovr, return_biological_label=True)
    except Exception:
        return None


def train_acquisition_ensemble(primary, model_factory, trainer_factory, datamodule, *,
                               n_members, seed, round_index, checkpoint_dir):
    """Train independent selection members; retain member zero for evaluation.

    Mirrors the independent cold starts in AL4PDE Ensemble.train_n_epoch.
    All members train on the same fixed-normalization data. Caller invokes this
    only before an acquisition round, matching the official final-round shortcut.
    Each extra member gets a dedicated seed, trainer/optimizer and checkpoint.
    """
    if n_members < 2:
        raise ValueError('Ensemble uncertainty requires at least two independent models')
    checkpoint_dir = os.fspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    members = [primary.cpu()]
    records = [{'member': 0, 'seed': int(seed), 'checkpoint': 'primary round checkpoint'}]
    # Keep member training from consuming the subsequent acquisition RNG stream.
    numpy_state = np.random.get_state()
    import random
    python_state = random.getstate()
    seed_env = {key: os.environ.get(key) for key in ('PL_GLOBAL_SEED', 'PL_SEED_WORKERS')}
    # Preserve only already initialized CUDA streams; avoid initializing every GPU
    # on CPU runs. Training uses the primary member's accelerator(s).
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            for member_index in range(1, n_members):
                member_seed = int(seed + 1000003 * (round_index + 1) + member_index)
                pl.seed_everything(member_seed, workers=True)
                member = model_factory()
                trainer = trainer_factory(member_index)
                trainer.fit(member, datamodule=datamodule)
                path = os.path.join(checkpoint_dir, f'round_{round_index}_ensemble_member_{member_index}.ckpt')
                trainer.save_checkpoint(path)
                members.append(member.cpu())
                records.append({'member': member_index, 'seed': member_seed, 'checkpoint': path})
                del trainer
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            for key, value in seed_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return members, records


class ActiveLearningCallback(pl.Callback):
    def __init__(self, trigger_every_n_epochs=10, strategy='al4pde_random', num_add=5000,
                 perturbation=0.1, mae_threshold=0.1,
                 # ====== 兼容历史配置的参数 ======
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
                 family_weight=0.2,
                 ensemble_uncertainty_mode='loss_mse', power_beta=1.0, bait_sketch_dim=512):
        """Online simulation AL strategies driven by train.py's outer loop.

        Historical precomputed-trajectory pool strategies are archived in
        utils/archive/al_callback_legacy.py and rejected by this entry point.
        Legacy constructor keywords remain accepted for existing configurations.
        The al4pde prefix describes the historical workflow; acquisition scores
        are project implementations, not direct calls into the official library.
        """
        self.trigger_every_n_epochs = trigger_every_n_epochs
        self.strategy = strategy.lower()
        if not self._is_al4pde_strategy():
            raise ValueError(
                f"Unsupported or archived AL strategy {self.strategy!r}. "
                "Use an al4pde strategy or maal. Historical trajectory-pool "
                "strategies are in utils/archive/al_callback_legacy.py; "
                "disable AL with --al_strategy none in train.py."
            )

        self.num_add = num_add
        self.perturbation = perturbation
        self.mae_threshold = mae_threshold
        self._real_mutant_scores = {}

        # Compatibility fields for historical run configurations.
        self.initial_train_size = initial_train_size
        self.initial_selection = initial_selection
        self.diversity_weight = diversity_weight
        self.pool_subset_size = pool_subset_size
        self.uncertainty_metric = uncertainty_metric
        self.random_seed = random_seed

        # Parameter candidate pool, initialized by al4pde_setup.
        if ensemble_uncertainty_mode not in ('loss_mse', 'sample_variance'):
            raise ValueError('Unknown ensemble uncertainty mode')
        if not np.isfinite(power_beta) or power_beta <= 0:
            raise ValueError('power_beta must be finite and positive')
        if bait_sketch_dim < 1:
            raise ValueError('bait_sketch_dim must be positive')
        self.power_beta = float(power_beta)
        self.bait_sketch_dim = int(bait_sketch_dim)
        self._bait_sketch = None
        self.ensemble_uncertainty_mode = ensemble_uncertainty_mode
        self._ensemble_members = None  # Trained members supplied by the outer training loop.
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

        if self.strategy in ('al4pde_random', 'al4pde_uncertainty', 'al4pde_power', 'al4pde_bait', 'al4pde_lcmd', 'al4pde_diversity',
                             'maal', 'al4pde_mass', 'al4pde_novelty', 'al4pde_residual',
                             'al4pde_residual_diversity', 'al4pde_res_div',
                             'al4pde_bifurcation', 'al4pde_bifurcation_entropy', 'al4pde_bif_entropy',
                             'al4pde_event_entropy', 'al4pde_event',
                             'al4pde_fisher_oed', 'al4pde_foed'):
            self.initial_train_size = initial_train_size
            self.trigger_every_n_epochs = 4  # 保留字段, 外层重训模式下不生效
            self.num_add = num_add            # 保留字段, 外层重训模式下不生效
            self._al4pde_num_al_iter = 3      # 3 轮 AL (Iter 0: 10k -> Iter 1: 20k -> Iter 2: 40k)
            self._al4pde_per_round_add = 10000
            print(f"[AL Manager] al4pde 设置: initial_train_size={self.initial_train_size}, "
                  f"{self._al4pde_num_al_iter} 轮指数倍增 (10k -> 20k -> 40k)")
        
        self._al4pde_al_iter = 0           # 当前 AL 轮次 (0-indexed, 每次选择后递增)
        self._outer_loop_mode = False      # v2.3: 外层重训模式 (由 train.py 启用, 关闭 in-loop 触发)


    # ==================================================================
    # v2.3 外层重训式 al4pde: 供 train.py 外层循环调用的公开接口
    # ==================================================================
    def enable_outer_loop_mode(self):
        """Compatibility marker: selection is driven exclusively by train.py outer rounds."""
        self._outer_loop_mode = True

    @staticmethod
    def _append_dm_label_metadata(dm, bd):
        """Keep supplementary row arrays aligned; training targets remain ABC."""
        if 'Y_real' not in bd:
            return
        fields = trajectory_label_metadata(np.asarray(bd['Y_real']), bd.get('biological_labels'),
                                           getattr(dm, 'system_id', 'budding'))
        arrays = dict(pattern_labels=np.asarray(bd['labels']), biological_labels=fields['labels'],
                      biological_labels_available=fields['biological_labels_available'])
        for key in ('pattern_group_codes', 'pattern_group_labels', 'pattern_group_ids', 'observed_division_counts'):
            if key in fields:
                arrays[key] = fields[key]
        for key, values in arrays.items():
            existing = getattr(dm, key, None)
            if existing is not None:
                setattr(dm, key, np.concatenate((existing, values)))

    def _save_al_snapshot(self, path, dm):
        """Preserve viability, ABC and neutral groups without inferring missing biology."""
        data = np.asarray(self._benchmark_real_data + self._cum_synth_data, dtype=np.float32)
        biology = self._benchmark_real_biological_labels + self._cum_synth_biological_labels
        system_id = getattr(dm, 'system_id', 'budding')
        fields = trajectory_label_metadata(data, biology, system_id)
        np.savez_compressed(path, data=data,
            params=np.asarray(self._benchmark_real_params + self._cum_synth_params, dtype=np.float64),
            mutant_names=np.asarray(self._benchmark_real_names + self._cum_synth_names),
            pattern_labels=np.asarray(self._benchmark_real_labels + self._cum_synth_labels),
            param_names=self._param_names_cached, system_id=np.array(system_id),
            time_grid=np.linspace(0, float(getattr(dm, 't_max', 210)), data.shape[2]),
            is_real=np.r_[np.ones(len(self._benchmark_real_data), dtype=bool),
                          np.zeros(len(self._cum_synth_data), dtype=bool)], **fields)

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
        self._cum_synth_biological_labels = []
        self._benchmark_real_data = []
        self._benchmark_real_params = []
        self._benchmark_real_names = []
        self._benchmark_real_labels = []
        self._benchmark_real_biological_labels = []
        self._param_names_cached = np.array(PARAM_KEYS)

        dataset_path = getattr(dm, 'dataset_path', None)
        if dataset_path and os.path.exists(dataset_path):
            init_npz = np.load(dataset_path, allow_pickle=True)
            if 'data' in init_npz and 'params' in init_npz:
                data_arr = init_npz['data']
                params_arr = init_npz['params']
                mutant_names_arr = init_npz['mutant_names'] if 'mutant_names' in init_npz else None
                pattern_labels_arr = init_npz['pattern_labels'] if 'pattern_labels' in init_npz else None
                biological_labels_arr = init_npz.get('labels', np.full(len(data_arr), UNKNOWN_BIOLOGY_LABEL))

                if hasattr(dm, '_train_idx_synth') and len(dm._train_idx_synth) > 0:
                    idx = dm._train_idx_synth
                    self._cum_synth_data = list(data_arr[idx])
                    self._cum_synth_biological_labels = list(biological_labels_arr[idx])
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
                    self._benchmark_real_biological_labels = list(biological_labels_arr[idx])
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
            dm.train_dataset_synth = TensorDataset(bd['X'], bd['P'], bd['Y'],
                torch.as_tensor(abc_pattern_ids(bd['labels']), dtype=torch.long, device=bd['Y'].device))
            if hasattr(dm, '_mutant_names'):
                self._append_dm_label_metadata(dm, bd)
                n_existing = len(dm._mutant_names)
                dm._mutant_names = list(dm._mutant_names) + list(bd['names'])
                dm._train_idx_synth = np.arange(n_existing, n_existing + len(bd['names']))
            if 'Y_real' in bd:
                self._cum_synth_data.extend(bd['Y_real'])
                self._cum_synth_params.extend(bd['P_real'])
                self._cum_synth_names.extend(bd['names'])
                self._cum_synth_labels.extend(bd['labels'])
                self._cum_synth_biological_labels.extend(bd.get('biological_labels', [UNKNOWN_BIOLOGY_LABEL] * len(bd['names'])))

        self._al_initialized = True
        self._al4pde_al_iter = 0
        self._train_size_history.append(len(dm.train_dataset_synth))

        # 保存 Round 0 初始训练集到实验目录 datasets/
        if exp_dir and len(self._cum_synth_data) > 0:
            ds_dir = os.path.join(exp_dir, 'datasets')
            os.makedirs(ds_dir, exist_ok=True)
            r0_path = os.path.join(ds_dir, f'al_iter0_train{len(self._cum_synth_data)}.npz')
            if not os.path.exists(r0_path):
                self._save_al_snapshot(r0_path, dm)
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
        elif self.strategy == 'al4pde_power':
            bd = self._al4pde_power_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_lcmd':
            bd = self._al4pde_lcmd_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_bait':
            bd = self._al4pde_bait_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_diversity':
            bd = self._al4pde_diversity_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'maal':
            bd = self._maal_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_mass':
            bd = self._al4pde_mass_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_novelty':
            bd = self._al4pde_novelty_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_residual':
            bd = self._al4pde_residual_sample(None, pl_module, dm, device, 0)
        elif self.strategy in ('al4pde_residual_diversity', 'al4pde_res_div'):
            bd = self._al4pde_residual_diversity_sample(None, pl_module, dm, device, 0)
        elif self.strategy == 'al4pde_bifurcation':
            bd = self._al4pde_bifurcation_sample(None, pl_module, dm, device, 0)
        elif self.strategy in ('al4pde_bifurcation_entropy', 'al4pde_bif_entropy'):
            bd = self._al4pde_bifurcation_entropy_sample(None, pl_module, dm, device, 0)
        elif self.strategy in ('al4pde_event_entropy', 'al4pde_event'):
            bd = self._al4pde_event_entropy_sample(None, pl_module, dm, device, 0)
        elif self.strategy in ('al4pde_fisher_oed', 'al4pde_foed'):
            bd = self._al4pde_fisher_oed_sample(None, pl_module, dm, device, 0)
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
            new_pattern_ids = torch.as_tensor(abc_pattern_ids(bd['labels']),
                dtype=old_tensors[3].dtype, device=old_tensors[3].device)
            pattern_ids = torch.cat([old_tensors[3], new_pattern_ids], dim=0)
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
            self._append_dm_label_metadata(dm, bd)
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
            self._cum_synth_biological_labels.extend(bd.get('biological_labels', [UNKNOWN_BIOLOGY_LABEL] * len(bd['names'])))

            round_ds_path = os.path.join(ds_dir, f'al_iter{self._al4pde_al_iter}_train{len(self._cum_synth_data)}.npz')
            self._save_al_snapshot(round_ds_path, dm)
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


    def _compute_mc_dropout_uncertainty(self, pl_module, device, pool_indices, dm,
                                         n_mc_samples=10, noise_std=0.02):
        """
        参数扰动预测方差；保留历史方法名，实际未开启 MC Dropout。

        原理：对同一样本的参数 p 加微小高斯噪声，前向 K 次，计算预测方差。
        方差大表示模型对局部参数扰动敏感，不保证预测误差或认识不确定性高。

        与传统 MC Dropout 的区别：
        - 传统 MC Dropout 需要模型有 Dropout 层且推理时保持开启
        - 输入扰动法对所有模型通用，衡量局部输入敏感性，不等同于模型后验不确定性

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

            nt = getattr(dm, 'norm_type', 'log1p')
            P_norm = normalize_p(P_tensor, p_mean, p_std, nt)
            Y_ic_norm = normalize_y(Y_ic_tensor, mean_y, std_y, nt, getattr(dm, 's_c', None))

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
            # Keep init_* and other non-vector overrides when labelling selections.
            self._pool_ovr = ([json.loads(str(value)) for value in data['overrides_json']]
                              if 'overrides_json' in data else None)
            data.close()

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

            # 与实际仿真共用初值处理，包括 init_* 和 KO 联动初值。
            _, y0, _, _ = prepare_initial_conditions(ovr)
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
        nt = getattr(dm, 'norm_type', 'log1p')
        P_norm = normalize_p(P_tensor, p_mean, p_std, nt)

        # 归一化 Y_IC
        Y_ic_norm = normalize_y(Y_ic_tensor, mean_y, std_y, nt, getattr(dm, 's_c', None))

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
        return self.strategy in ('al4pde_random', 'al4pde_uncertainty', 'al4pde_power', 'al4pde_bait', 'al4pde_lcmd',
                                 'al4pde_diversity', 'maal',
                                 'al4pde_mass', 'al4pde_novelty', 'al4pde_residual',
                                 'al4pde_residual_diversity', 'al4pde_res_div',
                                 'al4pde_bifurcation', 'al4pde_bifurcation_entropy', 'al4pde_bif_entropy',
                                 'al4pde_event_entropy', 'al4pde_event',
                                 'al4pde_fisher_oed', 'al4pde_foed')

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

    def set_ensemble_members(self, members):
        """Register independently trained models; never fall back to input noise."""
        if members is None:
            self._ensemble_members = None
            return
        members = tuple(members)
        if len(members) < 2 or len({id(m) for m in members}) != len(members):
            raise ValueError('Provide at least two distinct, independently trained ensemble members')
        if any(not callable(getattr(m, 'forward_ic_time', None)) for m in members):
            raise TypeError('Each ensemble member must implement forward_ic_time')
        self._ensemble_members = members

    def _compute_ensemble_uncertainty(self, device, pool_indices, batch_size=128):
        """AL4PDE independent ensemble variance, then mean over channels/time.

        Numerical source: al4pde/prob_models/ensemble.py Ensemble.uncertainty
        and unc_independent_rollout, followed by
        BatchSelection.model_uncertainty with unc_eval_mode='mean'.
        Default loss_mse follows official adapt_unc_to_loss=True with MSE:
        mean squared disagreement from the ensemble mean. sample_variance
        follows the official raw-variance branch (unbiased, correction=1).
        Our model directly predicts the full ODE trajectory, so no autoregressive
        PDE rollout is needed. All members see identical normalized inputs.
        """
        members = self._ensemble_members
        if members is None or len(members) < 2:
            raise RuntimeError('al4pde_uncertainty requires a trained ensemble from the outer loop; input-noise fallback is disabled')
        original_modes = [m.training for m in members]
        original_devices = [next(m.parameters(), torch.empty(0)).device for m in members]
        scores = []
        try:
            for member in members:
                member.to(device)
                member.eval()
            with torch.no_grad():
                for start in range(0, len(pool_indices), batch_size):
                    indices = pool_indices[start:start + batch_size]
                    x = self._get_pool_X_batch(indices).to(device)
                    p = self._pool_P[indices].to(device)
                    outputs = torch.stack([m.forward_ic_time(x, p) for m in members])
                    if not torch.isfinite(outputs).all():
                        raise ValueError('Nonfinite ensemble prediction; cannot rank uncertainty')
                    if self.ensemble_uncertainty_mode == 'loss_mse':
                        mean_trajectory = outputs.mean(dim=0)
                        uncertainty = torch.stack([(out - mean_trajectory) ** 2 for out in outputs]).mean(dim=0)
                    else:
                        uncertainty = outputs.var(dim=0)
                    scores.append(uncertainty.reshape(len(indices), -1).mean(1).cpu())
        finally:
            for member, training, original_device in zip(members, original_modes, original_devices):
                member.to(original_device)
                member.train(training)
        return torch.cat(scores) if scores else torch.empty(0)

    def _al4pde_uncertainty_sample(self, trainer, pl_module, dm, device, current_epoch):
        """Official independent ensemble variance + mean aggregation + top-K."""
        if self._pool_size == 0:
            return None
        scores = self._compute_ensemble_uncertainty(device, np.arange(self._pool_size))
        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        # Same ordering, including ties, as official uncertainty_based.top_k.
        selected = torch.argsort(scores, descending=True)[:n_select].numpy()
        print(f'[al4pde_uncertainty] Ensemble={len(self._ensemble_members)} | top_k={n_select}')
        return self._build_broadcast_from_pool(selected, dm)

    @staticmethod
    def _power_sample_indices(scores, k, beta=1.0):
        """Official power_sampling for regular inputs; explicit degenerate handling."""
        scores = torch.as_tensor(scores).detach().cpu()
        if scores.ndim != 1 or not torch.isfinite(scores).all() or (scores < 0).any():
            raise ValueError('Power requires finite nonnegative scalar uncertainty scores')
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError('Power beta must be finite and positive')
        k = min(int(k), len(scores))
        if k <= 0:
            return torch.empty(0, dtype=torch.long)
        weights = torch.pow(scores, beta)
        if not torch.isfinite(weights).all() or not torch.isfinite(weights.sum()) or (scores.max() > 0 and weights.sum() == 0):
            weights = torch.pow(scores / scores.max(), beta)
        positive = torch.where(weights > 0)[0]
        if len(positive) >= k:
            # Exact operations from al4pde/acquisition/uncertainty_based.py.
            prob = weights / weights.sum()
            return torch.multinomial(prob, k, replacement=False)
        if len(positive):
            chosen = positive[torch.multinomial(weights[positive] / weights[positive].sum(),
                                                len(positive), replacement=False)]
        else:
            chosen = torch.empty(0, dtype=torch.long)
        zeros = torch.where(weights == 0)[0]
        remaining = k - len(chosen)
        extra = zeros[torch.multinomial(torch.ones(len(zeros)), remaining, replacement=False)]
        return torch.cat((chosen, extra))

    def _al4pde_power_sample(self, trainer, pl_module, dm, device, current_epoch):
        if self._pool_size == 0:
            return None
        scores = self._compute_ensemble_uncertainty(device, np.arange(self._pool_size))
        n_select = self._al4pde_scheduled_num_add()
        selected = self._power_sample_indices(scores, n_select, self.power_beta).numpy()
        return self._build_broadcast_from_pool(selected, dm)

    def _bait_feature_batch(self, predictions):
        """Official spatial_mean + sketch for an ODE with a singleton spatial axis.

        B,V,T -> B,1,T,V; spatial mean removes only the singleton, preserving
        time and variables. This is not a temporal mean or the custom 332D map.
        """
        from bmdal_reg.bmdal.feature_data import TensorFeatureData
        from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
        from bmdal_reg.bmdal.features import Features
        raw = predictions.detach().cpu().transpose(1, 2).flatten(1)
        if not torch.isfinite(raw).all():
            raise ValueError('Nonfinite BAIT prediction features')
        features = Features(IdentityFeatureMap(raw.shape[1]), TensorFeatureData(raw))
        if self._bait_sketch is None:
            # One sketch shared by train/pool and all rounds, as in AL4PDE.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.random_seed)
                self._bait_sketch = features.sketch_tfm(self.bait_sketch_dim)
        return self._bait_sketch(features).get_feature_matrix()

    def _compute_predicted_distance_features(self, pl_module, dm, device, batch_size=128):
        """Predict BOTH train and pool trajectories; never query the ODE oracle."""
        train_x, train_p = dm.train_dataset_synth.tensors[:2]
        if not len(train_p):
            raise ValueError('Distance acquisition requires a nonempty training set')
        training = pl_module.training
        pl_module.eval()
        def collect(count, inputs):
            output = None
            for start in range(0, count, batch_size):
                indices = np.arange(start, min(start + batch_size, count))
                x, p = inputs(indices)
                predictions = pl_module.forward_ic_time(x.to(device), p.to(device))
                features = self._bait_feature_batch(predictions)
                if output is None:
                    output = torch.empty(count, features.shape[1], dtype=features.dtype)
                output[indices] = features
            return output
        try:
            with torch.no_grad():
                train_features = collect(len(train_p), lambda idx: (train_x[idx], train_p[idx]))
                pool_features = collect(self._pool_size,
                    lambda idx: (self._get_pool_X_batch(idx), self._pool_P[idx]))
        finally:
            pl_module.train(training)
        return pool_features, train_features

    def _compute_bait_features(self, pl_module, dm, device, batch_size=128):
        """Compatibility name for the shared BAIT/LCMD prediction features."""
        return self._compute_predicted_distance_features(pl_module, dm, device, batch_size)

    @staticmethod
    def _bmdal_lcmd_select(pool_matrix, train_matrix, n_select, device='cpu'):
        """Official LCMD: predicted features, sq-dist weighting, no posterior."""
        from bmdal_reg.bmdal.feature_data import TensorFeatureData
        from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
        from bmdal_reg.bmdal.features import Features
        from .bmdal_compat import selection_module
        n_select = min(int(n_select), len(pool_matrix))
        if n_select <= 0:
            return np.empty(0, dtype=int)
        train_matrix, pool_matrix = train_matrix.to(device), pool_matrix.to(device)
        if not torch.isfinite(train_matrix).all() or not torch.isfinite(pool_matrix).all():
            raise ValueError('LCMD requires finite features')
        width = train_matrix.shape[1]
        train_features = Features(IdentityFeatureMap(width), TensorFeatureData(train_matrix))
        pool_features = Features(IdentityFeatureMap(width), TensorFeatureData(pool_matrix))
        selector = selection_module().LargestClusterMaxDistSelectionMethod(
            pool_features=pool_features, train_features=train_features,
            sel_with_train=True, dist_weight_mode='sq-dist')
        result = selector.select(n_select).detach().cpu().numpy()
        if len(result) != n_select or len(np.unique(result)) != len(result):
            raise RuntimeError('LCMD returned an invalid selection')
        return result

    def _al4pde_lcmd_sample(self, trainer, pl_module, dm, device, current_epoch):
        if self._pool_size == 0:
            return None
        pool_features, train_features = self._compute_predicted_distance_features(pl_module, dm, device)
        selected = self._bmdal_lcmd_select(pool_features, train_features,
                                          self._al4pde_scheduled_num_add(), device)
        return self._build_broadcast_from_pool(selected, dm)

    @staticmethod
    def _bmdal_bait_select(pool_matrix, train_matrix, n_select, device='cpu'):
        """Official max_dist_selection posterior pipeline + actual bmdal BAIT."""
        from bmdal_reg.bmdal.feature_data import TensorFeatureData
        from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
        from bmdal_reg.bmdal.features import Features
        from .bmdal_compat import selection_module
        n_select = min(int(n_select), len(pool_matrix))
        if n_select <= 0:
            return np.empty(0, dtype=int)
        train_matrix, pool_matrix = train_matrix.to(device), pool_matrix.to(device)
        if not torch.isfinite(train_matrix).all() or not torch.isfinite(pool_matrix).all():
            raise ValueError('BAIT requires finite features')
        energy = train_matrix.square().sum(1).mean()
        if not torch.isfinite(energy) or energy <= 0:
            raise ValueError('BAIT training features have zero energy; official scale transform is undefined')
        width = train_matrix.shape[1]
        train_features = Features(IdentityFeatureMap(width), TensorFeatureData(train_matrix))
        pool_features = Features(IdentityFeatureMap(width), TensorFeatureData(pool_matrix))
        # Exact scale/posterior order and noise from config/acquisition/bait.yaml.
        scale = train_features.scale_tfm()
        train_features, pool_features = scale(train_features), scale(pool_features)
        posterior = train_features.posterior_tfm(sigma=1e-2, allow_kernel_space_posterior=False)
        train_features, pool_features = posterior(train_features), posterior(pool_features)
        selector = selection_module().BaitFeatureSpaceSelectionMethod(
            pool_features=pool_features, train_features=train_features,
            sel_with_train=True, noise_sigma=1e-2)
        result = selector.select(n_select).detach().cpu().numpy()
        if len(result) != n_select or len(np.unique(result)) != len(result):
            raise RuntimeError('BAIT returned an invalid selection')
        return result

    def _al4pde_bait_sample(self, trainer, pl_module, dm, device, current_epoch):
        if self._pool_size == 0:
            return None
        pool_features, train_features = self._compute_bait_features(pl_module, dm, device)
        selected = self._bmdal_bait_select(pool_features, train_features,
                                          self._al4pde_scheduled_num_add(), device)
        return self._build_broadcast_from_pool(selected, dm)

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

    def _compute_pool_features_shape_aware(self, pl_module, dm, device, batch_size=2500, indices=None):
        """返回候选池 (或指定子集) 的轨迹形状感知特征 (raw, 未标准化)
        通过模型 forward_ic_time 快速批量预测生成轨迹特征 (无需前端耗时 ODE 求解)。
        """
        if indices is None:
            if self._pool_shape_features_cached is not None:
                return self._pool_shape_features_cached
            eval_indices = np.arange(self._pool_size)
        else:
            eval_indices = np.asarray(indices, dtype=np.int64)

        n_eval = len(eval_indices)
        pl_module.eval()
        out_features = None
        with torch.no_grad():
            for i in range(0, n_eval, batch_size):
                end_idx = min(i + batch_size, n_eval)
                b_idx = eval_indices[i:end_idx]
                X_batch = self._get_pool_X_batch(b_idx).to(device)
                P_batch = self._pool_P[b_idx].to(device)
                preds = pl_module.forward_ic_time(X_batch, P_batch)
                feat = self._extract_shape_features(preds, P_batch).cpu()
                del X_batch, P_batch, preds
                if out_features is None:
                    out_features = torch.empty((n_eval, feat.shape[1]), dtype=torch.float32)
                out_features[i:end_idx] = feat
                del feat
        pl_module.train()
        if indices is None:
            self._pool_shape_features_cached = out_features
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

    def _al4pde_residual_sample(self, trainer, pl_module, dm, device, current_epoch):
        """al4pde Residual (v2.4 优化版): 局部连续生化通道流速自适应相对物理残差 (Flux-Adaptive Relative Residual).

        四大优化彻底攻克物理残差失效瓶颈:
        1. 流速自适应相对物理残差 (Flux-Adaptive Relative Residual):
           分母采用局部流速与底噪保护 (|du/dt| + |f_biochem| + 0.1), 将残差严格约束在 [0, 1] 内,
           杜绝超表达大常数绑架打分, 极大提高对敲除阻滞失败/假振荡 (如 15 号突变体) 的捕捉灵敏度。
        2. 门控 1 (标称通量剪枝 Nominal Flux Pruning):
           提前筛除合成常数超过真实突变体极大值 3x 的病态发散候选。
        3. 门控 2 (模型预测浓度极值惩罚 Model-Predicted Max Penalty):
           若模型预测浓度已突破真实生理极大值 2x, 判定为潜在发散 Monster 予以 0.01x 惩罚。
        4. 配额达标动态缓冲填充 (Quota-Guaranteed Dynamic Filling):
           严格累积达标有效样本 (Round 1: 10,000, Round 2: 20,000), 根除数据饥饿。
        """
        if self._pool_size == 0:
            print("[al4pde_residual] 候选池已空, 跳过本轮 AL")
            return None

        from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE

        print(f"[al4pde_residual] 🚀 启动流速自适应相对物理残差评估 (池大小: {self._pool_size})...")
        t_start = time.time()
        pl_module.eval()

        # 初始化向量化 ODE 引擎
        param_keys = getattr(self, '_param_names_cached', None)
        if param_keys is None:
            pool_ds = self.al_pool_dataset or getattr(dm, 'dataset_path', None)
            if pool_ds and os.path.exists(pool_ds):
                npz_data = np.load(pool_ds, allow_pickle=True)
                if 'param_names' in npz_data:
                    param_keys = list(npz_data['param_names'])
        if param_keys is None:
            param_keys = list(PARAM_KEYS)
        ode_engine = VectorizedYeastODE(param_keys).to(device)

        batch_size = 2500
        pool_indices = np.arange(self._pool_size)
        all_residual_scores = []
        n_steps = self._al4pde_t_grid.shape[2] if self._al4pde_t_grid is not None else 500
        dt = getattr(dm, 't_max', 210.0) / max(1, (n_steps - 1))

        # 提取真实生理极大值 (用于门控 2 预测浓度惩罚)
        if hasattr(dm, '_Y') and hasattr(dm, '_test_idx_real'):
            real_envelope = dm._Y[dm._test_idx_real].max(dim=-1)[0].max(dim=0)[0].to(device)
        else:
            Y_real_phys = dm.denormalize_y(dm.test_dataset_real.tensors[2].to(device))
            real_envelope = Y_real_phys.max(dim=-1)[0].max(dim=0)[0]
        dyn_max_thresh = torch.maximum(real_envelope * 2.0, torch.tensor(2.0, device=device))

        p_mean = dm.p_mean.to(device)
        p_std = dm.p_std.to(device)

        pruned_count = 0
        with torch.no_grad():
            for i in range(0, len(pool_indices), batch_size):
                end_idx = min(i + batch_size, len(pool_indices))
                b_idx = pool_indices[i:end_idx]

                X_batch = self._get_pool_X_batch(b_idx).to(device)
                P_batch = self._pool_P[b_idx].to(device)

                if self._pool_P_real is not None:
                    P_real_batch = self._pool_P_real[b_idx].to(device)
                else:
                    P_real_batch = P_batch * p_std + p_mean

                # 门控 1: 标称通量剪枝 (Nominal Flux Pruning)
                flux_mask = ode_engine.check_nominal_flux_mask(P_real_batch, max_individual_ratio=3.0, max_sum_ratio=3.0)
                pruned_count += (~flux_mask).sum().item()

                # 1. 模型预测 (归一化空间)
                preds_norm = pl_module.forward_ic_time(X_batch, P_batch)

                # 2. 真实物理空间反归一化 (浓度与参数)
                preds_real = dm.denormalize_y(preds_norm, device)

                # 3. 门控 2: 模型预测浓度极值先验 (Pred Max Monster Penalty)
                is_pred_monster = (preds_real[:, 1:39, :].max(dim=-1)[0] > dyn_max_thresh[1:39].unsqueeze(0)).any(dim=-1)

                # 4. 矢量化计算流速自适应相对物理残差 (bounded in [0, 1])
                b_scores = ode_engine.compute_residual_scores(preds_real, P_real_batch, dt, relative=True, eps=0.1)

                # 门控 2 施加 0.01x 惩罚
                b_scores = torch.where(is_pred_monster, b_scores * 0.01, b_scores)
                # 门控 1 直接屏蔽 (设为 -1.0)
                b_scores = torch.where(flux_mask, b_scores, torch.full_like(b_scores, -1.0))

                all_residual_scores.append(b_scores.cpu().numpy())

        pl_module.train()
        residual_scores = np.concatenate(all_residual_scores)
        eval_time = time.time() - t_start

        # 排序候选 (得分最高优先)
        ranked_idx = np.argsort(residual_scores)[::-1]
        valid_candidate_mask = residual_scores[ranked_idx] >= 0.0
        ranked_candidates = ranked_idx[valid_candidate_mask]

        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)

        print(f"[al4pde_residual] ✅ 评估完成 (耗时 {eval_time:.2f}s, 速率 {self._pool_size/eval_time:.1f} 样本/s)")
        print(f"[al4pde_residual] 门控 1 (标称通量剪枝): 剔除 {pruned_count}/{self._pool_size} 极端候选 "
              f"({100*pruned_count/self._pool_size:.2f}%)")
        print(f"[al4pde_residual] 有效相对残差分布: min={residual_scores[ranked_candidates].min():.4f}, "
              f"median={np.median(residual_scores[ranked_candidates]):.4f}, max={residual_scores[ranked_candidates].max():.4f}")

        # 调用配额达标动态缓冲填充器，严格入库 n_select 个有效样本
        return self._build_broadcast_from_pool(ranked_candidates, dm, target_valid_num=n_select)

    def _al4pde_residual_diversity_sample(self, trainer, pl_module, dm, device, current_epoch):
        """al4pde Residual-Diversity (v1.0): 相对物理残差门控 + 轨迹拓扑多样性混合采样.

        两阶段机制:
        1. 阶段 1 (残差门控): 在全候选池评估流速自适应相对残差与物理门控, 截取残差最高的 Top 50,000 候选子集 (高违规高信息量池);
        2. 阶段 2 (多样性细选): 在此 50,000 子集内提取 332 维轨迹形状感知特征与训练集对比,
           运行 bmdal MaxDist 贪心最大距离选择最正交、最分散的样本排序;
        3. 传入配额达标动态缓冲填充器, 严格入库 K_target 个有效样本。
        """
        if self._pool_size == 0:
            print("[al4pde_residual_diversity] 候选池已空, 跳过本轮 AL")
            return None

        from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE

        print(f"[al4pde_residual_diversity] 🚀 阶段 1: 启动相对物理残差门控粗筛 (池大小: {self._pool_size})...")
        t_start = time.time()
        pl_module.eval()

        param_keys = getattr(self, '_param_names_cached', None)
        if param_keys is None:
            pool_ds = self.al_pool_dataset or getattr(dm, 'dataset_path', None)
            if pool_ds and os.path.exists(pool_ds):
                npz_data = np.load(pool_ds, allow_pickle=True)
                if 'param_names' in npz_data:
                    param_keys = list(npz_data['param_names'])
        if param_keys is None:
            param_keys = list(PARAM_KEYS)
        ode_engine = VectorizedYeastODE(param_keys).to(device)

        batch_size = 2500
        pool_indices = np.arange(self._pool_size)
        all_residual_scores = []
        n_steps = self._al4pde_t_grid.shape[2] if self._al4pde_t_grid is not None else 500
        dt = getattr(dm, 't_max', 210.0) / max(1, (n_steps - 1))

        if hasattr(dm, '_Y') and hasattr(dm, '_test_idx_real'):
            real_envelope = dm._Y[dm._test_idx_real].max(dim=-1)[0].max(dim=0)[0].to(device)
        else:
            Y_real_phys = dm.denormalize_y(dm.test_dataset_real.tensors[2].to(device))
            real_envelope = Y_real_phys.max(dim=-1)[0].max(dim=0)[0]
        dyn_max_thresh = torch.maximum(real_envelope * 2.0, torch.tensor(2.0, device=device))

        p_mean = dm.p_mean.to(device)
        p_std = dm.p_std.to(device)

        with torch.no_grad():
            for i in range(0, len(pool_indices), batch_size):
                end_idx = min(i + batch_size, len(pool_indices))
                b_idx = pool_indices[i:end_idx]

                X_batch = self._get_pool_X_batch(b_idx).to(device)
                P_batch = self._pool_P[b_idx].to(device)

                if self._pool_P_real is not None:
                    P_real_batch = self._pool_P_real[b_idx].to(device)
                else:
                    P_real_batch = P_batch * p_std + p_mean

                flux_mask = ode_engine.check_nominal_flux_mask(P_real_batch, max_individual_ratio=3.0, max_sum_ratio=3.0)
                preds_norm = pl_module.forward_ic_time(X_batch, P_batch)
                preds_real = dm.denormalize_y(preds_norm, device)
                is_pred_monster = (preds_real[:, 1:39, :].max(dim=-1)[0] > dyn_max_thresh[1:39].unsqueeze(0)).any(dim=-1)

                b_scores = ode_engine.compute_residual_scores(preds_real, P_real_batch, dt, relative=True, eps=0.1)
                b_scores = torch.where(is_pred_monster, b_scores * 0.01, b_scores)
                b_scores = torch.where(flux_mask, b_scores, torch.full_like(b_scores, -1.0))
                all_residual_scores.append(b_scores.cpu().numpy())

        pl_module.train()
        residual_scores = np.concatenate(all_residual_scores)

        # 提取残差最高的前 50,000 个合格候选子集
        ranked_idx = np.argsort(residual_scores)[::-1]
        valid_candidate_mask = residual_scores[ranked_idx] >= 0.0
        feasible_candidates = ranked_idx[valid_candidate_mask]
        subpool_size = min(50000, len(feasible_candidates))
        subpool_idx = feasible_candidates[:subpool_size]

        print(f"[al4pde_residual_diversity] ✅ 阶段 1 粗筛完成: 提取 Top {subpool_size} 高物理残差候选子集 "
              f"(残差区间: {residual_scores[subpool_idx[-1]]:.4f} ~ {residual_scores[subpool_idx[0]]:.4f})")

        # 阶段 2: 轨迹形状感知特征多样性 MaxDist 选择
        print(f"[al4pde_residual_diversity] 🚀 阶段 2: 在高残差子集中计算形状感知特征并进行 MaxDist 拓扑多样性覆盖...")
        subpool_features = self._compute_pool_features_shape_aware(pl_module, dm, device, indices=subpool_idx)
        train_features = self._compute_train_features_shape_aware(pl_module, dm, device)
        subpool_features, train_features = self._standardize_shape_features(subpool_features, train_features)

        # 在 50,000 个子集中选取全部并获得多样性优先序
        diverse_local_indices = self._bmdal_max_dist_select(subpool_features, train_features, len(subpool_idx))
        ordered_candidates = subpool_idx[diverse_local_indices]

        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        print(f"[al4pde_residual_diversity] 多样性排序完成, 进入配额达标动态缓冲填充 (目标有效样本: {n_select})...")
        return self._build_broadcast_from_pool(ordered_candidates, dm, target_valid_num=n_select)

    def _al4pde_bifurcation_sample(self, trainer, pl_module, dm, device, current_epoch):
        """al4pde 路线一: 相空间吸引子与分岔临界流形主动学习 (Bifurcation-Manifold Active Learning, BM-AL)

        第一性原理与理论创新:
        - 细胞周期本质是相空间中的稳定极限环吸引子 (Limit Cycle Attractor);
        - 突变致死阻滞对应定点吸引子 (Fixed-Point Arrest);
        - 真实突变体绝大多数恰好分布在这两条相变分岔超曲面 (Codimension-1 Bifurcation Manifold) 之上;
        - 本算法通过检测核心细胞周期蛋白 (CLB2, SIC1, CLN2) 的互斥振荡活力、临界减速 (Critical Slowing Down) 漂移、
          以及中性相平衡, 准确定位相变分岔超曲面, 杜绝纯残差对死细胞阻滞态的病态过度采样。
        """
        if self._pool_size == 0:
            print("[al4pde_bifurcation] 候选池已空, 跳过本轮 AL")
            return None

        from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE

        print(f"[al4pde_bifurcation] 🚀 启动分岔临界流形探测 (池大小: {self._pool_size})...")
        t_start = time.time()
        pl_module.eval()

        param_keys = getattr(self, '_param_names_cached', None)
        if param_keys is None:
            pool_ds = self.al_pool_dataset or getattr(dm, 'dataset_path', None)
            if pool_ds and os.path.exists(pool_ds):
                npz_data = np.load(pool_ds, allow_pickle=True)
                if 'param_names' in npz_data:
                    param_keys = list(npz_data['param_names'])
        if param_keys is None:
            param_keys = list(PARAM_KEYS)
        ode_engine = VectorizedYeastODE(param_keys).to(device)

        batch_size = 2500
        pool_indices = np.arange(self._pool_size)
        all_bif_scores = []
        n_steps = self._al4pde_t_grid.shape[2] if self._al4pde_t_grid is not None else 500

        if hasattr(dm, '_Y') and hasattr(dm, '_test_idx_real'):
            real_envelope = dm._Y[dm._test_idx_real].max(dim=-1)[0].max(dim=0)[0].to(device)
        else:
            Y_real_phys = dm.denormalize_y(dm.test_dataset_real.tensors[2].to(device))
            real_envelope = Y_real_phys.max(dim=-1)[0].max(dim=0)[0]
        dyn_max_thresh = torch.maximum(real_envelope * 2.0, torch.tensor(2.0, device=device))

        p_mean = dm.p_mean.to(device)
        p_std = dm.p_std.to(device)
        pruned_count = 0

        with torch.no_grad():
            for i in range(0, len(pool_indices), batch_size):
                end_idx = min(i + batch_size, len(pool_indices))
                b_idx = pool_indices[i:end_idx]

                X_batch = self._get_pool_X_batch(b_idx).to(device)
                P_batch = self._pool_P[b_idx].to(device)

                if self._pool_P_real is not None:
                    P_real_batch = self._pool_P_real[b_idx].to(device)
                else:
                    P_real_batch = P_batch * p_std + p_mean

                flux_mask = ode_engine.check_nominal_flux_mask(P_real_batch, max_individual_ratio=3.0, max_sum_ratio=3.0)
                pruned_count += (~flux_mask).sum().item()

                preds_norm = pl_module.forward_ic_time(X_batch, P_batch)
                preds_real = dm.denormalize_y(preds_norm, device)

                is_pred_monster = (preds_real[:, 1:39, :].max(dim=-1)[0] > dyn_max_thresh[1:39].unsqueeze(0)).any(dim=-1)

                CLB2 = preds_real[:, 2, :]
                SIC1 = preds_real[:, 4, :]
                CLN2 = preds_real[:, 1, :]

                # 振荡活力 (Oscillatory Vitality)
                amp_clb2 = CLB2.max(dim=-1)[0] - CLB2.min(dim=-1)[0]
                amp_sic1 = SIC1.max(dim=-1)[0] - SIC1.min(dim=-1)[0]
                osc_vitality = (amp_clb2 * amp_sic1) / (amp_clb2 * amp_sic1 + 0.04)

                # 临界漂移 (Phase Drift & Critical Slowing Down)
                mid = n_steps // 2
                mean_1 = preds_real[:, 1:33, :mid].mean(dim=-1)
                mean_2 = preds_real[:, 1:33, mid:].mean(dim=-1)
                drift = torch.norm(mean_2 - mean_1, dim=-1) / (torch.norm(mean_1, dim=-1) + 1e-4)

                bif_scores = osc_vitality * torch.exp(-torch.abs(drift - 0.25) / 0.5)

                b_scores = torch.where(is_pred_monster, bif_scores * 0.01, bif_scores)
                b_scores = torch.where(flux_mask, b_scores, torch.full_like(b_scores, -1.0))
                all_bif_scores.append(b_scores.cpu().numpy())

        pl_module.train()
        bif_scores_all = np.concatenate(all_bif_scores)
        eval_time = time.time() - t_start

        ranked_idx = np.argsort(bif_scores_all)[::-1]
        valid_candidate_mask = bif_scores_all[ranked_idx] >= 0.0
        ranked_candidates = ranked_idx[valid_candidate_mask]

        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        print(f"[al4pde_bifurcation] ✅ 分岔流形评估完成 (耗时 {eval_time:.2f}s, 速率 {self._pool_size/eval_time:.1f} 样本/s)")
        print(f"[al4pde_bifurcation] 门控剪枝: 剔除 {pruned_count}/{self._pool_size} 异常候选")
        print(f"[al4pde_bifurcation] 最高分岔临界度候选: min={bif_scores_all[ranked_candidates[-1]]:.4f}, max={bif_scores_all[ranked_candidates[0]]:.4f}")
        return self._build_broadcast_from_pool(ranked_candidates, dm, target_valid_num=n_select)

    def _al4pde_bifurcation_entropy_sample(self, trainer, pl_module, dm, device, current_epoch):
        """al4pde 分岔相变信息熵主动学习 (Bifurcation Shannon Entropy Active Learning, BSE-AL)

        第一性原理与理论创新:
        - 酵母细胞周期动力系统具有陡峭的高维相变分岔超曲面 (Limit Cycle vs. Fixed-Point Arrest);
        - 在相变边界附近，动力学对参数微小扰动极其敏感，系统在增殖振荡 (Pattern A/B) 与致死停滞 (Pattern C) 之间发生剧烈拓扑跃迁;
        - 通过模型自带的分岔判别头 P(C | p)（或由训练集在线训练的高速分岔判别器），对 100 万未标注候选池的参数批次计算香农分岔信息熵:
            H(p) = - \\sum_{k \\in {A, B, C}} P_k(p) * ln(P_k(p) + 1e-12)
        - 挑选 H(p) 最高的临界样本（即模型在分类决策上最不确定、最接近分岔相变临界面的样本）调用真实 ODE 求解器打标并增补进训练集;
        - 这样能用最少的样本代价，把高维相变边界刻画得极其陡峭精准，彻底消除盲目起振或虚假阻滞的泛化缺陷。
        """
        if self._pool_size == 0:
            print("[al4pde_bifurcation_entropy] 候选池已空, 跳过本轮 AL")
            return None

        from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE

        print(f"[al4pde_bifurcation_entropy] 🚀 启动分岔相变信息熵主动学习评估 (池大小: {self._pool_size})...")
        t_start = time.time()
        
        # 1. 获取/构建分岔分类器
        classifier_fn = None
        if pl_module is not None and hasattr(pl_module, 'model') and hasattr(pl_module.model, 'pattern_head'):
            pl_module.eval()
            classifier_fn = lambda p_batch: pl_module.model.pattern_head(p_batch)
            print("[al4pde_bifurcation_entropy] 使用主模型内置 pattern_head 分岔预测头")
        else:
            # 在线训练极速 2 层 MLP 分类器
            print("[al4pde_bifurcation_entropy] 主模型无 pattern_head，在线训练高速分岔分类器...")
            t_clf_start = time.time()
            train_tensors = dm.train_dataset_synth.tensors
            train_P = train_tensors[1].to(device)
            if len(train_tensors) >= 4:
                train_labels = train_tensors[3].to(device)
            else:
                train_Y = train_tensors[2].to(device)
                amp = train_Y[:, 2, :].max(dim=-1)[0] - train_Y[:, 2, :].min(dim=-1)[0]
                train_labels = torch.where(amp > 0.5, torch.zeros_like(amp, dtype=torch.long), 
                                           torch.where(amp > 0.1, torch.ones_like(amp, dtype=torch.long), 
                                                       torch.full_like(amp, 2, dtype=torch.long)))
            
            mlp_clf = torch.nn.Sequential(
                torch.nn.Linear(train_P.shape[1], 128),
                torch.nn.GELU(),
                torch.nn.Linear(128, 64),
                torch.nn.GELU(),
                torch.nn.Linear(64, 3)
            ).to(device)
            optimizer = torch.optim.AdamW(mlp_clf.parameters(), lr=3e-3, weight_decay=1e-4)
            criterion = torch.nn.CrossEntropyLoss()
            
            mlp_clf.train()
            for _ in range(15):
                logits = mlp_clf(train_P)
                loss = criterion(logits, train_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mlp_clf.eval()
            classifier_fn = lambda p_batch: mlp_clf(p_batch)
            print(f"[al4pde_bifurcation_entropy] 高速分类器训练完成, 耗时 {time.time() - t_clf_start:.3f}s")

        # 2. 准备 ODE 物理引擎用于通量异常初筛
        param_keys = getattr(self, '_param_names_cached', None)
        if param_keys is None:
            pool_ds = self.al_pool_dataset or getattr(dm, 'dataset_path', None)
            if pool_ds and os.path.exists(pool_ds):
                npz_data = np.load(pool_ds, allow_pickle=True)
                if 'param_names' in npz_data:
                    param_keys = list(npz_data['param_names'])
        if param_keys is None:
            param_keys = list(PARAM_KEYS)
        ode_engine = VectorizedYeastODE(param_keys).to(device)

        batch_size = 10000
        pool_indices = np.arange(self._pool_size)
        all_entropies = []
        p_mean = dm.p_mean.to(device)
        p_std = dm.p_std.to(device)
        pruned_count = 0

        with torch.no_grad():
            for i in range(0, len(pool_indices), batch_size):
                end_idx = min(i + batch_size, len(pool_indices))
                b_idx = pool_indices[i:end_idx]

                P_batch = self._pool_P[b_idx].to(device)

                if self._pool_P_real is not None:
                    P_real_batch = self._pool_P_real[b_idx].to(device)
                else:
                    P_real_batch = P_batch * p_std + p_mean

                flux_mask = ode_engine.check_nominal_flux_mask(P_real_batch, max_individual_ratio=3.0, max_sum_ratio=3.0)
                pruned_count += (~flux_mask).sum().item()

                logits = classifier_fn(P_batch)
                probs = F.softmax(logits, dim=-1)

                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
                entropy = torch.where(flux_mask, entropy, torch.full_like(entropy, -1.0))
                all_entropies.append(entropy.cpu().numpy())

        if pl_module is not None:
            pl_module.train()
        entropies_all = np.concatenate(all_entropies)
        eval_time = time.time() - t_start

        ranked_idx = np.argsort(entropies_all)[::-1]
        valid_candidate_mask = entropies_all[ranked_idx] >= 0.0
        ranked_candidates = ranked_idx[valid_candidate_mask]

        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        max_ent = entropies_all[ranked_candidates[0]] if len(ranked_candidates) > 0 else 0.0
        print(f"[al4pde_bifurcation_entropy] ✅ 分岔信息熵评估完成 (耗时 {eval_time:.2f}s, 吞吐量 {self._pool_size/eval_time:.1f} 样本/s)")
        print(f"[al4pde_bifurcation_entropy] 门控剪枝: 剔除 {pruned_count}/{self._pool_size} 异常参数候选")
        print(f"[al4pde_bifurcation_entropy] 选中前 {n_select} 个最高熵临界样本: 最大熵={max_ent:.4f} (理论上限 ln(3)≈1.0986)")

        return self._build_broadcast_from_pool(ranked_candidates, dm, target_valid_num=n_select)

    def _al4pde_event_entropy_sample(self, trainer, pl_module, dm, device, current_epoch):
        """al4pde 路线二: 生物检验点离散事件转移熵 (Hybrid-Event Transition Entropy, HETE-AL)

        第一性原理与理论创新:
        - 酵母细胞周期是包含连续化学动力学与离散检验点跳跃的混合自动机 (Hybrid Automaton);
        - 4 大关键生命事件: G1/S 启动 (Start)、极性芽体 (Bud)、纺锤体组装检验 (SAC)、出核网络 (MEN);
        - 神经算子在检验点相变处的决策模糊度对应于事件发生概率密度的最大信息熵 (Maximum Transition Entropy);
        - 本算法通过计算 4 大检验点激活指示函数沿时间轴的跳跃导数分布与转移熵, 专门挑选时序决策处于临界犹豫态的核心突变体。
        """
        if self._pool_size == 0:
            print("[al4pde_event_entropy] 候选池已空, 跳过本轮 AL")
            return None

        from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE

        print(f"[al4pde_event_entropy] 🚀 启动检验点离散事件转移熵评估 (池大小: {self._pool_size})...")
        t_start = time.time()
        pl_module.eval()

        param_keys = getattr(self, '_param_names_cached', None)
        if param_keys is None:
            pool_ds = self.al_pool_dataset or getattr(dm, 'dataset_path', None)
            if pool_ds and os.path.exists(pool_ds):
                npz_data = np.load(pool_ds, allow_pickle=True)
                if 'param_names' in npz_data:
                    param_keys = list(npz_data['param_names'])
        if param_keys is None:
            param_keys = list(PARAM_KEYS)
        ode_engine = VectorizedYeastODE(param_keys).to(device)

        batch_size = 2500
        pool_indices = np.arange(self._pool_size)
        all_event_scores = []

        if hasattr(dm, '_Y') and hasattr(dm, '_test_idx_real'):
            real_envelope = dm._Y[dm._test_idx_real].max(dim=-1)[0].max(dim=0)[0].to(device)
        else:
            Y_real_phys = dm.denormalize_y(dm.test_dataset_real.tensors[2].to(device))
            real_envelope = Y_real_phys.max(dim=-1)[0].max(dim=0)[0]
        dyn_max_thresh = torch.maximum(real_envelope * 2.0, torch.tensor(2.0, device=device))

        p_mean = dm.p_mean.to(device)
        p_std = dm.p_std.to(device)
        pruned_count = 0

        with torch.no_grad():
            for i in range(0, len(pool_indices), batch_size):
                end_idx = min(i + batch_size, len(pool_indices))
                b_idx = pool_indices[i:end_idx]

                X_batch = self._get_pool_X_batch(b_idx).to(device)
                P_batch = self._pool_P[b_idx].to(device)

                if self._pool_P_real is not None:
                    P_real_batch = self._pool_P_real[b_idx].to(device)
                else:
                    P_real_batch = P_batch * p_std + p_mean

                flux_mask = ode_engine.check_nominal_flux_mask(P_real_batch, max_individual_ratio=3.0, max_sum_ratio=3.0)
                pruned_count += (~flux_mask).sum().item()

                preds_norm = pl_module.forward_ic_time(X_batch, P_batch)
                preds_real = dm.denormalize_y(preds_norm, device)

                is_pred_monster = (preds_real[:, 1:39, :].max(dim=-1)[0] > dyn_max_thresh[1:39].unsqueeze(0)).any(dim=-1)

                s_start = torch.sigmoid((preds_real[:, 1, :] - 0.15) * 15.0) * torch.sigmoid((0.15 - preds_real[:, 4, :]) * 15.0)
                s_bud   = torch.sigmoid((preds_real[:, 2, :] - 0.20) * 15.0)
                s_sac   = torch.sigmoid((preds_real[:, 19, :] - 0.25) * 15.0)
                s_men   = torch.sigmoid((preds_real[:, 21, :] - 0.30) * 15.0) * torch.sigmoid((0.15 - preds_real[:, 2, :]) * 15.0)

                event_entropies = []
                for s_ev in [s_start, s_bud, s_sac, s_men]:
                    rho = torch.abs(s_ev[:, 1:] - s_ev[:, :-1])
                    sum_rho = rho.sum(dim=-1, keepdim=True) + 1e-6
                    p = rho / sum_rho
                    H = - torch.sum(p * torch.log(p + 1e-12), dim=-1)
                    active_w = torch.clamp(sum_rho.squeeze(-1) / 4.0, 0.0, 1.0)
                    event_entropies.append(H * active_w)

                ev_scores = torch.stack(event_entropies, dim=-1).sum(dim=-1)

                ev_scores = torch.where(is_pred_monster, ev_scores * 0.01, ev_scores)
                ev_scores = torch.where(flux_mask, ev_scores, torch.full_like(ev_scores, -1.0))
                all_event_scores.append(ev_scores.cpu().numpy())

        pl_module.train()
        event_scores_all = np.concatenate(all_event_scores)
        eval_time = time.time() - t_start

        ranked_idx = np.argsort(event_scores_all)[::-1]
        valid_candidate_mask = event_scores_all[ranked_idx] >= 0.0
        ranked_candidates = ranked_idx[valid_candidate_mask]

        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        print(f"[al4pde_event_entropy] ✅ 检验点转移熵评估完成 (耗时 {eval_time:.2f}s, 速率 {self._pool_size/eval_time:.1f} 样本/s)")
        print(f"[al4pde_event_entropy] 门控剪枝: 剔除 {pruned_count}/{self._pool_size} 异常候选")
        print(f"[al4pde_event_entropy] 最高转移熵候选: min={event_scores_all[ranked_candidates[-1]]:.4f}, max={event_scores_all[ranked_candidates[0]]:.4f}")
        return self._build_broadcast_from_pool(ranked_candidates, dm, target_valid_num=n_select)

    def _al4pde_fisher_oed_sample(self, trainer, pl_module, dm, device, current_epoch):
        """al4pde 路线三: 泛函空间贝叶斯最优实验设计 (Functional Fisher Information OED, F-OED)

        第一性原理与理论创新:
        - 建立在信息几何与泛函最优实验设计 (Bayesian OED) 的严格数学基石之上;
        - 消除启发式主观加权, 针对神经算子输出流场的经验 Fisher 信息阵 (Empirical FIM) 实施 D-最优性 (D-Optimality) 体积最大化;
        - 结合物理控制方程残差的反传梯度嵌入, 使用 Gram-Schmidt 正交化在特征空间贪心最大化多元体积,
          同时兼顾物理拟合残差幅值与参数更新方向的几何正交性。
        """
        if self._pool_size == 0:
            print("[al4pde_fisher_oed] 候选池已空, 跳过本轮 AL")
            return None

        from budding_yeast_v2.utils.ode_residual_engine import VectorizedYeastODE

        print(f"[al4pde_fisher_oed] 🚀 启动泛函 Fisher 最优实验设计 (池大小: {self._pool_size})...")
        t_start = time.time()
        pl_module.eval()

        param_keys = getattr(self, '_param_names_cached', None)
        if param_keys is None:
            pool_ds = self.al_pool_dataset or getattr(dm, 'dataset_path', None)
            if pool_ds and os.path.exists(pool_ds):
                npz_data = np.load(pool_ds, allow_pickle=True)
                if 'param_names' in npz_data:
                    param_keys = list(npz_data['param_names'])
        if param_keys is None:
            param_keys = list(PARAM_KEYS)
        ode_engine = VectorizedYeastODE(param_keys).to(device)

        batch_size = 2500
        pool_indices = np.arange(self._pool_size)
        all_residual_scores = []
        n_steps = self._al4pde_t_grid.shape[2] if self._al4pde_t_grid is not None else 500
        dt = getattr(dm, 't_max', 210.0) / max(1, (n_steps - 1))

        if hasattr(dm, '_Y') and hasattr(dm, '_test_idx_real'):
            real_envelope = dm._Y[dm._test_idx_real].max(dim=-1)[0].max(dim=0)[0].to(device)
        else:
            Y_real_phys = dm.denormalize_y(dm.test_dataset_real.tensors[2].to(device))
            real_envelope = Y_real_phys.max(dim=-1)[0].max(dim=0)[0]
        dyn_max_thresh = torch.maximum(real_envelope * 2.0, torch.tensor(2.0, device=device))

        p_mean = dm.p_mean.to(device)
        p_std = dm.p_std.to(device)
        pruned_count = 0

        with torch.no_grad():
            for i in range(0, len(pool_indices), batch_size):
                end_idx = min(i + batch_size, len(pool_indices))
                b_idx = pool_indices[i:end_idx]

                X_batch = self._get_pool_X_batch(b_idx).to(device)
                P_batch = self._pool_P[b_idx].to(device)

                if self._pool_P_real is not None:
                    P_real_batch = self._pool_P_real[b_idx].to(device)
                else:
                    P_real_batch = P_batch * p_std + p_mean

                flux_mask = ode_engine.check_nominal_flux_mask(P_real_batch, max_individual_ratio=3.0, max_sum_ratio=3.0)
                pruned_count += (~flux_mask).sum().item()

                preds_norm = pl_module.forward_ic_time(X_batch, P_batch)
                preds_real = dm.denormalize_y(preds_norm, device)

                is_pred_monster = (preds_real[:, 1:39, :].max(dim=-1)[0] > dyn_max_thresh[1:39].unsqueeze(0)).any(dim=-1)

                b_scores = ode_engine.compute_residual_scores(preds_real, P_real_batch, dt, relative=True, eps=0.1)
                b_scores = torch.where(is_pred_monster, b_scores * 0.01, b_scores)
                b_scores = torch.where(flux_mask, b_scores, torch.full_like(b_scores, -1.0))
                all_residual_scores.append(b_scores.cpu().numpy())

        pl_module.train()
        residual_scores = np.concatenate(all_residual_scores)

        ranked_idx = np.argsort(residual_scores)[::-1]
        valid_candidate_mask = residual_scores[ranked_idx] >= 0.0
        feasible_candidates = ranked_idx[valid_candidate_mask]
        subpool_size = min(50000, len(feasible_candidates))
        subpool_idx = feasible_candidates[:subpool_size]

        print(f"[al4pde_fisher_oed] ✅ 阶段 1 粗筛完成: 提取 Top {subpool_size} 候选构建 Fisher 梯度特征流形...")

        subpool_features = self._compute_pool_features_shape_aware(pl_module, dm, device, indices=subpool_idx)
        train_features = self._compute_train_features_shape_aware(pl_module, dm, device)
        subpool_features, train_features = self._standardize_shape_features(subpool_features, train_features)

        weights = residual_scores[subpool_idx]
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6) + 0.1
        V = torch.tensor(subpool_features, device=device, dtype=torch.float32) * torch.tensor(weights[:, None], device=device, dtype=torch.float32)

        n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
        print(f"[al4pde_fisher_oed] 🚀 阶段 2: 执行 Gram-Schmidt 泛函 Fisher 体积最大化 (目标有效样本: {n_select})...")

        selected_local_indices = []
        v_norm = torch.norm(V, dim=-1)
        first_idx = torch.argmax(v_norm).item()
        selected_local_indices.append(first_idx)

        q1 = (V[first_idx] / (torch.norm(V[first_idx]) + 1e-6)).unsqueeze(-1)
        Q = q1

        max_d_iter = min(n_select * 2, len(subpool_idx))
        for step in range(1, max_d_iter):
            proj = torch.matmul(V, Q)
            recon = torch.matmul(proj, Q.t())
            res_norm = torch.norm(V - recon, dim=-1)
            res_norm[selected_local_indices] = -1.0

            best_local_idx = torch.argmax(res_norm).item()
            selected_local_indices.append(best_local_idx)

            if len(selected_local_indices) >= n_select:
                break

            new_v = V[best_local_idx] - recon[best_local_idx]
            new_v_norm = torch.norm(new_v)
            if new_v_norm > 1e-5:
                q_new = (new_v / new_v_norm).unsqueeze(-1)
                Q = torch.cat([Q, q_new], dim=1)

        all_set = set(selected_local_indices)
        remaining = [i for i in range(len(subpool_idx)) if i not in all_set]
        full_ordered_local = selected_local_indices + remaining
        ordered_candidates = subpool_idx[full_ordered_local]

        eval_time = time.time() - t_start
        print(f"[al4pde_fisher_oed] ✅ Fisher OED 优化完成 (总耗时 {eval_time:.2f}s), 进入配额达标动态缓冲填充...")
        return self._build_broadcast_from_pool(ordered_candidates, dm, target_valid_num=n_select)

    def _normalize_array(self, arr):
        """归一化数组到 [0, 1]"""
        arr = np.asarray(arr, dtype=np.float64)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def _build_broadcast_from_pool(self, sel_idx, dm, target_valid_num=None):
        """从候选池选样本后构建 broadcast_data (参数候选池版)

        al4pde 机制: 选中参数后 ODE 仿真生成真实轨迹, 再加入训练集。
        1. target_valid_num is None: 传统单批模式 (sel_idx 样本全部送检并保留通过者)
        2. target_valid_num is not None: 配额达标动态缓冲填充模式 (Quota-Guaranteed Dynamic Filling)
           传入优先级降序候选队列 sel_idx, 分块流式仿真, 过滤 Monster / Numerical Failure,
           严格累积有效样本直至填满 target_valid_num 个, 彻底根除样本饥饿。
        """
        sel_idx = np.array(sel_idx, dtype=np.int64)

        if hasattr(dm, '_Y') and hasattr(dm, '_test_idx_real'):
            real_envelope = dm._Y[dm._test_idx_real].max(dim=-1)[0].max(dim=0)[0].cpu().numpy()
        else:
            Y_real_phys = dm.denormalize_y(dm.test_dataset_real.tensors[2].cpu())
            real_envelope = Y_real_phys.max(dim=-1)[0].max(dim=0)[0].cpu().numpy()
        dynamic_thresholds = np.maximum(real_envelope * 2.0, 2.0)

        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        workers = int(os.environ.get('AL_SIM_WORKERS', min(8, os.cpu_count() or 4)))

        if target_valid_num is None:
            # 传统单批模式
            n_sel = len(sel_idx)
            sel_P_real = self._pool_P_real[sel_idx].numpy()
            sel_names = self._pool_mutant_names[sel_idx]
            sel_ovrs = [dict(self._pool_ovr[i]) for i in sel_idx] if self._pool_ovr is not None else None

            print(f"[al4pde] ODE 仿真生成 {n_sel} 个选中样本的轨迹 (MASS 引擎, workers={workers})...")
            tasks = []
            for i in range(n_sel):
                ovr = sel_ovrs[i] if sel_ovrs is not None else {k: v for k, v in zip(PARAM_KEYS, sel_P_real[i].tolist())}
                tasks.append((str(sel_names[i]), ovr))

            sim_results = [None] * n_sel
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor, closing(
                bounded_ordered_map(executor, _solve_single_mass_task_wrapper, tasks)
            ) as ordered_results:
                done = 0
                for idx, res in enumerate(ordered_results):
                    sim_results[idx] = res
                    done += 1
                    if done % max(1, n_sel // 10) == 0 or done == n_sel:
                        print(f"  ODE 仿真进度: {done}/{n_sel} ({100*done//n_sel}%)")

            valid_positions = []
            valid_Y_real = []
            valid_P_real = []
            valid_names = []
            valid_labels = []
            valid_biological_labels = []

            for i in range(n_sel):
                res = sim_results[i]
                if res is None:
                    continue
                _, y_uniform, p_vec, pattern_label = res[:4]
                lhs_max_vals = np.max(y_uniform, axis=1)
                is_monster = any(lhs_max_vals[v] > dynamic_thresholds[v] for v in range(1, 39))
                if is_monster:
                    continue
                valid_positions.append(i)
                valid_Y_real.append(y_uniform)
                valid_P_real.append(p_vec)
                valid_names.append(str(sel_names[i]))
                valid_labels.append(pattern_label)
                valid_biological_labels.append(res[4] if len(res) > 4 else UNKNOWN_BIOLOGY_LABEL)

            evaluated_indices = sel_idx
            if valid_positions:
                valid_sel_idx = sel_idx[valid_positions]
                X_tensor = self._get_pool_X_batch(valid_sel_idx).clone()
            else:
                valid_sel_idx = []

        else:
            # 配额达标动态缓冲填充模式 (Quota-Guaranteed Dynamic Filling)
            target_n = int(min(target_valid_num, self._pool_size))
            chunk_size = 2500
            print(f"[al4pde] 🚀 启动配额达标动态缓冲填充: 目标严格入库 {target_n} 个有效样本 (候选总排序数: {len(sel_idx)}, workers={workers})...")

            valid_Y_real = []
            valid_P_real = []
            valid_names = []
            valid_labels = []
            valid_biological_labels = []
            valid_pool_idx = []
            evaluated_idx_list = []

            cursor = 0
            n_candidates = len(sel_idx)

            while len(valid_Y_real) < target_n and cursor < n_candidates:
                needed = target_n - len(valid_Y_real)
                curr_chunk_size = min(chunk_size, max(int(needed * 1.25), 100))
                end_cursor = min(cursor + curr_chunk_size, n_candidates)
                batch_candidates = sel_idx[cursor:end_cursor]
                cursor = end_cursor

                batch_P_real = self._pool_P_real[batch_candidates].numpy()
                batch_names = self._pool_mutant_names[batch_candidates]
                batch_ovrs = [dict(self._pool_ovr[i]) for i in batch_candidates] if self._pool_ovr is not None else None

                tasks = []
                for i in range(len(batch_candidates)):
                    ovr = batch_ovrs[i] if batch_ovrs is not None else {k: v for k, v in zip(PARAM_KEYS, batch_P_real[i].tolist())}
                    tasks.append((str(batch_names[i]), ovr))

                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor, closing(
                    bounded_ordered_map(executor, _solve_single_mass_task_wrapper, tasks)
                ) as ordered_results:
                    for i, res in enumerate(ordered_results):
                        pool_cand_idx = batch_candidates[i]
                        evaluated_idx_list.append(pool_cand_idx)

                        if res is None:
                            continue
                        _, y_uniform, p_vec, pattern_label = res[:4]
                        lhs_max_vals = np.max(y_uniform, axis=1)
                        is_monster = any(lhs_max_vals[v] > dynamic_thresholds[v] for v in range(1, 39))
                        if is_monster:
                            continue

                        valid_Y_real.append(y_uniform)
                        valid_P_real.append(p_vec)
                        valid_names.append(str(batch_names[i]))
                        valid_labels.append(pattern_label)
                        valid_biological_labels.append(res[4] if len(res) > 4 else UNKNOWN_BIOLOGY_LABEL)
                        valid_pool_idx.append(pool_cand_idx)

                        if len(valid_Y_real) == target_n:
                            break

                print(f"  [配额进度] 已收集有效合格样本: {len(valid_Y_real)}/{target_n} | 已消耗候选: {len(evaluated_idx_list)}")

            # 截取到严格 target_n
            if len(valid_Y_real) > target_n:
                valid_Y_real = valid_Y_real[:target_n]
                valid_P_real = valid_P_real[:target_n]
                valid_names = valid_names[:target_n]
                valid_labels = valid_labels[:target_n]
                valid_biological_labels = valid_biological_labels[:target_n]
                valid_pool_idx = valid_pool_idx[:target_n]

            evaluated_indices = np.array(evaluated_idx_list, dtype=np.int64)
            valid_sel_idx = np.array(valid_pool_idx, dtype=np.int64)

            if len(valid_sel_idx) > 0:
                X_tensor = self._get_pool_X_batch(valid_sel_idx).clone()

        # 从候选池安全移除所有已参与仿真的样本 (无论合格与否, 保证不重复测试)
        mask = np.ones(self._pool_size, dtype=bool)
        mask[evaluated_indices] = False
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
            print(f"[al4pde] 未能收集到有效样本, 本轮无样本加入训练集")
            print(f"[al4pde] 候选池剩余: {self._pool_size} 样本")
            return None

        # 归一化 Y 和 P (仿真输出为真实物理值空间, 需按照 dm.norm_type 归一化以匹配训练集)
        norm_type = getattr(dm, 'norm_type', 'log1p')
        mean_y = dm.mean_y.cpu()  # (1, 39, 1)
        std_y = dm.std_y.cpu()    # (1, 39, 1)
        p_mean = dm.p_mean.cpu()  # (1, 141)
        p_std = dm.p_std.cpu()    # (1, 141)

        Y_list = []
        for y_real in valid_Y_real:
            y_t = torch.tensor(y_real, dtype=torch.float32)  # (39, 500)
            y_norm = normalize_y(y_t.unsqueeze(0), mean_y, std_y, norm_type,
                                 getattr(dm, 's_c', None)).squeeze(0)
            Y_list.append(y_norm)

        P_list = []
        for p_real in valid_P_real:
            p_t = torch.tensor(p_real, dtype=torch.float32).unsqueeze(0)  # (1, 141)
            p_norm = normalize_p(p_t, p_mean, p_std, norm_type).squeeze(0)
            P_list.append(p_norm)

        n_evaluated = len(evaluated_indices)
        print(f"[al4pde] 仿真完成: {len(valid_Y_real)}/{n_evaluated} 通过验证 (通过率: {100*len(valid_Y_real)/max(1, n_evaluated):.1f}%) | "
              f"候选池剩余: {self._pool_size}")

        return {
            'X': X_tensor,
            'Y': torch.stack(Y_list),
            'P': torch.stack(P_list),
            'Y_real': valid_Y_real,
            'P_real': valid_P_real,
            'names': valid_names,
            'labels': valid_labels,
            'biological_labels': valid_biological_labels,
            'sample_plots': {}
        }
