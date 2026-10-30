"""AL orchestration and compatibility facade; numerical services live in modules."""
import os
import torch
from functools import wraps
from importlib import import_module
from .oracle import PARAM_KEYS
from budding_yeast_v2.data.label_metadata import UNKNOWN_BIOLOGY_LABEL
import numpy as np
import pytorch_lightning as pl
from .contracts import ALConfig, SelectionContext, SelectionPlan, QueryBatch
from .pool import CandidatePool, POOL_FIELDS
from .registry import build_strategy, canonical_name, is_strategy, SPECS, validate_system
from . import pool, prediction, features, oracle, dataset, io, backends
from budding_yeast_v2.utils.runtime_logging import log_event, log_stage


class ActiveLearningEngine(pl.Callback):
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
        self.pool = CandidatePool()
        self.config = ALConfig(**{key: value for key, value in locals().items() if key != 'self'})
        self.trigger_every_n_epochs = trigger_every_n_epochs
        self.strategy = str(strategy).lower()
        if not self._is_al4pde_strategy():
            raise ValueError(
                f"Unsupported or archived AL strategy {self.strategy!r}. "
                "Use a registered online strategy (al4pde, ours_response or maal). Historical trajectory-pool "
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

        if self._is_al4pde_strategy():
            self.initial_train_size = initial_train_size
            self.trigger_every_n_epochs = 4  # 保留字段, 外层重训模式下不生效
            self.num_add = num_add            # 保留字段, 外层重训模式下不生效
            self._al4pde_num_al_iter = 4
            self._al4pde_per_round_add = 10000
            print(f"[AL Manager] al4pde 设置: initial_train_size={self.initial_train_size}, "
                  f"{self._al4pde_num_al_iter} rounds; acquisitions = 1x, 2x, 3x initial training budget")

        self._al4pde_al_iter = 0           # 当前 AL 轮次 (0-indexed, 每次选择后递增)
        self._outer_loop_mode = False      # v2.3: 外层重训模式 (由 train.py 启用, 关闭 in-loop 触发)

        self.acquisition = build_strategy(self.strategy)

    def enable_outer_loop_mode(self):
        """Compatibility marker: selection is driven exclusively by train.py outer rounds."""
        self._outer_loop_mode = True


    def al4pde_setup(self, dm, device='cuda', exp_dir=None):
        """v2.3 外层重训式: 一次性初始化 (加载候选池 + 确定初始训练集 + 累积数据缓存)。
        供 train.py 在外层循环开始前调用一次。"""
        from torch.utils.data import TensorDataset

        self.system_id = getattr(dm, 'system_id', 'budding')
        validate_system(self.strategy, self.system_id)
        self._generate_param_only_pool(dm)  # 加载预存候选池 (如 pure_random_1m_pool.npz)

        # 缓存累积的物理空间数据，用于后续各轮次主动学习训练集落盘 (.npz)
        self.pool.validate()
        self._cum_synth_data = []
        self._cum_synth_params = []
        self._cum_synth_names = []
        self._cum_synth_labels = []
        self._cum_synth_biological_labels = []
        self._cum_synth_levels = []
        self._cum_synth_requested_levels = []
        self._cum_synth_module_counts = []
        self._cum_synth_provenance = []
        self._benchmark_real_provenance = []
        self._heldout = {}
        self._heldout_val_count = 0
        self._benchmark_real_data = []
        self._benchmark_real_params = []
        self._benchmark_real_names = []
        self._benchmark_real_labels = []
        self._benchmark_real_biological_labels = []
        self._param_names_cached = np.array(getattr(dm, 'param_names', PARAM_KEYS))

        dataset_path = getattr(dm, 'dataset_path', None)
        if dataset_path and os.path.exists(dataset_path):
            from budding_yeast_v2.data.dataset_archive import open_dataset
            init_npz = open_dataset(dataset_path, allow_pickle=True)
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
                io.cache_source_metadata(self,dm,init_npz)
            init_npz.close()
            io.configure_snapshot_storage(self, dm, exp_dir)

        # 检查是否已直接通过 --dataset_name 加载了预生成的初始训练集 (如 al_iter0_10000_210m_500s.npz)
        if hasattr(dm, 'train_dataset_synth') and len(dm.train_dataset_synth) > 0:
            print(f"[al4pde] ✅ 直接复用预生成的初始训练集: {len(dm.train_dataset_synth)} 样本 (无须重复进行初始 ODE 仿真)")
            if hasattr(dm, '_mutant_names'):
                existing_names = set(str(n) for n in dm._mutant_names) | getattr(self,'_initial_consumed_names',set())
                mask = np.isin(self._pool_mutant_names, list(existing_names), invert=True)
                if mask.sum() < len(mask):
                    n_removed = len(mask) - mask.sum()
                    print(f"[al4pde] 从候选池中剔除初始训练集已占用的 {n_removed} 个样本")
                    self.pool.keep(mask)

        else:
            initial_size = min(self.initial_train_size, self._pool_size)
            rng = np.random.RandomState(self.random_seed + 1)
            from budding_yeast_v2.data.systems.generation import level_schedule
            from budding_yeast_v2.data.systems.sampling_policy import POOL_RATIOS
            targets = np.bincount(level_schedule(initial_size,POOL_RATIOS,rng),minlength=4)
            levels = np.array([row['level'] for row in self._pool_mass_meta])
            selected = []
            for level in (1,2,3):
                available = np.flatnonzero(levels==level)
                if len(available)<targets[level]:raise ValueError(f'Pool cannot supply initial L{level} quota')
                selected.extend(rng.choice(available,int(targets[level]),replace=False))
            sel_idx = np.asarray(selected,dtype=np.int64)
            rng.shuffle(sel_idx)
            print(f"[al4pde] 按 5/40/55 分层随机抽取初始集: {initial_size}/{self._pool_size} (seed={self.random_seed + 1})")
            bd = self._build_broadcast_from_pool(sel_idx, dm)
            if bd is None or bd['X'].shape[0] == 0:
                raise RuntimeError("[al4pde] 初始训练集仿真全部失败, 无法继续")
            dm.train_dataset_synth = TensorDataset(bd['X'], bd['P'], bd['Y'],
                torch.as_tensor(dataset.pattern_target_ids(bd['labels'], self.system_id), dtype=torch.long, device=bd['Y'].device))
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
                self._cum_synth_provenance.extend(io.batch_provenance(bd))
                for key in ('levels','requested_levels','module_counts'):
                    getattr(self,'_cum_synth_'+key).extend(bd[key])

        self._al_initialized = True
        self._al4pde_al_iter = 0
        self._train_size_history.append(len(dm.train_dataset_synth))

        # 保存 Round 0 初始训练集到实验目录 datasets/
        if exp_dir and len(self._cum_synth_data) > 0:
            ds_dir = os.path.join(exp_dir,'rounds','round_00')
            os.makedirs(ds_dir, exist_ok=True)
            r0_path = os.path.join(ds_dir,'dataset.json')
            if not os.path.exists(r0_path):
                self._save_al_snapshot(r0_path, dm)
                print(f"💾 [al4pde] 已将 Round 0 初始数据集落盘保存至: {r0_path}")
            self._latest_dataset_path = r0_path

        print(f"[al4pde] 初始训练集就绪: {len(dm.train_dataset_synth)} 样本 | 候选池剩余: {self._pool_size} | 策略: {self.strategy}")


    def _generate_param_only_pool(self, dm):
        if getattr(dm, 'system_id', 'budding') == 'fission':
            from .fission import load_pool
            return load_pool(self, dm)
        return pool._generate_param_only_pool(self, dm)

    _get_pool_X_batch = pool._get_pool_X_batch
    refresh_model_inputs = pool.refresh_model_inputs

    _compute_mc_dropout_uncertainty = prediction._compute_mc_dropout_uncertainty

    set_ensemble_members = prediction.set_ensemble_members

    _compute_ensemble_uncertainty = prediction._compute_ensemble_uncertainty

    _compute_train_family_uncertainty = prediction._compute_train_family_uncertainty

    _bait_feature_batch = features._bait_feature_batch

    _compute_predicted_distance_features = features._compute_predicted_distance_features

    _compute_bait_features = features._compute_bait_features

    _compute_pool_features_spatial_mean = features._compute_pool_features_spatial_mean

    _compute_train_features_spatial_mean = features._compute_train_features_spatial_mean

    _extract_shape_features = features._extract_shape_features

    _standardize_shape_features = features._standardize_shape_features

    _compute_pool_features_shape_aware = features._compute_pool_features_shape_aware

    _compute_train_features_shape_aware = features._compute_train_features_shape_aware

    _compute_diversity_scores = features._compute_diversity_scores

    _normalize_array = features._normalize_array

    _append_dm_label_metadata = staticmethod(dataset._append_dm_label_metadata)

    _save_al_snapshot = io._save_al_snapshot

    _bmdal_lcmd_select = staticmethod(backends._bmdal_lcmd_select)

    _bmdal_predicted_max_dist_select = staticmethod(backends._bmdal_predicted_max_dist_select)

    _bmdal_bait_select = staticmethod(backends._bmdal_bait_select)

    _bmdal_max_dist_select = backends._bmdal_max_dist_select

    def __getattr__(self, name):
        if name in POOL_FIELDS:
            return getattr(self.pool, POOL_FIELDS[name])
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in POOL_FIELDS and 'pool' in self.__dict__:
            setattr(self.pool, POOL_FIELDS[name], value)
        else:
            object.__setattr__(self, name, value)

    def _is_al4pde_strategy(self):
        return is_strategy(self.strategy)

    def _al4pde_scheduled_num_add(self):
        """Valid-data budgets: initial_train_size times acquisition number 1, 2, 3."""
        iter_idx = getattr(self, '_al4pde_al_iter', 0)
        base_add = getattr(self, 'initial_train_size', 10000)
        scheduled = getattr(self,'_pending_add_budget',base_add * (iter_idx + 1))
        if getattr(self,'fill_acquisition_budget',False) and self._pool_size<scheduled:
            raise RuntimeError(f'Insufficient AL candidates for requested acquisition: {self._pool_size}/{scheduled}')
        return min(scheduled, self._pool_size)


    def make_selection_plan(self, indices, dm=None, target_valid_num=None):
        return self.pool.plan(indices, self.strategy, target_valid_num)

    def select(self, model, dm, device='cuda'):
        self.pool.ensure_ids()
        # Independent correctness fix: a new acquisition must use the current
        # model, not a full-pool feature matrix cached by an earlier round.
        self._pool_shape_features_cached = None
        context = SelectionContext(self, model, dm, device)
        plan = self.acquisition.select(context, self._al4pde_scheduled_num_add())
        if plan is not None and not isinstance(plan, SelectionPlan):
            raise TypeError('Strategies must return SelectionPlan or None')
        return plan

    def query(self, plan, dm):
        before = self.pool.ensure_ids().copy()
        bd = self.execute_plan(plan, dm)
        removed = set(map(int, np.setdiff1d(before, self.pool.ensure_ids(), assume_unique=True)))
        consumed = tuple(candidate_id for candidate_id in plan.candidate_ids if candidate_id in removed) if plan else ()
        accepted = tuple(map(int, bd.get('candidate_ids', ()))) if bd else ()
        self.last_query = QueryBatch(bd, consumed, accepted, self.strategy)
        if hasattr(self, '_pending_query_feedback'):
            from .strategies.query_feedback import observe_query
            observe_query(self, self.last_query)
        if hasattr(self, '_pending_hybrid_edges'):
            from .strategies.hybrid_feedback import observe_edges
            observe_edges(self,self.last_query)
        return self.last_query

    def execute_plan(self, plan, dm):
        if plan is None:
            return None
        if canonical_name(plan.strategy) != self.acquisition.spec.name:
            raise ValueError('Selection plan belongs to a different strategy')
        indices = self.pool.resolve(plan)
        if not len(indices):
            return None
        if plan.target_valid_num is None:
            return self._build_broadcast_from_pool(indices, dm)
        return self._build_broadcast_from_pool(indices, dm, target_valid_num=plan.target_valid_num)

    def _build_broadcast_from_pool(self, sel_idx, dm, target_valid_num=None):
        if getattr(dm, 'system_id', 'budding') == 'fission':
            from .fission import query_pool
            return query_pool(self, sel_idx, dm, target_valid_num)
        # Historical keyword/dict return contract; only the oracle mutates the pool.
        return oracle._build_broadcast_from_pool(self, sel_idx, dm, target_valid_num)

    def al4pde_select_and_append(self, pl_module, dm, device='cuda', exp_dir=None):
        if pl_module is not None:
            pl_module = pl_module.to(device)
            pl_module.eval()
        try:
            target=self._al4pde_scheduled_num_add()
            collected=[];consumed=[];accepted=0
            while accepted<target:
                self._pending_add_budget=target-accepted
                previous_size=self._pool_size
                with log_stage('candidate scoring and selection',strategy=self.strategy,
                               acquisition=self._al4pde_al_iter+1,pool_size=int(self._pool_size),
                               budget=int(self._pending_add_budget)):
                    plan = self.select(pl_module, dm, device)
                with log_stage('selected candidates ODE and validation',strategy=self.strategy,
                               acquisition=self._al4pde_al_iter+1,
                               selected=0 if plan is None else len(plan.candidate_ids)):
                    batch = self.query(plan, dm)
                if batch.data is not None:
                    collected.append(batch.data);accepted+=len(batch.data['names'])
                consumed.extend(batch.consumed_ids)
                log_event('PROGRESS','valid acquisition quota',strategy=self.strategy,
                          accepted=accepted,target=int(target),consumed=len(consumed))
                if not getattr(self,'fill_acquisition_budget',False):break
                if accepted<target and (self._pool_size==0 or self._pool_size>=previous_size):
                    raise RuntimeError(f'AL valid acquisition quota exhausted: {accepted}/{target}')
            if collected:
                bd=dict(collected[0])
                for key in ('X','Y','P'):bd[key]=torch.cat([row[key] for row in collected],dim=0)
                for key in ('Y_real','P_real','names','labels','biological_labels','levels','requested_levels','module_counts'):
                    bd[key]=[value for row in collected for value in row[key]]
                bd['provenance']=[value for row in collected for value in io.batch_provenance(row)]
                bd['candidate_ids']=np.concatenate([row['candidate_ids'] for row in collected])
                bd['consumed_ids']=np.asarray(consumed,dtype=np.int64)
            else:bd=None
            if bd is None:
                return 0
            with log_stage('training dataset append',strategy=self.strategy,samples=len(bd['names'])):
                n_added = dataset.append_training_batch(self, dm, bd)
            with log_stage('acquisition snapshot and audit saving',strategy=self.strategy,
                           acquisition=int(self._al4pde_al_iter),samples=n_added):
                io.save_round(self, dm, bd, exp_dir)
            return n_added
        finally:
            if hasattr(self,'_pending_add_budget'):del self._pending_add_budget
            if pl_module is not None:
                pl_module.train()


# Preferred public names; historical method names retain identical behavior.
ActiveLearningEngine.setup = ActiveLearningEngine.al4pde_setup
ActiveLearningEngine.select_and_append = ActiveLearningEngine.al4pde_select_and_append

ActiveLearningCallback = ActiveLearningEngine


# Backward-compatible private helpers delegate to actual service modules. They
# are not used to dispatch new acquisitions; the registry owns that dispatch.
def _legacy_query_method(module_name, function_name):
    function = getattr(import_module('.' + module_name, __package__), function_name)
    @wraps(function)
    def query(self, *args, **kwargs):
        plan = function(self, *args, **kwargs)
        dm = args[2] if len(args) > 2 else kwargs.get('dm')
        return self.execute_plan(plan, dm)
    return query

for _spec in SPECS.values():
    setattr(ActiveLearningEngine, _spec.function, _legacy_query_method(_spec.module, _spec.function))
del _spec
ActiveLearningEngine._power_sample_indices = staticmethod(getattr(import_module('.strategies.uncertainty', __package__), '_power_sample_indices'))
ActiveLearningEngine._compute_mass_priority_scores = getattr(import_module('.strategies.biology', __package__), '_compute_mass_priority_scores')
ActiveLearningEngine._compute_combination_novelty_scores = getattr(import_module('.strategies.biology', __package__), '_compute_combination_novelty_scores')
ActiveLearningEngine._map_family_unc_to_pool = getattr(import_module('.strategies.biology', __package__), '_map_family_unc_to_pool')
ActiveLearningEngine._map_family_priority_train_based = getattr(import_module('.strategies.biology', __package__), '_map_family_priority_train_based')
ActiveLearningEngine._SHAPE_FEAT_GROUP_SIZES = features._SHAPE_FEAT_GROUP_SIZES
ActiveLearningEngine._SHAPE_FEAT_GROUP_WEIGHTS = features._SHAPE_FEAT_GROUP_WEIGHTS
