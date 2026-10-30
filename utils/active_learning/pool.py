"""pool services extracted from the pre-package implementation."""
from budding_yeast_v2.utils.active_learning.oracle import PARAM_KEYS
from budding_yeast_v2.utils.active_learning.oracle import _MASS_DEFAULTS
from budding_yeast_v2.data.systems.budding.api import initial_conds as initial_conds_mass
from budding_yeast_v2.data.systems.budding.api import is_duplicate_real_mutant
import json
from budding_yeast_v2.data.normalization import normalize_p
from budding_yeast_v2.data.normalization import normalize_y
import numpy as np
import os
from budding_yeast_v2.data.systems.budding.api import prepare_initial_conditions
from budding_yeast_v2.data.systems.budding.api import sample_mass_random
import time
import torch


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


def refresh_model_inputs(self, dm):
    """Encode this round's model buffers without changing physical provenance.

    Exported ICs can be float64. Models and DataModule tensors use float32;
    normalization intentionally preserves its input dtype, so cast the local
    encoding inputs explicitly, on every round rather than only pool loading.
    """
    params = self._pool_P_real.detach().to(device='cpu', dtype=torch.float32)
    initial = self._pool_initial_physical.detach().to(device='cpu', dtype=torch.float32)
    self._pool_P = normalize_p(params, dm.p_mean.cpu(), dm.p_std.cpu(), dm.norm_type)
    self._pool_Y_ic = normalize_y(initial, dm.mean_y.cpu(), dm.std_y.cpu(), dm.norm_type,
                                 getattr(dm, 's_c', None))
    self._pool_X = None  # Any materialized inputs belong to the previous encoding.
    self._pool_shape_features_cached = None
    self._al4pde_mean_y, self._al4pde_std_y = dm.mean_y, dm.std_y


def _generate_param_only_pool(self, dm):
    """Load a checked pool; fallback uses the shared balanced quota generator."""
    if not self.al_pool_dataset:
        import tempfile
        from pathlib import Path
        from budding_yeast_v2.data.systems.contracts import GenerationConfig
        from budding_yeast_v2.data.systems.generation import generate_dataset
        with tempfile.TemporaryDirectory(prefix='budding_al_pool_') as directory:
            path = str(Path(directory)/'pool.npz')
            generate_dataset(GenerationConfig('budding_pool',n_samples=100000,
                seed=self.random_seed,output=path,include_real=False,parameter_only=True))
            self.al_pool_dataset = path
            try:return _generate_param_only_pool(self,dm)
            finally:self.al_pool_dataset = None
    if not os.path.isfile(self.al_pool_dataset):
        raise FileNotFoundError(self.al_pool_dataset)
    print(f'[AL pool] Reading cached candidates: {self.al_pool_dataset}', flush=True)
    from budding_yeast_v2.data.systems.sampling_policy import VERSION, valid_candidate
    from budding_yeast_v2.data.systems.contracts import Candidate
    with np.load(self.al_pool_dataset,allow_pickle=True) as data:
        if str(data.get('system_id','budding')) != 'budding':
            raise ValueError('Budding AL cannot load a Fission candidate pool')
        from budding_yeast_v2.data.systems.budding.al_pool import validate_pool_contract
        validate_pool_contract(data)
        physical = np.asarray(data['params'])
        names = np.asarray(data['mutant_names'])
        if physical.shape != (len(names),len(PARAM_KEYS)):
            raise ValueError('Budding pool parameter dimensions mismatch')
        if 'param_names' in data and tuple(data['param_names']) != tuple(PARAM_KEYS):
            raise ValueError('Budding pool parameter order mismatch')
        overrides = ([json.loads(str(value)) for value in data['overrides_json']]
                     if 'overrides_json' in data else None)
        stored_metadata = data['candidate_metadata_json'] if 'candidate_metadata_json' in data else None
        metadata = []
        print(f'[AL pool] Checking mutation metadata for {len(names)} candidates', flush=True)
        for i,name in enumerate(names):
            desc = _mass_desc_of(name)
            if desc is None:raise ValueError('Pool candidate lacks auditable module description; regenerate the pool')
            level = int(str(name).rsplit('_L',1)[1].split('_')[0])
            source_meta = json.loads(str(stored_metadata[i])) if stored_metadata is not None else {}
            source_meta['description'] = desc
            candidate = valid_candidate(Candidate(i,str(name),{} if overrides is None else overrides[i],level,
                                       metadata=source_meta), 'budding')
            if candidate is None:
                raise ValueError('Pool contains unsupported mutations or violates exact L1/L2/L3 policy; regenerate it')
            metadata.append(dict(candidate.metadata,genes=candidate.metadata['pairs'],level=level))
            if (i+1) % 100000 == 0 or i+1 == len(names):
                print(f'[AL pool] Mutation metadata {i+1}/{len(names)}', flush=True)
        if overrides is None:
            overrides = [dict(zip(PARAM_KEYS,map(float,row))) for row in physical]
        initial = np.asarray(data['y_init']) if 'y_init' in data else np.array([prepare_initial_conditions(ovr)[1] for ovr in overrides])
        if initial.shape != (len(names),39) or not np.isfinite(initial).all() or not np.isfinite(physical).all():
            raise ValueError('Malformed/nonfinite Budding pool inputs')
        # Historical pools may predate alias-aware real deduplication. Reject
        # them rather than silently changing the parameters a strategy scored.
        print('[AL pool] Checking parameters, initial states and real collisions (no ODE)', flush=True)
        for i,ovr in enumerate(overrides):
            expected = np.array([ovr.get(key,_MASS_DEFAULTS[key]) for key in PARAM_KEYS])
            if (not np.allclose(physical[i],expected,rtol=1e-6,atol=1e-8)
                    or not np.allclose(initial[i],prepare_initial_conditions(ovr)[1],rtol=1e-6,atol=1e-8)):
                raise ValueError(f'Pool inputs disagree with overrides for {names[i]}; regenerate the pool')
            if is_duplicate_real_mutant(ovr,_MASS_DEFAULTS):
                raise ValueError(f'Pool candidate {names[i]} collides with a real mutant; regenerate the pool')
            if (i+1) % 100000 == 0 or i+1 == len(names):
                print(f'[AL pool] Input validation {i+1}/{len(names)}', flush=True)
        steps = dm.train_dataset_synth.tensors[2].shape[-1] if hasattr(dm,'train_dataset_synth') else 500
        if 'time_grid' in data and not np.allclose(data['time_grid'],np.linspace(0,float(getattr(dm,'t_max',210)),steps)):
            raise ValueError('Budding pool time grid mismatch')
    p = torch.tensor(physical,dtype=torch.float32)
    self._pool_P_real,self._pool_ovr,self._pool_mass_meta = p,overrides,metadata
    # Preserve the exported physical ICs for provenance; model inputs above
    # intentionally remain float32 and use the existing normalization.
    self._pool_initial_physical = torch.tensor(initial,dtype=torch.float64)
    refresh_model_inputs(self, dm)
    self._pool_mutant_names = names
    self._pool_X = self._pool_Y = None
    self._pool_size = len(names)
    self._al4pde_t_grid = torch.linspace(0,1,steps).reshape(1,1,steps)
    self._al4pde_pool_loaded = True
    print(f'[AL pool] Loaded {len(names)} exact L1/L2/L3 candidates ({VERSION})')


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
    t_grid = self._al4pde_t_grid.to(device=IC.device, dtype=IC.dtype).expand(IC.shape[0], IC.shape[1], T)
    return torch.stack([IC, t_grid], dim=2)  # (B, 39, 2, T)




# Legacy attribute names map into ONE authoritative pool object. Aliases do not
# copy arrays; they retain compatibility with historical analysis/private probes.
POOL_FIELDS = {
    '_pool_P': 'params_normalized', '_pool_P_real': 'params_physical',
    '_pool_X': 'inputs', '_pool_Y': 'targets', '_pool_Y_ic': 'initial_states',
    '_pool_initial_physical': 'initial_physical',
    '_pool_mutant_names': 'names', '_pool_ovr': 'overrides',
    '_pool_mass_meta': 'mutation_metadata', '_pool_shape_features_cached': 'shape_features',
    '_al4pde_pool_idx': 'source_indices', '_pool_size': 'size',
}


class CandidatePool:
    """Own aligned candidate rows and immutable IDs across pool shrinkage."""
    def __init__(self):
        self.size = 0
        for field in set(POOL_FIELDS.values()) - {'size'}:
            setattr(self, field, None)
        self.candidate_ids = None
        self.revision = 0

    def ensure_ids(self):
        if self.candidate_ids is None:
            self.candidate_ids = np.arange(int(self.size), dtype=np.int64)
        elif len(self.candidate_ids) != self.size:
            raise ValueError('Pool size changed without an aligned pool operation')
        return self.candidate_ids

    def validate(self):
        self.ensure_ids()
        for field in set(POOL_FIELDS.values()) - {'size'}:
            value = getattr(self, field)
            if value is not None and len(value) != self.size:
                raise ValueError(f'Pool row alignment error: {field}')

    def keep(self, mask):
        self.validate()
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.size,):
            raise ValueError('Pool mask has the wrong shape')
        values = {}
        for field in set(POOL_FIELDS.values()) - {'size'}:
            value = getattr(self, field)
            if value is not None:
                values[field] = ([v for v, keep in zip(value, mask) if keep]
                                 if isinstance(value, list) else value[mask])
        for field, value in values.items():
            setattr(self, field, value)
        self.candidate_ids = self.candidate_ids[mask]
        self.size = int(mask.sum())
        self.revision += 1

    def plan(self, indices, strategy, target_valid_num=None):
        from .contracts import SelectionPlan
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= self.size).any():
            raise ValueError('Selection contains an invalid pool index')
        if len(np.unique(indices)) != len(indices):
            raise ValueError('Selection contains duplicate candidates')
        ids = self.ensure_ids()[indices]
        return SelectionPlan(tuple(map(int, ids)), id(self), self.revision,
                             strategy, target_valid_num)

    def resolve(self, plan):
        if plan.pool_token != id(self) or plan.pool_revision != self.revision:
            raise ValueError('Stale selection plan: candidate pool changed')
        ids = self.ensure_ids()
        wanted = np.asarray(plan.candidate_ids, dtype=np.int64)
        if len(np.unique(wanted)) != len(wanted):
            raise ValueError('Selection plan contains duplicate candidates')
        indices = np.searchsorted(ids, wanted)
        if (indices >= len(ids)).any() or not np.array_equal(ids[indices], wanted):
            raise ValueError('Selected candidate is no longer available')
        return indices
