import math
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import re
from .dataset_utils import load_yeast_dataset_universal
from .yeast_graph import build_yeast_graph
from .normalization import (check_norm_type, transform_y, normalize_y, normalize_p,
                            denormalize_y, denormalize_p)
from .preprocessing_contract import (dataset_contract, make_preprocessing_state,
                                     validate_preprocessing_state, assert_compatible,
                                     check_fit_scope, TRAIN_FIT_SCOPE)


class YeastDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path='mydata/mass_100000_dataset_210min_500steps_dual_labels.npz',
        adj_matrix_path=None,
        batch_size_synth=32,
        batch_size_real=8,
        val_batch_size=32,
        num_workers=4,
        seed=42,
        use_adj_matrix=True,
        norm_type='log1p',
        graph_view='hybrid',
        preprocessing_state=None,
        fit_scope=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['preprocessing_state'])

        self.dataset_path = dataset_path
        self.adj_matrix_path = adj_matrix_path
        self.batch_size_synth = batch_size_synth
        self.batch_size_real = batch_size_real
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.use_adj_matrix = use_adj_matrix
        self.norm_type = check_norm_type(norm_type)
        self._expected_preprocessing = (
            validate_preprocessing_state(preprocessing_state) if preprocessing_state is not None else None
        )
        if self._expected_preprocessing is not None and self._expected_preprocessing['norm_type'] != norm_type:
            raise ValueError('DataModule normalization contradicts checkpoint preprocessing')
        saved_scope = (self._expected_preprocessing['fit_scope']
                       if self._expected_preprocessing is not None else None)
        if fit_scope is not None:
            check_fit_scope(fit_scope)
        if saved_scope is not None and fit_scope is not None and fit_scope != saved_scope:
            raise ValueError('DataModule fit_scope contradicts checkpoint preprocessing')
        self.fit_scope = check_fit_scope(saved_scope or fit_scope or TRAIN_FIT_SCOPE)
        self.hparams.fit_scope = self.fit_scope
        self.graph_view = graph_view
        
        # 🌟 新增：动态解析物理时间 t_max
        match = re.search(r'(\d+)min', dataset_path)
        self.t_max = int(match.group(1)) if match else 210

        self.mean_y = None
        self.std_y = None
        self.s_c = None
        self.p_mean = None
        self.p_std = None
        self.T_STEPS = None
        self.num_params = None
        self.adj_matrix = None

        self.fixed_samples_for_plot = None
        self.is_mass_dataset = False

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # v2.3: setup 幂等 —— al4pde 外层重训会多次 trainer.fit(datamodule=...),
        # 若每次都重建数据集会抹掉 AL 追加的样本; 故仅首次真正构建。
        if getattr(self, '_setup_done', False):
            return
        raw_data, param_conds, mutant_names, num_params, pattern_labels = load_yeast_dataset_universal(self.dataset_path)
        self.pattern_labels = pattern_labels  # 🌟 新增：保存形态标签供 eval.py 使用

        self.num_params = num_params
        self.T_STEPS = raw_data.shape[2]
        self._dataset_contract = dataset_contract(
            raw_data, param_conds, mutant_names, pattern_labels, self.dataset_path, self.t_max
        )
        if (self._expected_preprocessing is not None
                and self._expected_preprocessing.get('dataset') != self._dataset_contract):
            raise ValueError('Checkpoint preprocessing mismatch: dataset; refusing silent refit/reinterpretation')

        if self.use_adj_matrix:
            # Compatibility exposure only: production models own their graph.
            self.adj_matrix = (
                np.load(self.adj_matrix_path, allow_pickle=False) if self.adj_matrix_path is not None
                else build_yeast_graph().adjacency(self.graph_view, self_loops='all').astype(np.float32)
            )
            if self.adj_matrix.shape != (39, 39) or not np.isfinite(self.adj_matrix).all():
                raise ValueError('Expected a finite 39x39 target/source variable graph')

        Y = torch.tensor(raw_data, dtype=torch.float32)
        Params = torch.tensor(param_conds, dtype=torch.float32)

        Y = torch.nan_to_num(Y, nan=0.0, posinf=1e4, neginf=0.0)
        Params = torch.nan_to_num(Params, nan=0.0, posinf=1e4, neginf=0.0)

        num_mutants, num_vars, T_STEPS = Y.shape

        # ================================================================
        #  数据集类型通用智能识别:
        #  - LHS 数据集: 合成样本名含 '_LHS_' (按母体家族两级分层划分)。
        #  - 非 LHS 数据集 (MASS / WB-MASS / 任意未来自定义生成数据集):
        #    自动按真实突变体特征 (如 1_WT_Glc, 15_cln3Δ...) 区分真实测试集与合成样本，
        #    跳过家族分层, 仅按 Pattern 宏观划分。两类数据集都严格将 real mutants
        #    全部划入测试集 (test_real); 新训练的归一化统计量只用合成训练拆分。
        #    这不保证等价输入跨拆分隔离，重复样本分组策略尚未修改。
        # ================================================================
        has_lhs = any('_LHS_' in str(name) for name in mutant_names)
        self.is_mass_dataset = not has_lhs

        if has_lhs:
            synth_indices = [i for i, name in enumerate(mutant_names) if '_LHS_' in str(name)]
            real_indices = [i for i, name in enumerate(mutant_names) if '_LHS_' not in str(name)]
            print(f"📊 [LHS 家族数据集] 提取到合成样本: {len(synth_indices)} 个 | 真实测试突变体: {len(real_indices)} 个")
        else:
            # 通用非 LHS 识别逻辑:
            # 真实突变体均为 Chen(2004) 标准命名 (如 1_WT_Glc, 15_cln3Δ..., 126_...), 均以纯数字+下划线开头
            # 合成样本为 MASS_..., WB-MASS_..., Random_..., Synth_... 等自定义命名
            def _is_real_mutant(n):
                s = str(n)
                return bool(re.match(r'^\d+(\.\d+)?_', s)) and not any(s.startswith(p) for p in ('MASS_', 'WB-MASS_', 'WB_', 'SYNTH_', 'RANDOM_'))

            real_indices = [i for i, name in enumerate(mutant_names) if _is_real_mutant(name)]
            synth_indices = [i for i, name in enumerate(mutant_names) if not _is_real_mutant(name)]
            print(f"📊 [通用非 LHS 数据集] 提取到合成样本: {len(synth_indices)} 个 | "
                  f"真实测试突变体: {len(real_indices)} 个 (跳过 LHS 家族分层, 仅按 Pattern 宏观划分)")

        if len(real_indices) == 0:
            print("⚠️ 警告: 未在数据集中发现任何真实评估样本！自动切出最后 10 个样本作为测试集。")
            real_indices = synth_indices[-10:]
            synth_indices = synth_indices[:-10]

        # ================================================================
        #  双级分层 + 孤本保护 数据划分机制
        #  步骤 2: 第一级 — 按母体家族（Family）分组
        #  步骤 3: 第二级 — 按动态表型（Pattern）子组分组
        #  步骤 4: 比例抽样 + 孤本保护（每个 Pattern 子组内独立计算）
        # ================================================================
        np.random.seed(self.seed)

        # 第一级：按突变体家族分组（提取 _LHS_ 前的前缀）
        # MASS 数据集无家族结构 → 跳过家族分层: 将全部 MASS 合成样本归为单一分组 'MASS',
        # 后续仍按 Pattern 子组做宏观 train/val/test 划分 (保留 Pattern 分层 + 孤本保护)。
        family_groups = {}
        if self.is_mass_dataset:
            family_groups['MASS'] = list(synth_indices)
        else:
            for idx in synth_indices:
                name = str(mutant_names[idx])
                family = name.split('_LHS_')[0] if '_LHS_' in name else name
                family_groups.setdefault(family, []).append(idx)

        train_idx_synth = []
        val_idx_synth = []
        test_idx_synth = []

        for family, family_indices in family_groups.items():
            # 第二级：在该家族内按 Pattern 子组分组
            pattern_subgroups = {}
            for idx in family_indices:
                pattern = str(pattern_labels[idx]).split(':')[0]  # "Pattern_A/B/C"
                pattern_subgroups.setdefault(pattern, []).append(idx)

            # 每个 Pattern 子组内部洗牌后，独立计算抽样名额
            for pat, pat_indices in pattern_subgroups.items():
                np.random.shuffle(pat_indices)
                n_pat = len(pat_indices)

                # --- 默认分配：各 10% ---
                val_quota = math.ceil(n_pat * 0.10)
                test_quota = math.ceil(n_pat * 0.10)

                # --- ⚠️ 孤本保护机制 ---
                if n_pat <= 2:
                    # 样本极少，全部保送训练集，防止流形丢失
                    val_quota = 0
                    test_quota = 0
                elif n_pat <= 4:
                    # 只保留验证集，不抽测试集
                    test_quota = 0

                # --- 安全钳：防止 val + test 超过子组总量 ---
                if val_quota + test_quota > n_pat:
                    test_quota = max(0, n_pat - val_quota)

                # --- 执行切分：前 val_quota → 验证集，接着 test_quota → 测试集，剩余 → 训练集 ---
                val_from_pat = pat_indices[:val_quota]
                test_from_pat = pat_indices[val_quota:val_quota + test_quota]
                train_from_pat = pat_indices[val_quota + test_quota:]

                val_idx_synth.extend(val_from_pat)
                test_idx_synth.extend(test_from_pat)
                train_idx_synth.extend(train_from_pat)

                print(f"   [Family={family}] {pat}: total={n_pat:>5d}  "
                      f"→ train={len(train_from_pat):>5d}  val={len(val_from_pat):>3d}  test={len(test_from_pat):>3d}")

        # 最终整体洗牌
        np.random.shuffle(train_idx_synth)
        np.random.shuffle(val_idx_synth)
        np.random.shuffle(test_idx_synth)

        test_idx_real = real_indices

        n_families = len(family_groups)
        print(f"📊 [两级分层+孤本保护] 家族数: {n_families} | "
              f"训练集: {len(train_idx_synth)} | 验证集: {len(val_idx_synth)} | "
              f"合成测试集: {len(test_idx_synth)} | 真实测试集: {len(test_idx_real)}")

        # Split first, fit on source-file synthetic TRAIN only, then freeze the
        # same transform for every split/AL round. Legacy replay is explicit;
        # a saved payload always restores its original tensors and policy.
        fit_indices = train_idx_synth if self.fit_scope == TRAIN_FIT_SCOPE else synth_indices
        if not fit_indices:
            raise ValueError('No synthetic rows available for normalization fitting')
        if self._expected_preprocessing is not None:
            state = validate_preprocessing_state(self._expected_preprocessing, num_vars, Params.shape[1])
            for key in ('mean_y', 'std_y', 'p_mean', 'p_std', 's_c'):
                value = state.get(key)
                setattr(self, key, value.clone() if isinstance(value, torch.Tensor) else None)
        else:
            if self.norm_type == 'asinh_scale':
                sub_n = min(len(fit_indices), 20000)
                s_c_np = np.quantile(np.maximum(Y[fit_indices[:sub_n]].numpy(), 0.0),
                                     0.90, axis=(0, 2), keepdims=True)
                self.s_c = torch.tensor(np.maximum(s_c_np, 0.01), dtype=torch.float32)
            Y_trans = transform_y(Y[fit_indices], self.norm_type, self.s_c)
            self.mean_y = Y_trans.mean(dim=(0, 2), keepdim=True)
            self.std_y = Y_trans.std(dim=(0, 2), keepdim=True) + 1e-6
            P_fit = Params[fit_indices]
            P_trans = torch.log10(P_fit.clamp(min=1e-8)) if self.norm_type == 'asinh_scale' else P_fit
            self.p_mean = P_trans.mean(dim=0, keepdim=True)
            self.p_std = P_trans.std(dim=0, keepdim=True)
            if self.norm_type in ('log1p', 'linear'):
                self.p_std = self.p_std + 1e-6
            del Y_trans
        print(f'📐 Normalization: {self.norm_type}; fit_scope={self.fit_scope}; '
              f'{"restored checkpoint statistics" if self._expected_preprocessing is not None else f"fit rows={len(fit_indices)}"}')
        Y_norm = self.normalize_y(Y)
        Params_norm = self.normalize_p(Params)

        IC = Y_norm[:, :, 0:1].expand(-1, -1, T_STEPS)
        t_grid = torch.linspace(0, 1, T_STEPS).reshape(1, 1, T_STEPS).expand(num_mutants, num_vars, T_STEPS)
        X = torch.stack([IC, t_grid], dim=2)

        # 映射 Pattern 分岔形态标签为 0 (A), 1 (B), 2 (C)
        def _to_pat_id(lbl):
            s = str(lbl)
            if 'Pattern_A' in s: return 0
            elif 'Pattern_B' in s: return 1
            elif 'Pattern_C' in s: return 2
            return 1
        pattern_ids = torch.tensor([_to_pat_id(lbl) for lbl in pattern_labels], dtype=torch.long)

        self.train_dataset_synth = TensorDataset(X[train_idx_synth], Params_norm[train_idx_synth], Y_norm[train_idx_synth], pattern_ids[train_idx_synth])
        self.val_dataset_synth = TensorDataset(X[val_idx_synth], Params_norm[val_idx_synth], Y_norm[val_idx_synth])
        self.test_dataset_synth = TensorDataset(X[test_idx_synth], Params_norm[test_idx_synth], Y_norm[test_idx_synth])
        self.test_dataset_real = TensorDataset(X[test_idx_real], Params_norm[test_idx_real], Y_norm[test_idx_real])

        # ======== 4. 基于 Pattern 与 Badcase 靶向监控样本选取 (精简至 14 个核心样本, 零 LHS 标识) ========
        self.fixed_samples_for_plot = {}
        patterns = ['Pattern_A', 'Pattern_B', 'Pattern_C']
        
        def find_samples_by_pattern(indices, pattern_kw, n_samples):
            """从 indices 中按 pattern_kw 选出 n_samples 个样本的索引列表"""
            matched = [idx for idx in indices if pattern_kw in str(pattern_labels[idx])]
            if len(matched) == 0:
                matched = indices[:n_samples] if len(indices) >= n_samples else indices
            return matched[:n_samples]

        def find_real_sample_by_prefix(indices, prefix):
            """从真实突变体 indices 中按前缀精确查找样本索引"""
            for idx in indices:
                name = str(mutant_names[idx])
                if name.startswith(prefix) or name.split('_')[0] == prefix.rstrip('_'):
                    return idx
            # 备用容错匹配
            for idx in indices:
                name = str(mutant_names[idx])
                if prefix in name:
                    return idx
            return None

        # A. 合成训练集: 每个 Pattern 选 1 个 (共 3 个)
        for p in patterns:
            for idx in find_samples_by_pattern(train_idx_synth, p, 1):
                self.fixed_samples_for_plot[f'synth_train/{p}'] = (
                    X[idx:idx+1].clone(), Params_norm[idx:idx+1].clone(), 
                    Y_norm[idx:idx+1].clone(), mutant_names[idx], pattern_labels[idx]
                )

        # B. 合成测试集: 每个 Pattern 选 1 个 (共 3 个)
        for p in patterns:
            for idx in find_samples_by_pattern(test_idx_synth, p, 1):
                self.fixed_samples_for_plot[f'synth_test/{p}'] = (
                    X[idx:idx+1].clone(), Params_norm[idx:idx+1].clone(), 
                    Y_norm[idx:idx+1].clone(), mutant_names[idx], pattern_labels[idx]
                )

        # C. 真实突变体典型对照样本 (指定 3 个非 WT 高保真样本: #112 mad2Δ, #87 GAL-TEM1, #100 GAL-CDC14)
        typical_prefixes = ['112_', '87_', '100_']
        for prefix in typical_prefixes:
            matched_idx = find_real_sample_by_prefix(test_idx_real, prefix)
            if matched_idx is not None:
                self.fixed_samples_for_plot[f'real_typical/{mutant_names[matched_idx]}'] = (
                    X[matched_idx:matched_idx+1].clone(), Params_norm[matched_idx:matched_idx+1].clone(),
                    Y_norm[matched_idx:matched_idx+1].clone(), mutant_names[matched_idx], pattern_labels[matched_idx]
                )

        # D. 真实突变体共有核心 Badcase (指定 5 个跨全部实验共有最差样本: #15, #22, #20, #42, #76)
        badcase_prefixes = ['15_', '22_', '20_', '42_', '76_']
        for prefix in badcase_prefixes:
            matched_idx = find_real_sample_by_prefix(test_idx_real, prefix)
            if matched_idx is not None:
                self.fixed_samples_for_plot[f'real_badcases/{mutant_names[matched_idx]}'] = (
                    X[matched_idx:matched_idx+1].clone(), Params_norm[matched_idx:matched_idx+1].clone(),
                    Y_norm[matched_idx:matched_idx+1].clone(), mutant_names[matched_idx], pattern_labels[matched_idx]
                )

        self._raw_data = raw_data
        self._param_conds = param_conds
        self._mutant_names = mutant_names
        self._Y = Y
        self._train_idx_synth = train_idx_synth
        self._val_idx_synth = val_idx_synth
        self._test_idx_synth = test_idx_synth
        self._test_idx_real = test_idx_real
        # Freeze the source-file split for preprocessing replay. AL may later
        # change the effective training indices, but not normalization fit data
        # or the held-out source split. This is not an AL-pool resume snapshot.
        self._preprocessing_split = {
            'seed': self.seed, 'train': list(train_idx_synth), 'val': list(val_idx_synth),
            'test': list(test_idx_synth), 'real': list(test_idx_real),
        }
        if self._expected_preprocessing is not None:
            assert_compatible(self._expected_preprocessing, self.preprocessing_state())
        self._setup_done = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset_synth, batch_size=self.batch_size_synth,
                          shuffle=True, drop_last=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return [
            DataLoader(self.val_dataset_synth, batch_size=self.val_batch_size,
                       shuffle=False, num_workers=self.num_workers, pin_memory=True),
            DataLoader(self.test_dataset_synth, batch_size=self.val_batch_size,
                       shuffle=False, num_workers=self.num_workers, pin_memory=True),
            DataLoader(self.test_dataset_real, batch_size=self.val_batch_size,
                       shuffle=False, num_workers=self.num_workers, pin_memory=True),
        ]

    @property
    def test_data(self):
        return self._raw_data, self._param_conds, self._mutant_names

    @property
    def test_indices(self):
        return self._test_idx_synth, self._test_idx_real

    @property
    def train_dataset_lhs(self):
        """兼容属性: 自动重定向到 train_dataset_synth"""
        return self.train_dataset_synth

    @train_dataset_lhs.setter
    def train_dataset_lhs(self, value):
        self.train_dataset_synth = value

    @property
    def _train_idx_lhs(self):
        """兼容属性: 自动重定向到 _train_idx_synth"""
        return self._train_idx_synth

    @_train_idx_lhs.setter
    def _train_idx_lhs(self, value):
        self._train_idx_synth = value

    def denormalize_y(self, y_norm, device=None):
        return denormalize_y(y_norm, self.mean_y, self.std_y, self.norm_type, self.s_c)

    def denormalize_p(self, p_norm, device=None):
        return denormalize_p(p_norm, self.p_mean, self.p_std, self.norm_type)

    def normalize_y(self, y):
        return normalize_y(y, self.mean_y, self.std_y, self.norm_type, self.s_c)

    def normalize_p(self, p):
        return normalize_p(p, self.p_mean, self.p_std, self.norm_type)

    def preprocessing_state(self):
        return make_preprocessing_state(self)
