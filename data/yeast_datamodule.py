import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import re # 记得在文件顶部导入 re
from .dataset_utils import load_yeast_dataset_universal


class YeastDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path='mydata/lhs_massive_210min_500steps_dual_labels.npz',
        adj_matrix_path='mydata/yeast_signed_adjacency_matrix.npy',
        batch_size_lhs=32,
        batch_size_real=8,
        val_batch_size=32,
        num_workers=4,
        seed=42,
        use_adj_matrix=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_path = dataset_path
        self.adj_matrix_path = adj_matrix_path
        self.batch_size_lhs = batch_size_lhs
        self.batch_size_real = batch_size_real
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.use_adj_matrix = use_adj_matrix
        
        # 🌟 新增：动态解析物理时间 t_max
        match = re.search(r'(\d+)min', dataset_path)
        self.t_max = int(match.group(1)) if match else 210

        self.mean_y = None
        self.std_y = None
        self.p_mean = None
        self.p_std = None
        self.T_STEPS = None
        self.num_params = None
        self.adj_matrix = None

        self.fixed_samples_for_plot = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        raw_data, param_conds, mutant_names, num_params, pattern_labels = load_yeast_dataset_universal(self.dataset_path)
        self.pattern_labels = pattern_labels  # 🌟 新增：保存形态标签供 eval.py 使用

        self.num_params = num_params
        self.T_STEPS = raw_data.shape[2]

        if self.use_adj_matrix:
            try:
                self.adj_matrix = np.load(self.adj_matrix_path)
                print(f"Successfully loaded {self.adj_matrix.shape[0]}x{self.adj_matrix.shape[1]} ODE physical topology matrix!")
            except FileNotFoundError:
                print("Warning: yeast_signed_adjacency_matrix.npy not found, using pure data-driven mode.")
                self.adj_matrix = None

        Y = torch.tensor(raw_data, dtype=torch.float32)
        Params = torch.tensor(param_conds, dtype=torch.float32)

        Y = torch.nan_to_num(Y, nan=0.0, posinf=1e4, neginf=0.0)
        Params = torch.nan_to_num(Params, nan=0.0, posinf=1e4, neginf=0.0)

        num_mutants, num_vars, T_STEPS = Y.shape

        lhs_indices = [i for i, name in enumerate(mutant_names) if '_LHS_' in str(name)]
        real_indices = [i for i, name in enumerate(mutant_names) if '_LHS_' not in str(name)]

        print(f"📊 [数据划分成功] 提取到 LHS 训练样本: {len(lhs_indices)} 个 | 真实测试突变体: {len(real_indices)} 个")

        if len(real_indices) == 0:
            print("⚠️ 警告: 未在数据集中发现任何真实评估样本！自动切出最后 10 个样本作为测试集。")
            real_indices = lhs_indices[-10:]
            lhs_indices = lhs_indices[:-10]

        # ======== 两级分层划分：先按突变体家族，再按 Pattern ========
        np.random.seed(self.seed)

        # 第一级：按突变体家族分组
        family_groups = {}
        for idx in lhs_indices:
            name = str(mutant_names[idx])
            family = name.split('_LHS_')[0] if '_LHS_' in name else name
            family_groups.setdefault(family, []).append(idx)

        train_idx_lhs = []
        val_idx_lhs = []

        for family, family_indices in family_groups.items():
            # 第二级：在该家族内按 Pattern 分组
            pattern_subgroups = {}
            for idx in family_indices:
                pattern = str(pattern_labels[idx]).split(':')[0]  # "Pattern_A", "Pattern_B", "Pattern_C"
                pattern_subgroups.setdefault(pattern, []).append(idx)

            n_family = len(family_indices)
            n_val_target = max(1, int(n_family * 0.1))  # 该家族应分给验证集的样本数

            # 每个 Pattern 子组内部先洗牌
            for pat, pat_indices in pattern_subgroups.items():
                np.random.shuffle(pat_indices)

            # 按各 Pattern 子组的样本量比例分配验证集名额
            val_from_family = []
            train_from_family = []

            # 按 Pattern 子组大小降序排列，优先保证大组的划分精度
            sorted_patterns = sorted(pattern_subgroups.items(), key=lambda x: len(x[1]), reverse=True)

            # 计算每个 Pattern 子组应分给验证集的数量（按比例）
            val_quota_per_pattern = {}
            remaining_val_quota = n_val_target
            remaining_total = n_family

            for pat, pat_indices in sorted_patterns:
                n_pat = len(pat_indices)
                # 按该子组在家族中的占比分配验证名额
                quota = round(n_pat / remaining_total * remaining_val_quota) if remaining_total > 0 else 0
                # 边界保护：子组只有1个样本时不强制抽验证集（除非整个家族只有1个样本）
                if n_pat == 1 and n_family > 1:
                    quota = 0
                quota = min(quota, n_pat, remaining_val_quota)
                val_quota_per_pattern[pat] = quota
                remaining_val_quota -= quota
                remaining_total -= n_pat

            # 若因取整导致还有剩余名额，从最大的子组中补齐
            if remaining_val_quota > 0:
                for pat, pat_indices in sorted_patterns:
                    if remaining_val_quota <= 0:
                        break
                    available = len(pat_indices) - val_quota_per_pattern[pat]
                    extra = min(available, remaining_val_quota)
                    val_quota_per_pattern[pat] += extra
                    remaining_val_quota -= extra

            # 执行划分
            for pat, pat_indices in sorted_patterns:
                n_val = val_quota_per_pattern.get(pat, 0)
                val_from_family.extend(pat_indices[:n_val])
                train_from_family.extend(pat_indices[n_val:])

            train_idx_lhs.extend(train_from_family)
            val_idx_lhs.extend(val_from_family)

        np.random.shuffle(train_idx_lhs)
        np.random.shuffle(val_idx_lhs)

        n_families = len(family_groups)
        print(f"📊 [两级分层划分] 家族数: {n_families} | 训练集: {len(train_idx_lhs)} | 验证集: {len(val_idx_lhs)}")

        test_idx_real = real_indices

        # 归一化统计量使用全部 LHS 数据，避免稀有 Pattern 缺席导致的偏差
        self.mean_y = Y[lhs_indices].mean(dim=(0, 2), keepdim=True)
        self.std_y = Y[lhs_indices].std(dim=(0, 2), keepdim=True) + 1e-6

        self.p_mean = Params[lhs_indices].mean(dim=0, keepdim=True)
        self.p_std = Params[lhs_indices].std(dim=0, keepdim=True) + 1e-6

        Y_norm = (Y - self.mean_y) / self.std_y
        Y_norm = torch.nan_to_num(Y_norm, nan=0.0, posinf=10.0, neginf=-10.0)

        Params_norm = (Params - self.p_mean) / self.p_std
        Params_norm = torch.nan_to_num(Params_norm, nan=0.0, posinf=10.0, neginf=-10.0)

        IC = Y_norm[:, :, 0:1].expand(-1, -1, T_STEPS)
        t_grid = torch.linspace(0, 1, T_STEPS).reshape(1, 1, T_STEPS).expand(num_mutants, num_vars, T_STEPS)
        X = torch.stack([IC, t_grid], dim=2)

        self.train_dataset_lhs = TensorDataset(X[train_idx_lhs], Params_norm[train_idx_lhs], Y_norm[train_idx_lhs])
        self.val_dataset_lhs = TensorDataset(X[val_idx_lhs], Params_norm[val_idx_lhs], Y_norm[val_idx_lhs])
        self.test_dataset_real = TensorDataset(X[test_idx_real], Params_norm[test_idx_real], Y_norm[test_idx_real])

        fixed_lhs_idx_1 = train_idx_lhs[0]
        fixed_lhs_idx_2 = train_idx_lhs[1] if len(train_idx_lhs) > 1 else train_idx_lhs[0]

        wt_idx = None
        ko_idx = None

        for i, name in enumerate(mutant_names):
            if name == '000_WT_Healthy' or name == '1_WT_Glc':
                wt_idx = i
            elif name == '3_cln1_cln2_KO':
                ko_idx = i

        if wt_idx is None:
            wt_idx = test_idx_real[0] if len(test_idx_real) > 0 else train_idx_lhs[0]
        if ko_idx is None:
            ko_idx = test_idx_real[min(1, len(test_idx_real) - 1)] if len(test_idx_real) > 0 else val_idx_lhs[0]

        # ======== 4. 基于 Pattern Label 的分层画图样本选取 ========
        self.fixed_samples_for_plot = {}
        patterns = ['Pattern_A', 'Pattern_B', 'Pattern_C']
        
        def find_sample_by_pattern(indices, pattern_kw, fallback_indices):
            for idx in indices:
                if pattern_kw in str(pattern_labels[idx]):
                    return idx
            return fallback_indices[0] if len(fallback_indices) > 0 else 0

        # A. 训练集 (LHS) 抽取 3 种形态
        for p in patterns:
            idx = find_sample_by_pattern(train_idx_lhs, p, train_idx_lhs)
            self.fixed_samples_for_plot[f'LHS_{p}'] = (
                X[idx:idx+1].clone(), Params_norm[idx:idx+1].clone(), 
                Y_norm[idx:idx+1].clone(), mutant_names[idx], pattern_labels[idx]
            )

        # B. 盲测集 (Real) 抽取 3 种形态
        for p in patterns:
            idx = find_sample_by_pattern(test_idx_real, p, test_idx_real)
            self.fixed_samples_for_plot[f'Real_{p}'] = (
                X[idx:idx+1].clone(), Params_norm[idx:idx+1].clone(), 
                Y_norm[idx:idx+1].clone(), mutant_names[idx], pattern_labels[idx]
            )

        self._raw_data = raw_data
        self._param_conds = param_conds
        self._mutant_names = mutant_names
        self._Y = Y
        self._train_idx_lhs = train_idx_lhs
        self._val_idx_lhs = val_idx_lhs
        self._test_idx_real = test_idx_real

    def train_dataloader(self):
        return DataLoader(self.train_dataset_lhs, batch_size=self.batch_size_lhs,
                          shuffle=True, drop_last=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return [
            DataLoader(self.val_dataset_lhs, batch_size=self.val_batch_size,
                       shuffle=False, num_workers=self.num_workers, pin_memory=True),
            DataLoader(self.test_dataset_real, batch_size=self.val_batch_size,
                       shuffle=False, num_workers=self.num_workers, pin_memory=True),
        ]

    @property
    def test_data(self):
        return self._raw_data, self._param_conds, self._mutant_names

    @property
    def test_indices(self):
        return self._test_idx_real, self._val_idx_lhs