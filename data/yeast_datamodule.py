import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import re
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
        test_idx_lhs = []

        for family, family_indices in family_groups.items():
            # 第二级：在该家族内按 Pattern 分组
            pattern_subgroups = {}
            for idx in family_indices:
                pattern = str(pattern_labels[idx]).split(':')[0]  # "Pattern_A", "Pattern_B", "Pattern_C"
                pattern_subgroups.setdefault(pattern, []).append(idx)

            n_family = len(family_indices)
            n_val_target = max(1, int(n_family * 0.10))  # 该家族应分给验证集的样本数 (10%)
            n_test_target = max(1, int(n_family * 0.10))  # 该家族应分给测试集的样本数 (10%)

            # 每个 Pattern 子组内部先洗牌
            for pat, pat_indices in pattern_subgroups.items():
                np.random.shuffle(pat_indices)

            # 按各 Pattern 子组的样本量比例分配验证集和测试集名额
            val_from_family = []
            test_from_family = []
            train_from_family = []

            # 按 Pattern 子组大小降序排列，优先保证大组的划分精度
            sorted_patterns = sorted(pattern_subgroups.items(), key=lambda x: len(x[1]), reverse=True)

            # 计算每个 Pattern 子组应分给验证集和测试集的数量（按比例）
            val_quota_per_pattern = {}
            test_quota_per_pattern = {}
            remaining_val_quota = n_val_target
            remaining_test_quota = n_test_target
            remaining_total = n_family

            for pat, pat_indices in sorted_patterns:
                n_pat = len(pat_indices)
                # 按该子组在家族中的占比分配验证集名额
                val_quota = round(n_pat / remaining_total * remaining_val_quota) if remaining_total > 0 else 0
                test_quota = round(n_pat / remaining_total * remaining_test_quota) if remaining_total > 0 else 0
                # 边界保护：子组只有1-2个样本时不强制抽验证/测试集
                if n_pat <= 2 and n_family > 2:
                    val_quota = 0
                    test_quota = 0
                elif n_pat <= 4 and n_family > 4:
                    test_quota = 0
                val_quota = min(val_quota, n_pat, remaining_val_quota)
                test_quota = min(test_quota, n_pat - val_quota, remaining_test_quota)
                val_quota_per_pattern[pat] = val_quota
                test_quota_per_pattern[pat] = test_quota
                remaining_val_quota -= val_quota
                remaining_test_quota -= test_quota
                remaining_total -= n_pat

            # 若因取整导致还有剩余名额，从最大的子组中补齐
            for quota_dict, remaining_quota in [(val_quota_per_pattern, remaining_val_quota),
                                                 (test_quota_per_pattern, remaining_test_quota)]:
                if remaining_quota > 0:
                    for pat, pat_indices in sorted_patterns:
                        if remaining_quota <= 0:
                            break
                        used = val_quota_per_pattern.get(pat, 0) + test_quota_per_pattern.get(pat, 0)
                        available = len(pat_indices) - used
                        extra = min(available, remaining_quota)
                        quota_dict[pat] = quota_dict.get(pat, 0) + extra
                        remaining_quota -= extra

            # 执行划分：前 n_val 给验证集，接下来 n_test 给测试集，剩余给训练集
            for pat, pat_indices in sorted_patterns:
                n_val = val_quota_per_pattern.get(pat, 0)
                n_test = test_quota_per_pattern.get(pat, 0)
                val_from_family.extend(pat_indices[:n_val])
                test_from_family.extend(pat_indices[n_val:n_val + n_test])
                train_from_family.extend(pat_indices[n_val + n_test:])

            train_idx_lhs.extend(train_from_family)
            val_idx_lhs.extend(val_from_family)
            test_idx_lhs.extend(test_from_family)

        np.random.shuffle(train_idx_lhs)
        np.random.shuffle(val_idx_lhs)
        np.random.shuffle(test_idx_lhs)

        n_families = len(family_groups)
        print(f"📊 [两级分层划分] 家族数: {n_families} | 训练集: {len(train_idx_lhs)} | 验证集: {len(val_idx_lhs)} | LHS测试集: {len(test_idx_lhs)}")

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
        self.test_dataset_lhs = TensorDataset(X[test_idx_lhs], Params_norm[test_idx_lhs], Y_norm[test_idx_lhs])
        self.test_dataset_real = TensorDataset(X[test_idx_real], Params_norm[test_idx_real], Y_norm[test_idx_real])

        # ======== 4. 基于 Pattern Label 的分层画图样本选取 ========
        self.fixed_samples_for_plot = {}
        patterns = ['Pattern_A', 'Pattern_B', 'Pattern_C']
        
        def find_samples_by_pattern(indices, pattern_kw, n_samples):
            """从 indices 中按 pattern_kw 选出 n_samples 个样本的索引列表"""
            matched = [idx for idx in indices if pattern_kw in str(pattern_labels[idx])]
            if len(matched) == 0:
                matched = indices[:n_samples] if len(indices) >= n_samples else indices
            return matched[:n_samples]

        # A. LHS 训练集: 每个 Pattern 选 2 个
        for p in patterns:
            for i, idx in enumerate(find_samples_by_pattern(train_idx_lhs, p, 2)):
                self.fixed_samples_for_plot[f'LHS_train/{p}_{i}'] = (
                    X[idx:idx+1].clone(), Params_norm[idx:idx+1].clone(), 
                    Y_norm[idx:idx+1].clone(), mutant_names[idx], pattern_labels[idx]
                )

        # B. LHS 验证集: 每个 Pattern 选 2 个
        for p in patterns:
            for i, idx in enumerate(find_samples_by_pattern(val_idx_lhs, p, 2)):
                self.fixed_samples_for_plot[f'LHS_val/{p}_{i}'] = (
                    X[idx:idx+1].clone(), Params_norm[idx:idx+1].clone(), 
                    Y_norm[idx:idx+1].clone(), mutant_names[idx], pattern_labels[idx]
                )

        # C. LHS 测试集: 每个 Pattern 选 2 个
        for p in patterns:
            for i, idx in enumerate(find_samples_by_pattern(test_idx_lhs, p, 2)):
                self.fixed_samples_for_plot[f'LHS_test/{p}_{i}'] = (
                    X[idx:idx+1].clone(), Params_norm[idx:idx+1].clone(), 
                    Y_norm[idx:idx+1].clone(), mutant_names[idx], pattern_labels[idx]
                )

        # D. Real Mutants: 每个 Pattern 选 10 个
        for p in patterns:
            for i, idx in enumerate(find_samples_by_pattern(test_idx_real, p, 10)):
                self.fixed_samples_for_plot[f'real_mutants/{p}_{i}'] = (
                    X[idx:idx+1].clone(), Params_norm[idx:idx+1].clone(), 
                    Y_norm[idx:idx+1].clone(), mutant_names[idx], pattern_labels[idx]
                )

        self._raw_data = raw_data
        self._param_conds = param_conds
        self._mutant_names = mutant_names
        self._Y = Y
        self._train_idx_lhs = train_idx_lhs
        self._val_idx_lhs = val_idx_lhs
        self._test_idx_lhs = test_idx_lhs
        self._test_idx_real = test_idx_real

    def train_dataloader(self):
        return DataLoader(self.train_dataset_lhs, batch_size=self.batch_size_lhs,
                          shuffle=True, drop_last=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return [
            DataLoader(self.val_dataset_lhs, batch_size=self.val_batch_size,
                       shuffle=False, num_workers=self.num_workers, pin_memory=True),
            DataLoader(self.test_dataset_lhs, batch_size=self.val_batch_size,
                       shuffle=False, num_workers=self.num_workers, pin_memory=True),
            DataLoader(self.test_dataset_real, batch_size=self.val_batch_size,
                       shuffle=False, num_workers=self.num_workers, pin_memory=True),
        ]

    @property
    def test_data(self):
        return self._raw_data, self._param_conds, self._mutant_names

    @property
    def test_indices(self):
        return self._test_idx_lhs, self._test_idx_real