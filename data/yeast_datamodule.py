import math
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

        # ================================================================
        #  双级分层 + 孤本保护 数据划分机制
        #  步骤 2: 第一级 — 按母体家族（Family）分组
        #  步骤 3: 第二级 — 按动态表型（Pattern）子组分组
        #  步骤 4: 比例抽样 + 孤本保护（每个 Pattern 子组内独立计算）
        # ================================================================
        np.random.seed(self.seed)

        # 第一级：按突变体家族分组（提取 _LHS_ 前的前缀）
        family_groups = {}
        for idx in lhs_indices:
            name = str(mutant_names[idx])
            family = name.split('_LHS_')[0] if '_LHS_' in name else name
            family_groups.setdefault(family, []).append(idx)

        train_idx_lhs = []
        val_idx_lhs = []
        test_idx_lhs = []

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

                val_idx_lhs.extend(val_from_pat)
                test_idx_lhs.extend(test_from_pat)
                train_idx_lhs.extend(train_from_pat)

                print(f"   [Family={family}] {pat}: total={n_pat:>5d}  "
                      f"→ train={len(train_from_pat):>5d}  val={len(val_from_pat):>3d}  test={len(test_from_pat):>3d}")

        # 最终整体洗牌
        np.random.shuffle(train_idx_lhs)
        np.random.shuffle(val_idx_lhs)
        np.random.shuffle(test_idx_lhs)

        test_idx_real = real_indices

        n_families = len(family_groups)
        print(f"📊 [两级分层+孤本保护] 家族数: {n_families} | "
              f"训练集: {len(train_idx_lhs)} | 验证集: {len(val_idx_lhs)} | "
              f"LHS测试集: {len(test_idx_lhs)} | 真实测试集: {len(test_idx_real)}")

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