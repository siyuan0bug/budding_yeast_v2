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
        dataset_path='mydata/mass_100000_dataset_210min_500steps_dual_labels.npz',
        adj_matrix_path='mydata/yeast_signed_adjacency_matrix.npy',
        batch_size_synth=32,
        batch_size_real=8,
        val_batch_size=32,
        num_workers=4,
        seed=42,
        use_adj_matrix=True,
        norm_type='asinh_z',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_path = dataset_path
        self.adj_matrix_path = adj_matrix_path
        self.batch_size_synth = batch_size_synth
        self.batch_size_real = batch_size_real
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.use_adj_matrix = use_adj_matrix
        self.norm_type = norm_type
        
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

        # ================================================================
        #  数据集类型通用智能识别:
        #  - LHS 数据集: 合成样本名含 '_LHS_' (按母体家族两级分层划分)。
        #  - 非 LHS 数据集 (MASS / WB-MASS / 任意未来自定义生成数据集):
        #    自动按真实突变体特征 (如 1_WT_Glc, 15_cln3Δ...) 区分真实测试集与合成样本，
        #    跳过家族分层, 仅按 Pattern 宏观划分。两类数据集都严格将 real mutants
        #    全部划入测试集 (test_real), 避免数据泄露; 归一化统计量只用合成样本。
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

        # 归一化统计量计算：支持 asinh_z (推荐最优: 参数统一 asinh(Z) + 变量稳健 log1p), asinh_scale, log1p, 与 linear
        self.norm_type = getattr(self, 'norm_type', 'asinh_z')
        if self.norm_type == 'asinh_z':
            # 1. 变量变换: 稳健 log1p + 通道 Z-score
            # 10万样本全量评测证实 log1p 相对扰动放大率仅 4.71% (远低于 asinh 的 9.39%)，且完全杜绝 sinh 双向指数发散风险
            Y_trans = torch.log1p(torch.clamp(Y, min=0.0))
            self.mean_y = Y_trans[synth_indices].mean(dim=(0, 2), keepdim=True)
            self.std_y = Y_trans[synth_indices].std(dim=(0, 2), keepdim=True) + 1e-6
            Y_norm = (Y_trans - self.mean_y) / self.std_y
            Y_norm = torch.nan_to_num(Y_norm, nan=0.0, posinf=10.0, neginf=-10.0)

            # 2. 参数变换: 统一双曲反正弦标准化 asinh(Z)
            # 消除硬零点对数奇点与多突变模长爆炸问题，最大模长从 76.11 压缩到 12.99
            self.p_mean = Params[synth_indices].mean(dim=0, keepdim=True)
            self.p_std = Params[synth_indices].std(dim=0, keepdim=True)
            is_varying = self.p_std > 1e-5
            p_std_safe = torch.where(is_varying, self.p_std, torch.ones_like(self.p_std))
            z = (Params - self.p_mean) / p_std_safe
            Params_norm = torch.where(is_varying, torch.asinh(z), torch.zeros_like(Params))
            Params_norm = torch.nan_to_num(Params_norm, nan=0.0, posinf=10.0, neginf=-10.0)
        elif self.norm_type == 'asinh_scale':
            # 1. 变量变换: s_c 为合成样本每个通道的 90% 分位数 (下限 0.01)
            y_synth_np = Y[synth_indices].numpy()
            sub_n = min(len(synth_indices), 20000)
            s_c_np = np.quantile(np.maximum(y_synth_np[:sub_n], 0.0), 0.90, axis=(0, 2), keepdims=True)
            s_c_np = np.maximum(s_c_np, 0.01)
            self.s_c = torch.tensor(s_c_np, dtype=torch.float32, device=Y.device)
            Y_trans = torch.asinh(torch.clamp(Y, min=0.0) / (2.0 * self.s_c))

            self.mean_y = Y_trans[synth_indices].mean(dim=(0, 2), keepdim=True)
            self.std_y = Y_trans[synth_indices].std(dim=(0, 2), keepdim=True) + 1e-6
            Y_norm = (Y_trans - self.mean_y) / self.std_y
            Y_norm = torch.nan_to_num(Y_norm, nan=0.0, posinf=10.0, neginf=-10.0)

            # 2. 参数变换: 统一 log10 变换并设定敲除生理截断下界 1e-8
            P_trans = torch.log10(torch.clamp(Params, min=1e-8))
            self.p_mean = P_trans[synth_indices].mean(dim=0, keepdim=True)
            self.p_std = P_trans[synth_indices].std(dim=0, keepdim=True)

            # 非突变参数 (std < 1e-5) 严格赋值为恒定 0.0; 可变突变参数执行标准 Z-score
            is_varying = self.p_std > 1e-5
            p_std_safe = torch.where(is_varying, self.p_std, torch.ones_like(self.p_std))
            Params_norm = torch.where(is_varying, (P_trans - self.p_mean) / p_std_safe, torch.zeros_like(P_trans))
            Params_norm = torch.nan_to_num(Params_norm, nan=0.0, posinf=10.0, neginf=-10.0)
        elif self.norm_type == 'log1p':
            Y_trans = torch.log1p(torch.clamp(Y, min=0.0))
            self.mean_y = Y_trans[synth_indices].mean(dim=(0, 2), keepdim=True)
            self.std_y = Y_trans[synth_indices].std(dim=(0, 2), keepdim=True) + 1e-6
            Y_norm = (Y_trans - self.mean_y) / self.std_y
            Y_norm = torch.nan_to_num(Y_norm, nan=0.0, posinf=10.0, neginf=-10.0)

            self.p_mean = Params[synth_indices].mean(dim=0, keepdim=True)
            self.p_std = Params[synth_indices].std(dim=0, keepdim=True) + 1e-6
            Params_norm = (Params - self.p_mean) / self.p_std
            Params_norm = torch.nan_to_num(Params_norm, nan=0.0, posinf=10.0, neginf=-10.0)
        else:
            Y_trans = Y
            self.mean_y = Y_trans[synth_indices].mean(dim=(0, 2), keepdim=True)
            self.std_y = Y_trans[synth_indices].std(dim=(0, 2), keepdim=True) + 1e-6
            Y_norm = (Y_trans - self.mean_y) / self.std_y
            Y_norm = torch.nan_to_num(Y_norm, nan=0.0, posinf=10.0, neginf=-10.0)

            self.p_mean = Params[synth_indices].mean(dim=0, keepdim=True)
            self.p_std = Params[synth_indices].std(dim=0, keepdim=True) + 1e-6
            Params_norm = (Params - self.p_mean) / self.p_std
            Params_norm = torch.nan_to_num(Params_norm, nan=0.0, posinf=10.0, neginf=-10.0)

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

        # ======== 4. 基于 Pattern Label 的分层画图样本选取 ========
        self.fixed_samples_for_plot = {}
        patterns = ['Pattern_A', 'Pattern_B', 'Pattern_C']
        
        def find_samples_by_pattern(indices, pattern_kw, n_samples):
            """从 indices 中按 pattern_kw 选出 n_samples 个样本的索引列表"""
            matched = [idx for idx in indices if pattern_kw in str(pattern_labels[idx])]
            if len(matched) == 0:
                matched = indices[:n_samples] if len(indices) >= n_samples else indices
            return matched[:n_samples]

        # A. 合成训练集: 每个 Pattern 选 2 个
        for p in patterns:
            for i, idx in enumerate(find_samples_by_pattern(train_idx_synth, p, 2)):
                self.fixed_samples_for_plot[f'synth_train/{p}_{i}'] = (
                    X[idx:idx+1].clone(), Params_norm[idx:idx+1].clone(), 
                    Y_norm[idx:idx+1].clone(), mutant_names[idx], pattern_labels[idx]
                )

        # B. 合成验证集: 每个 Pattern 选 2 个
        for p in patterns:
            for i, idx in enumerate(find_samples_by_pattern(val_idx_synth, p, 2)):
                self.fixed_samples_for_plot[f'synth_val/{p}_{i}'] = (
                    X[idx:idx+1].clone(), Params_norm[idx:idx+1].clone(), 
                    Y_norm[idx:idx+1].clone(), mutant_names[idx], pattern_labels[idx]
                )

        # C. 合成测试集: 每个 Pattern 选 2 个
        for p in patterns:
            for i, idx in enumerate(find_samples_by_pattern(test_idx_synth, p, 2)):
                self.fixed_samples_for_plot[f'synth_test/{p}_{i}'] = (
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
        self._train_idx_synth = train_idx_synth
        self._val_idx_synth = val_idx_synth
        self._test_idx_synth = test_idx_synth
        self._test_idx_real = test_idx_real
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
        """
        Denormalizes y_norm back to physical concentration space (B, V, T).
        Supports both 'log1p' and 'linear' normalization modes.
        """
        if device is not None:
            mean_y = self.mean_y.to(device)
            std_y = self.std_y.to(device)
        else:
            mean_y = self.mean_y
            std_y = self.std_y

        y_shifted = y_norm * std_y + mean_y
        norm_type = getattr(self, 'norm_type', 'asinh_z')
        if norm_type == 'asinh_scale':
            s_c = self.s_c.to(device) if device is not None and hasattr(self, 's_c') and self.s_c is not None else getattr(self, 's_c', None)
            if s_c is None:
                s_c = 1.0
            y_clamped = torch.clamp(y_shifted, min=-10.0, max=10.0)
            y_real = torch.clamp(2.0 * s_c * torch.sinh(y_clamped), min=0.0)
            return y_real
        elif norm_type in ('log1p', 'asinh_z'):
            # Numerical safeguard against extreme logits before expm1
            y_clamped = torch.clamp(y_shifted, min=-10.0, max=15.0)
            y_real = torch.clamp(torch.expm1(y_clamped), min=0.0)
            return y_real
        else:
            return y_shifted

    def denormalize_p(self, p_norm, device=None):
        """
        Denormalizes p_norm back to physical parameter space.
        """
        if device is not None:
            p_mean = self.p_mean.to(device)
            p_std = self.p_std.to(device)
        else:
            p_mean = self.p_mean
            p_std = self.p_std

        norm_type = getattr(self, 'norm_type', 'asinh_z')
        if norm_type == 'asinh_scale':
            is_varying = (p_std > 1e-5)
            p_trans = torch.where(is_varying, p_norm * p_std + p_mean, p_mean)
            p_real = torch.clamp(10.0 ** torch.clamp(p_trans, min=-8.0, max=6.0), min=0.0)
            return p_real
        elif norm_type == 'asinh_z':
            is_varying = (p_std > 1e-5)
            p_std_safe = torch.where(is_varying, p_std, torch.ones_like(p_std))
            p_real = torch.where(is_varying, torch.sinh(p_norm) * p_std_safe + p_mean, p_mean)
            return torch.clamp(p_real, min=0.0)
        else:
            return p_norm * p_std + p_mean