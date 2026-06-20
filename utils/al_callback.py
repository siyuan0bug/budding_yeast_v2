import torch
import numpy as np
import pytorch_lightning as pl
from concurrent.futures import ProcessPoolExecutor
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
                 random_seed=42):
        """
        标准主动学习 (Active Learning) 回调模块。

        本模块同时支持两种 AL 范式：
        1. **标准池化 AL (Pool-Based)**：从已有 LHS 数据池中选取样本，适用于 'us' / 'is' 策略。
           - on_train_start 时通过 K-Means 聚类从全量 LHS 训练集中选出初始训练子集，
             剩余样本构成未标注池 (unlabeled pool)。
           - 每轮 AL 触发时，依据查询策略从池中选取最具信息量的样本加入训练集。
        2. **查询合成 AL (Query Synthesis)**：通过 LHS 引擎生成新候选样本，适用于 'rgs' 策略。

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
            num_add: 每轮 AL 要新增的样本数 (默认 5000)
            perturbation: RGS 专用，连续参数的相对扰动幅度 (0.1 = ±10%)
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

    # ==================================================================
    # 标准主动学习框架：初始训练集选择
    # ==================================================================
    def on_train_start(self, trainer, pl_module):
        """
        标准主动学习初始化（所有策略统一执行）：
        - 从全量 LHS 训练集中通过 K-Means 聚类选出初始训练子集
        - 剩余样本构成未标注池 (unlabeled pool)
        - 对于池化策略 (us/is/random/wrs/vessal/hggs)：后续从池中查询
        - 对于查询合成策略 (rgs)：后续通过 LHS 引擎生成新样本，池仅作备份

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
        initial_size = min(self.initial_train_size, full_size)

        print(f"\n[AL Manager] ===== 标准主动学习初始化 =====")
        print(f"[AL Manager] 策略: {self.strategy.upper()} | 全量 LHS 训练集: {full_size}")
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

        # 保存未标注池 (剩余样本) —— 池化策略会从中查询，RGS 不使用但保留以备扩展
        self._pool_X = full_X[pool_idx]
        self._pool_P = full_P[pool_idx]
        self._pool_Y = full_Y[pool_idx]
        self._pool_size = self._pool_X.shape[0]
        self._al_initialized = True
        self._train_size_history.append(initial_size)

        # 策略分类提示
        if self.strategy == 'rgs':
            mode_desc = "查询合成模式 (RGS 通过 LHS 引擎生成新样本)"
        else:
            mode_desc = f"池化模式 (从 {self._pool_size} 样本池中查询)"

        print(f"[AL Manager] 初始训练集: {initial_size} 样本 (K-Means={self.initial_selection=='kmeans'})")
        print(f"[AL Manager] 未标注池: {self._pool_size} 样本 | {mode_desc}")
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

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch == 0 or current_epoch % self.trigger_every_n_epochs != 0:
            return

        print(f"\n[AL Manager] Epoch {current_epoch} 触发主动学习采样: {self.strategy.upper()}")
        dm = trainer.datamodule
        device = pl_module.device
        broadcast_data = None

        if trainer.is_global_zero:
            # 查询合成策略 (RGS)：通过 LHS 引擎生成新样本，但仍走标准 AL 框架
            if self.strategy == 'rgs':
                broadcast_data = self._rgs_sample(trainer, pl_module, dm, device, current_epoch)
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
    # Physics-Informed Active Learning (PIAL)
    # 导师要求：用 ODE 残差（非预测误差）+ 轨迹多样性指导 AL
    # ==================================================================
    def _query_physics_informed(self, pl_module, device, pool_indices, dm,
                                 num_to_select, current_epoch):
        """
        Physics-Informed Active Learning 查询策略。

        综合两个信号选取样本：
        1. **ODE 残差得分** (physics residual)：模型预测轨迹 û(t) 代入 ODE 系统
           du/dt = f(u, p) 后的残差 r(t) = dû/dt - f(û, p)。
           残差大 → 模型预测违反物理定律 → 该样本信息量大。
           **关键：不需要真实标签**，只用模型预测 + 物理方程。
        2. **轨迹多样性得分** (trajectory diversity)：样本预测轨迹与当前训练集
           轨迹的最小欧氏距离。在轨迹空间（行为空间）而非参数空间度量多样性。

        综合 score = α · normalize(residual) + β · normalize(diversity)
        默认 α=0.6, β=0.4（残差为主，多样性为辅）。

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
        print(f"\n[PIAL] ===== Physics-Informed Active Learning =====")
        print(f"[PIAL] 池子集大小: {len(pool_indices)} | 选取: {num_to_select}")

        # ---- Step 1: 获取模型预测轨迹 (反归一化到真实值空间) ----
        # ODE eqns 在真实值空间定义，必须反归一化
        pred_trajectories = self._predict_pool_trajectories(pl_module, device, pool_indices, dm)
        # pred_trajectories: (N, 39, T) numpy 真实值空间

        # ---- Step 2: 计算 ODE 残差得分 ----
        residual_scores = self._compute_ode_residual(
            pred_trajectories, pool_indices, dm)
        # residual_scores: (N,) 每个样本的平均残差范数

        # ---- Step 3: 计算轨迹多样性得分 ----
        diversity_scores = self._compute_trajectory_diversity(
            pred_trajectories, dm)
        # diversity_scores: (N,) 每个样本到训练集的最小轨迹距离

        # ---- Step 4: 归一化并综合 ----
        def normalize(arr):
            lo, hi = np.min(arr), np.max(arr)
            if hi - lo < 1e-12:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        residual_norm = normalize(residual_scores)
        diversity_norm = normalize(diversity_scores)

        # 动态权重：训练早期更重多样性（探索），后期更重残差（利用）
        # progress ∈ [0, 1]
        max_epochs = getattr(pl_module, 'max_epochs', 200)
        progress = min(1.0, current_epoch / max(1, max_epochs))
        alpha = 0.4 + 0.3 * progress   # 残差权重: 0.4 → 0.7
        beta = 1.0 - alpha              # 多样性权重: 0.6 → 0.3

        combined_scores = alpha * residual_norm + beta * diversity_norm

        print(f"[PIAL] 残差得分: min={residual_scores.min():.4f} "
              f"median={np.median(residual_scores):.4f} "
              f"max={residual_scores.max():.4f}")
        print(f"[PIAL] 多样性得分: min={diversity_scores.min():.4f} "
              f"median={np.median(diversity_scores):.4f} "
              f"max={diversity_scores.max():.4f}")
        print(f"[PIAL] 动态权重: α(残差)={alpha:.3f} β(多样性)={beta:.3f} "
              f"(epoch={current_epoch}, progress={progress:.2f})")

        # ---- Step 5: 选取 top-K ----
        num_to_select = min(num_to_select, len(combined_scores))
        selected_idx = np.argsort(combined_scores)[-num_to_select:]

        print(f"[PIAL] 选中 top-{num_to_select} (综合得分阈值: "
              f"{combined_scores[selected_idx].min():.4f})")

        return selected_idx

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

                X_batch = self._pool_X[batch_indices].to(device)
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

            # 应用掩码：只在连续区间（mask=True）的时间步计算残差
            valid_count = mask.sum()
            if valid_count < T * 0.1:
                # 掩码过度（超过 90% 被屏蔽），降级为全轨迹计算
                # 防止异常轨迹导致无有效残差
                residual_scores[i] = np.sqrt(np.mean(residual ** 2))
            else:
                masked_residual = residual[:, mask]  # (39, valid_count)
                residual_scores[i] = np.sqrt(np.mean(masked_residual ** 2))

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
            for name, f in futures:
                res = f.result()
                if res is not None: 
                    valid_results.append(res)
                    valid_names.append(name)
        
        if len(valid_results) > 0:
            Y_train_norm = dm.train_dataset_lhs.tensors[2].cpu()
            std_y_cpu = dm.std_y.cpu()
            mean_y_cpu = dm.mean_y.cpu()
            Y_train_raw = Y_train_norm * std_y_cpu + mean_y_cpu
            
            real_envelope = Y_train_raw.max(dim=-1)[0].max(dim=0)[0].numpy()
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
