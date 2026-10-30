"""strategies diversity services extracted from the pre-package implementation."""


def _al4pde_max_dist_sample(self, trainer, pl_module, dm, device, current_epoch):
    """AL4PDE output-trajectory MaxDist; separate from custom shape diversity."""
    if self._pool_size == 0:
        return None
    pool_features, train_features = self._compute_predicted_distance_features(pl_module,dm,device)
    selected = self._bmdal_predicted_max_dist_select(pool_features,train_features,
        self._al4pde_scheduled_num_add(),device)
    return self.make_selection_plan(selected,dm)



def _al4pde_lcmd_sample(self, trainer, pl_module, dm, device, current_epoch):
    if self._pool_size == 0:
        return None
    pool_features, train_features = self._compute_predicted_distance_features(pl_module, dm, device)
    selected = self._bmdal_lcmd_select(pool_features, train_features,
                                      self._al4pde_scheduled_num_add(), device)
    return self.make_selection_plan(selected, dm)


def _al4pde_bait_sample(self, trainer, pl_module, dm, device, current_epoch):
    if self._pool_size == 0:
        return None
    pool_features, train_features = self._compute_bait_features(pl_module, dm, device)
    selected = self._bmdal_bait_select(pool_features, train_features,
                                      self._al4pde_scheduled_num_add(), device)
    return self.make_selection_plan(selected, dm)


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
    return self.make_selection_plan(sel_idx, dm)


