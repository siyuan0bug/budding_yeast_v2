"""strategies random services extracted from the pre-package implementation."""
import numpy as np


def _al4pde_random_sample(self, trainer, pl_module, dm, device, current_epoch):
    """al4pde Random: 从候选池随机选 scheduled_num_add 个样本 (ExponentialSchedule)"""
    if self._pool_size == 0:
        print("[al4pde_random] 候选池已空, 跳过本轮 AL")
        return None
    n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
    sel_idx = np.random.choice(self._pool_size, n_select, replace=False)
    print(f"[al4pde_random] 随机选取 {n_select} 个样本 (池剩余: {self._pool_size})")
    return self.make_selection_plan(sel_idx, dm)


