"""oracle services extracted from the pre-package implementation."""
from concurrent.futures import ProcessPoolExecutor
from budding_yeast_v2.data.label_metadata import UNKNOWN_BIOLOGY_LABEL
from budding_yeast_v2.data.simulation_guards import bounded_ordered_map
from contextlib import closing
from budding_yeast_v2.data.normalization import normalize_p
from budding_yeast_v2.data.normalization import normalize_y
import numpy as np
import os
from budding_yeast_v2.data.systems.budding.api import simulate_mutant as simulate_mutant_mass
import torch
from budding_yeast_v2.data.systems.budding.api import PARAM_KEYS, VARIABLE_NAMES
from budding_yeast_v2.data.systems.budding.api import get_default_params as get_default_params_mass

_MASS_DEFAULTS = get_default_params_mass()
_MASS_DEFAULTS.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
MASS_SIM_TIMEOUT = 15.0

def solve_single_mass_bridge(name, ovr, *, return_biological_label=False):
    """MASS 版 ODE 仿真桥: 用完整 ovr 字典调用 mass_dataset_generator.simulate_mutant。

    与 solve_single_mutant_bridge (lhs 版) 的关键差异:
    1. 接收完整 ovr 字典 (含 init_Tem1/init_Cdc15/Vi20/kbub2 等 lhs 版不认识的键),
       保证 TEM1-KO / CDC15-KO 等突变的初始条件覆盖与数据集生成时完全一致;
    2. mass 版引擎自带 MAD2-CSL / BUB2-CKO 的特殊处理 (is_mad2_ko / is_bub2_ko);
    3. mass 版引擎自带 mass>10 过滤 (返回 None + "Discarded_Mass");
    4. 补 SIGALRM 硬超时保护 (mass 版引擎无内建超时, 防止 BDF 卡死进程池 worker)。

    Args:
        name: 样本名 (仅用于日志/记录, mass 引擎不查表)
        ovr: sample_mass_mutant 返回的完整参数覆盖字典

    Returns:
        默认返回 (X_dummy, y_uniform, p_vec, pattern_label) 或 None；
        return_biological_label=True 时另附原始生物标签。
    """
    import signal

    def _sigalrm_handler(signum, frame):
        raise TimeoutError("MASS ODE BDF Timeout (SIGALRM)")

    old_handler = signal.signal(signal.SIGALRM, _sigalrm_handler)
    signal.alarm(max(1, int(MASS_SIM_TIMEOUT)))
    try:
        m_name, y_uniform, bio_label, pattern_label = simulate_mutant_mass(
            name, ovr, 210, 500)
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if y_uniform is None:
        return None

    # ovr → 141 维参数向量 (仅取 PARAM_KEYS 中的键, 与数据集 npz 导出逻辑一致)
    p_vec = np.array([ovr.get(k, _MASS_DEFAULTS[k]) for k in PARAM_KEYS],
                     dtype=np.float64)
    X_dummy = np.zeros((39, 2, 500))
    result = (X_dummy, y_uniform, p_vec, pattern_label)
    return (*result, bio_label) if return_biological_label else result


def _solve_single_mass_task_wrapper(task_tuple):
    """顶级函数包装器，供 ProcessPoolExecutor(mp_context='spawn') 进程池调用"""
    name, ovr = task_tuple
    try:
        return solve_single_mass_bridge(name, ovr, return_biological_label=True)
    except Exception:
        return None


def _build_broadcast_from_pool(self, sel_idx, dm, target_valid_num=None):
    """从候选池选样本后构建 broadcast_data (参数候选池版)

    al4pde 机制: 选中参数后 ODE 仿真生成真实轨迹, 再加入训练集。
    1. target_valid_num is None: 传统单批模式 (sel_idx 样本全部送检并保留通过者)
    2. target_valid_num is not None: 配额达标动态缓冲填充模式 (Quota-Guaranteed Dynamic Filling)
       传入优先级降序候选队列 sel_idx, 分块流式仿真, 过滤 Monster / Numerical Failure,
       严格累积有效样本直至填满 target_valid_num 个, 彻底根除样本饥饿。
    """
    sel_idx = np.array(sel_idx, dtype=np.int64)
    pool_ids = self.pool.ensure_ids().copy()

    if hasattr(dm, '_Y') and hasattr(dm, '_test_idx_real'):
        real_envelope = dm._Y[dm._test_idx_real].max(dim=-1)[0].max(dim=0)[0].cpu().numpy()
    else:
        Y_real_phys = dm.denormalize_y(dm.test_dataset_real.tensors[2].cpu())
        real_envelope = Y_real_phys.max(dim=-1)[0].max(dim=0)[0].cpu().numpy()
    dynamic_thresholds = np.maximum(real_envelope * 2.0, 2.0)

    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    workers = int(os.environ.get('AL_SIM_WORKERS', min(8, os.cpu_count() or 4)))

    if target_valid_num is None:
        # 传统单批模式
        n_sel = len(sel_idx)
        sel_P_real = self._pool_P_real[sel_idx].numpy()
        sel_names = self._pool_mutant_names[sel_idx]
        sel_ovrs = [dict(self._pool_ovr[i]) for i in sel_idx] if self._pool_ovr is not None else None

        print(f"[al4pde] ODE 仿真生成 {n_sel} 个选中样本的轨迹 (MASS 引擎, workers={workers})...")
        tasks = []
        for i in range(n_sel):
            ovr = sel_ovrs[i] if sel_ovrs is not None else {k: v for k, v in zip(PARAM_KEYS, sel_P_real[i].tolist())}
            tasks.append((str(sel_names[i]), ovr))

        sim_results = [None] * n_sel
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor, closing(
            bounded_ordered_map(executor, _solve_single_mass_task_wrapper, tasks)
        ) as ordered_results:
            done = 0
            for idx, res in enumerate(ordered_results):
                sim_results[idx] = res
                done += 1
                if done % max(1, n_sel // 10) == 0 or done == n_sel:
                    print(f"  ODE 仿真进度: {done}/{n_sel} ({100*done//n_sel}%)")

        valid_positions = []
        valid_Y_real = []
        valid_P_real = []
        valid_names = []
        valid_labels = []
        valid_biological_labels = []

        for i in range(n_sel):
            res = sim_results[i]
            if res is None:
                continue
            _, y_uniform, p_vec, pattern_label = res[:4]
            lhs_max_vals = np.max(y_uniform, axis=1)
            is_monster = any(lhs_max_vals[v] > dynamic_thresholds[v] for v in range(1, 39))
            if is_monster:
                continue
            valid_positions.append(i)
            valid_Y_real.append(y_uniform)
            valid_P_real.append(p_vec)
            valid_names.append(str(sel_names[i]))
            valid_labels.append(pattern_label)
            valid_biological_labels.append(res[4] if len(res) > 4 else UNKNOWN_BIOLOGY_LABEL)

        evaluated_indices = sel_idx
        if valid_positions:
            valid_sel_idx = sel_idx[valid_positions]
            X_tensor = self._get_pool_X_batch(valid_sel_idx).clone()
        else:
            valid_sel_idx = []

    else:
        # 配额达标动态缓冲填充模式 (Quota-Guaranteed Dynamic Filling)
        target_n = int(min(target_valid_num, self._pool_size))
        chunk_size = 2500
        print(f"[al4pde] 🚀 启动配额达标动态缓冲填充: 目标严格入库 {target_n} 个有效样本 (候选总排序数: {len(sel_idx)}, workers={workers})...")

        valid_Y_real = []
        valid_P_real = []
        valid_names = []
        valid_labels = []
        valid_biological_labels = []
        valid_pool_idx = []
        evaluated_idx_list = []

        cursor = 0
        n_candidates = len(sel_idx)

        while len(valid_Y_real) < target_n and cursor < n_candidates:
            needed = target_n - len(valid_Y_real)
            curr_chunk_size = min(chunk_size, max(int(needed * 1.25), 100))
            end_cursor = min(cursor + curr_chunk_size, n_candidates)
            batch_candidates = sel_idx[cursor:end_cursor]
            cursor = end_cursor

            batch_P_real = self._pool_P_real[batch_candidates].numpy()
            batch_names = self._pool_mutant_names[batch_candidates]
            batch_ovrs = [dict(self._pool_ovr[i]) for i in batch_candidates] if self._pool_ovr is not None else None

            tasks = []
            for i in range(len(batch_candidates)):
                ovr = batch_ovrs[i] if batch_ovrs is not None else {k: v for k, v in zip(PARAM_KEYS, batch_P_real[i].tolist())}
                tasks.append((str(batch_names[i]), ovr))

            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor, closing(
                bounded_ordered_map(executor, _solve_single_mass_task_wrapper, tasks)
            ) as ordered_results:
                for i, res in enumerate(ordered_results):
                    pool_cand_idx = batch_candidates[i]
                    evaluated_idx_list.append(pool_cand_idx)

                    if res is None:
                        continue
                    _, y_uniform, p_vec, pattern_label = res[:4]
                    lhs_max_vals = np.max(y_uniform, axis=1)
                    is_monster = any(lhs_max_vals[v] > dynamic_thresholds[v] for v in range(1, 39))
                    if is_monster:
                        continue

                    valid_Y_real.append(y_uniform)
                    valid_P_real.append(p_vec)
                    valid_names.append(str(batch_names[i]))
                    valid_labels.append(pattern_label)
                    valid_biological_labels.append(res[4] if len(res) > 4 else UNKNOWN_BIOLOGY_LABEL)
                    valid_pool_idx.append(pool_cand_idx)

                    if len(valid_Y_real) == target_n:
                        break

            print(f"  [配额进度] 已收集有效合格样本: {len(valid_Y_real)}/{target_n} | 已消耗候选: {len(evaluated_idx_list)}")

        # 截取到严格 target_n
        if len(valid_Y_real) > target_n:
            valid_Y_real = valid_Y_real[:target_n]
            valid_P_real = valid_P_real[:target_n]
            valid_names = valid_names[:target_n]
            valid_labels = valid_labels[:target_n]
            valid_biological_labels = valid_biological_labels[:target_n]
            valid_pool_idx = valid_pool_idx[:target_n]

        evaluated_indices = np.array(evaluated_idx_list, dtype=np.int64)
        valid_sel_idx = np.array(valid_pool_idx, dtype=np.int64)

        if len(valid_sel_idx) > 0:
            X_tensor = self._get_pool_X_batch(valid_sel_idx).clone()

    # 从候选池安全移除所有已参与仿真的样本 (无论合格与否, 保证不重复测试)
    mask = np.ones(self._pool_size, dtype=bool)
    mask[evaluated_indices] = False
    self.pool.keep(mask)

    if not valid_Y_real:
        print(f"[al4pde] 未能收集到有效样本, 本轮无样本加入训练集")
        print(f"[al4pde] 候选池剩余: {self._pool_size} 样本")
        return None

    # 归一化 Y 和 P (仿真输出为真实物理值空间, 需按照 dm.norm_type 归一化以匹配训练集)
    norm_type = getattr(dm, 'norm_type', 'log1p')
    mean_y = dm.mean_y.cpu()  # (1, 39, 1)
    std_y = dm.std_y.cpu()    # (1, 39, 1)
    p_mean = dm.p_mean.cpu()  # (1, 141)
    p_std = dm.p_std.cpu()    # (1, 141)

    Y_list = []
    for y_real in valid_Y_real:
        y_t = torch.tensor(y_real, dtype=torch.float32)  # (39, 500)
        y_norm = normalize_y(y_t.unsqueeze(0), mean_y, std_y, norm_type,
                             getattr(dm, 's_c', None)).squeeze(0)
        Y_list.append(y_norm)

    P_list = []
    for p_real in valid_P_real:
        p_t = torch.tensor(p_real, dtype=torch.float32).unsqueeze(0)  # (1, 141)
        p_norm = normalize_p(p_t, p_mean, p_std, norm_type).squeeze(0)
        P_list.append(p_norm)

    n_evaluated = len(evaluated_indices)
    print(f"[al4pde] 仿真完成: {len(valid_Y_real)}/{n_evaluated} 通过验证 (通过率: {100*len(valid_Y_real)/max(1, n_evaluated):.1f}%) | "
          f"候选池剩余: {self._pool_size}")

    return {
        'candidate_ids': pool_ids[valid_sel_idx],
        'consumed_ids': pool_ids[evaluated_indices],
        'X': X_tensor,
        'Y': torch.stack(Y_list),
        'P': torch.stack(P_list),
        'Y_real': valid_Y_real,
        'P_real': valid_P_real,
        'names': valid_names,
        'labels': valid_labels,
        'biological_labels': valid_biological_labels,
        'sample_plots': {}
    }


