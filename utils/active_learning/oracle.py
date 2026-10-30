"""oracle services extracted from the pre-package implementation."""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
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
from budding_yeast_v2.utils.runtime_logging import log_stage, ProgressLog, log_event

_MASS_DEFAULTS = get_default_params_mass()
_MASS_DEFAULTS.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
MASS_SIM_TIMEOUT = 15.0

def solve_single_mass_bridge(name, ovr, *, return_biological_label=False):
    """Call the canonical modular Budding solver with complete overrides.

    Initial-state overrides and MAD2/BUB2 event handling are preserved. Finite
    high-amplitude trajectories are retained; solver deadlines and numerical
    validity checks still apply.

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

    if (y_uniform is None or np.asarray(y_uniform).shape != (39, 500)
            or not np.isfinite(y_uniform).all()):
        return None

    # ovr → 141 维参数向量 (仅取 PARAM_KEYS 中的键, 与数据集 npz 导出逻辑一致)
    p_vec = np.array([ovr.get(k, _MASS_DEFAULTS[k]) for k in PARAM_KEYS],
                     dtype=np.float64)
    if not np.isfinite(p_vec).all():
        return None
    X_dummy = np.zeros((39, 2, 500))
    result = (X_dummy, y_uniform, p_vec, pattern_label)
    return (*result, bio_label) if return_biological_label else result


def _solve_single_mass_task_wrapper(task_tuple):
    """顶级函数包装器，供 ProcessPoolExecutor(mp_context='spawn') 进程池调用"""
    name, ovr = task_tuple
    try:
        result = solve_single_mass_bridge(name, ovr, return_biological_label=True)
        if result is None:
            return None
        # The dummy input is never consumed by the oracle. Returning 30k copies
        # used >9 GB of IPC/parent RAM. Match the existing float32 NPZ and model
        # trajectory contract; solver, actual IC and physical parameters stay 64.
        return (None, np.asarray(result[1], dtype=np.float32), *result[2:])
    except Exception:
        return None


def _valid_simulation_result(result):
    """Shared single-batch/refill validity gate, independent of amplitude or BIO."""
    if result is None:
        return False
    try:
        y, p = np.asarray(result[1]), np.asarray(result[2])
        return (y.shape == (39, 500) and p.shape == (len(PARAM_KEYS),)
                and np.isfinite(y).all() and np.isfinite(p).all())
    except (TypeError, ValueError, IndexError):
        return False


def _simulation_results(tasks,workers,context,*,stall_timeout=35.,retries=1):
    """Rebuild a failed pool once, without dropping/reordering unknown inputs.

    Already-yielded candidates are never rerun. A repeated infrastructure failure
    aborts explicitly; no candidate is silently called biologically inviable.
    """
    cursor=attempt=0
    while cursor<len(tasks):
        try:
            with ProcessPoolExecutor(max_workers=workers,mp_context=context) as executor, closing(
                bounded_ordered_map(executor,_solve_single_mass_task_wrapper,tasks[cursor:],
                                    stall_timeout=stall_timeout)) as ordered:
                for result in ordered:
                    cursor+=1
                    yield result
            return
        except (TimeoutError,BrokenProcessPool) as error:
            log_event('RETRY' if attempt<retries else 'FAIL','ODE pool rebuild',
                completed=cursor,total=len(tasks),attempt=attempt+1,error=repr(error))
            if attempt>=retries:raise
            attempt+=1


def _build_broadcast_from_pool(self, sel_idx, dm, target_valid_num=None):
    """从候选池选样本后构建 broadcast_data (参数候选池版)

    al4pde 机制: 选中参数后 ODE 仿真生成真实轨迹, 再加入训练集。
    1. target_valid_num is None: 传统单批模式 (sel_idx 样本全部送检并保留通过者)
    2. target_valid_num is not None: 配额达标动态缓冲填充模式 (Quota-Guaranteed Dynamic Filling)
       传入优先级降序候选队列 sel_idx, 分块流式仿真, 排除数值失败,
       严格累积有效样本直至填满 target_valid_num 个, 彻底根除样本饥饿。
    """
    sel_idx = np.array(sel_idx, dtype=np.int64)
    pool_ids = self.pool.ensure_ids().copy()
    pool_levels = [int(row['level']) for row in self._pool_mass_meta]

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
        progress=ProgressLog('ODE simulation',n_sel,workers=workers)
        with log_stage('ODE process pool',candidates=n_sel,workers=workers), closing(
            _simulation_results(tasks,workers,ctx)) as ordered_results:
            done = 0
            for idx, res in enumerate(ordered_results):
                sim_results[idx] = res
                done += 1
                progress.update(done)
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
            if not _valid_simulation_result(res):
                continue
            _, y_uniform, p_vec, pattern_label = res[:4]
            valid_positions.append(i)
            valid_Y_real.append(y_uniform)
            valid_P_real.append(p_vec)
            valid_names.append(str(sel_names[i]))
            valid_labels.append(pattern_label)
            valid_biological_labels.append(res[4] if len(res) > 4 else UNKNOWN_BIOLOGY_LABEL)

        evaluated_indices = sel_idx
        del sim_results
        if valid_positions:
            valid_sel_idx = sel_idx[valid_positions]
            X_tensor = self._get_pool_X_batch(valid_sel_idx)
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

            progress=ProgressLog('ODE refill chunk',len(batch_candidates),workers=workers)
            with log_stage('ODE refill process pool',candidates=len(batch_candidates),workers=workers), closing(
                _simulation_results(tasks,workers,ctx)) as ordered_results:
                for i, res in enumerate(ordered_results):
                    pool_cand_idx = batch_candidates[i]
                    evaluated_idx_list.append(pool_cand_idx)
                    progress.update(i+1)

                    if not _valid_simulation_result(res):
                        continue
                    _, y_uniform, p_vec, pattern_label = res[:4]

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
            X_tensor = self._get_pool_X_batch(valid_sel_idx)

    # 从候选池安全移除所有已参与仿真的样本 (无论合格与否, 保证不重复测试)
    from .io import pool_provenance
    provenance = pool_provenance(self, valid_sel_idx) if len(valid_sel_idx) else []
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

    # Allocate once instead of retaining N normalized tensors and another stack.
    normalized_y = torch.empty((len(valid_Y_real), 39, 500), dtype=torch.float32)
    for start in range(0, len(valid_Y_real), 256):
        block = torch.from_numpy(np.asarray(valid_Y_real[start:start+256], dtype=np.float32))
        normalized_y[start:start+len(block)] = normalize_y(block, mean_y, std_y, norm_type,
                                                         getattr(dm, 's_c', None))
    normalized_p = normalize_p(torch.as_tensor(np.asarray(valid_P_real), dtype=torch.float32),
                               p_mean, p_std, norm_type)

    n_evaluated = len(evaluated_indices)
    print(f"[al4pde] 仿真完成: {len(valid_Y_real)}/{n_evaluated} 通过验证 (通过率: {100*len(valid_Y_real)/max(1, n_evaluated):.1f}%) | "
          f"候选池剩余: {self._pool_size}")

    return {
        'candidate_ids': pool_ids[valid_sel_idx],
        'consumed_ids': pool_ids[evaluated_indices],
        'levels': [pool_levels[i] for i in valid_sel_idx],
        'requested_levels': [pool_levels[i] for i in valid_sel_idx],
        'module_counts': [pool_levels[i] for i in valid_sel_idx],
        'X': X_tensor,
        'Y': normalized_y,
        'P': normalized_p,
        'Y_real': valid_Y_real,
        'P_real': valid_P_real,
        'names': valid_names,
        'labels': valid_labels,
        'biological_labels': valid_biological_labels,
        'provenance': provenance,
        'sample_plots': {}
    }


