"""Track A fixed-pool CEBS selectors: order-stratified Power ablation & Virtual Epistasis-Amplified Power."""
from collections import Counter
import numpy as np
import torch

from ...baseline.uncertainty import _power_sample_indices
from ...contracts import SamplingResult
from .metrics import allocate_order_budgets, compute_cebs_scores, epistasis_amplified_scores


def order_stratified_power_select(scores, orders, budget, seed, *, beta=1.0):
    """Sample exactly native-proportion budgets per order k using Power (Gumbel-top-k) within each order."""
    scores = np.maximum(np.asarray(scores, dtype=np.float64), 1e-12)
    orders = np.asarray(orders, dtype=np.int64)
    alloc = allocate_order_budgets(orders, budget)
    chosen = []
    for k in sorted(alloc.keys()):
        b_k = int(alloc[k])
        if b_k <= 0:
            continue
        idx_k = np.flatnonzero(orders == k)
        if b_k >= len(idx_k):
            chosen.append(idx_k)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((int(seed) + 10007 * int(k)) & 0x7FFFFFFF)
                local = _power_sample_indices(scores[idx_k], b_k, beta=float(beta))
            chosen.append(idx_k[np.asarray(local, dtype=np.int64)])
    indices = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    return indices, alloc


def _select_cebs_pool(context, budget, *, method_name, compute_epistasis, lam, beta=1.0):
    views = compute_cebs_scores(
        context.predictor,
        context.pool,
        compute_epistasis=compute_epistasis,
    )
    scores, epi_medians = epistasis_amplified_scores(
        views['u_power'],
        views['epi_mag'],
        views['orders'],
        lam=lam,
    )
    indices, alloc = order_stratified_power_select(
        scores,
        views['orders'],
        budget,
        seed=context.seed,
        beta=beta,
    )
    pool_orders = Counter(int(k) for k in views['orders'])
    sel_orders = Counter(int(k) for k in views['orders'][indices])
    diagnostics = dict(
        method=method_name,
        ensemble_members=len(context.predictor.members),
        compute_epistasis=bool(compute_epistasis),
        epistasis_lambda=float(lam),
        power_beta=float(beta),
        pool_orders=dict(sorted(pool_orders.items())),
        order_budgets=dict(sorted(alloc.items())),
        selected_orders=dict(sorted(sel_orders.items())),
        epistasis_median_by_order={str(k): float(v) for k, v in sorted(epi_medians.items())},
        u_power_selected_mean=float(np.mean(views['u_power'][indices])),
        u_power_pool_mean=float(np.mean(views['u_power'])),
        epi_mag_selected_mean=float(np.mean(views['epi_mag'][indices])),
        epi_mag_pool_mean=float(np.mean(views['epi_mag'])),
    )
    return SamplingResult(context.pool.ids[indices], diagnostics=diagnostics)


def select_order_power_pool(context, budget):
    """Track A clean ablation: Module 1 (native compositional order marginal preservation) + pure Power (lam=0)."""
    return _select_cebs_pool(
        context,
        budget,
        method_name='ours_order_power_pool',
        compute_epistasis=False,
        lam=0.0,
        beta=1.0,
    )


def select_epistasis_pool(context, budget):
    """Track A main CEBS method: Module 1 (order preservation) + Module 2 (0-ODE virtual epistasis gain, lam=1.0)."""
    return _select_cebs_pool(
        context,
        budget,
        method_name='ours_epistasis_pool',
        compute_epistasis=True,
        lam=1.0,
        beta=1.0,
    )
