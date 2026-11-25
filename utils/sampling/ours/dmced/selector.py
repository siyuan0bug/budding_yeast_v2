"""Pillar 2: Self-annealing Epistemic-Risk-Weighted Voronoi (Weighted-LCMD) acquisition."""
import numpy as np
import torch

from ...contracts import SamplingResult
from ...baseline.uncertainty import _power_sample_indices
from .metrics import compute_pool_and_train_views


def round_epistemic_beta(n_train, budget, schedule=(0.5, 1.0, 1.5, 2.0)):
    """Self-annealing exponent resolving the early-coverage / late-refinement phase transition."""
    stage = int(round(float(n_train) / max(float(budget), 1.0))) - 1
    stage = max(0, min(len(schedule) - 1, stage))
    return float(schedule[stage]), stage


@torch.no_grad()
def epistemic_voronoi_select(pool_embed, train_embed, risk_weights, budget, *, beta=1.0, device='cpu'):
    """Minimize sum_i w_i^beta * min_{a in L U S} ||phi(x_i) - phi(a)||^2 via risk-weighted Voronoi LCMD."""
    pool = torch.as_tensor(pool_embed, dtype=torch.float64, device=device)
    train = torch.as_tensor(train_embed, dtype=torch.float64, device=device)
    weights = torch.as_tensor(risk_weights, dtype=torch.float64, device=device).clamp_min(1e-8).pow(float(beta))
    n_pool = len(pool)
    budget = min(int(budget), n_pool)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    if not torch.isfinite(pool).all() or not torch.isfinite(train).all() or not torch.isfinite(weights).all():
        raise ValueError('Epistemic Voronoi selection requires finite embeddings and weights')

    norms = pool.square().sum(dim=1)
    distances = torch.full((n_pool,), float('inf'), dtype=torch.float64, device=device)
    owners = torch.zeros(n_pool, dtype=torch.long, device=device)

    for start in range(0, len(train), 256):
        centers = train[start:start + 256]
        c_norms = centers.square().sum(dim=1)
        for offset in range(0, n_pool, 8192):
            rows = pool[offset:offset + 8192]
            sq = (norms[offset:offset + len(rows), None] + c_norms[None] - 2.0 * (rows @ centers.T)).clamp_min_(0.0)
            val, idx = sq.min(dim=1)
            old = distances[offset:offset + len(rows)]
            better = val < old
            owners[offset:offset + len(rows)][better] = idx[better] + start + 1
            old[better] = val[better]

    selected = []
    used = torch.zeros(n_pool, dtype=torch.bool, device=device)
    n_train = len(train)

    for step in range(budget):
        if n_train == 0 and step == 0:
            scores = weights * norms.clamp_min(1e-8)
        else:
            risk_dist = weights * distances
            risk_dist = torch.where(used, torch.zeros_like(risk_dist), risk_dist)
            mass = torch.bincount(owners, weights=risk_dist, minlength=n_train + step + 1)
            max_mass = mass.max()
            if max_mass > 0:
                in_top_cluster = (mass[owners] == max_mass)
                scores = torch.where(in_top_cluster, risk_dist, torch.full_like(risk_dist, -torch.inf))
            else:
                scores = weights.clone()
        scores = torch.where(used, torch.full_like(scores, -torch.inf), scores)
        index = int(scores.argmax())
        if used[index] or not torch.isfinite(scores[index]):
            index = int(torch.where(~used)[0][0])
        selected.append(index)
        used[index] = True
        sq = (norms + norms[index] - 2.0 * (pool @ pool[index])).clamp_min_(0.0)
        better = sq < distances
        owners[better] = n_train + step + 1
        distances[better] = sq[better]

    return np.asarray(selected, dtype=np.int64)


def select_dmced_pool(context, budget):
    """Track A full method: Dual-scale channel disagreement + Epistemic-Weighted Voronoi in joint space."""
    views = compute_pool_and_train_views(
        context.predictor,
        context.pool,
        context.train.inputs,
        sketch_dim=context.sketch_dim,
        eps=0.01,
        gamma=0.5,
        return_latent=True,
    )
    beta, stage = round_epistemic_beta(len(context.train.inputs.ids), budget)
    indices = epistemic_voronoi_select(
        views['pool_embed'],
        views['train_embed'],
        views['risk'],
        budget,
        beta=beta,
        device=context.device,
    )
    diagnostics = dict(
        method='ours_dmced_pool',
        ensemble_members=len(context.predictor.members),
        round_stage=stage,
        epistemic_beta=beta,
        failure_signature_gamma=0.5,
        risk_median=float(np.median(views['risk'])),
        risk_selected_mean=float(np.mean(views['risk'][indices])),
        risk_pool_mean=float(np.mean(views['risk'])),
        rel_norm_selected_mean=float(np.mean(views['r_rel'][indices])),
        phys_log_selected_mean=float(np.mean(views['r_phys'][indices])),
    )
    return SamplingResult(context.pool.ids[indices], diagnostics=diagnostics)


def select_dmced_power(context, budget):
    """Track A Pillar-1 ablation: Dual-scale channel disagreement risk with Power stochastic sampling (beta=1)."""
    views = compute_pool_and_train_views(
        context.predictor,
        context.pool,
        None,
        sketch_dim=context.sketch_dim,
        eps=0.01,
        return_latent=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(context.seed))
        indices = _power_sample_indices(views['risk'], budget, beta=1.0).numpy()
    diagnostics = dict(
        method='ours_dmced_power',
        ensemble_members=len(context.predictor.members),
        power_beta=1.0,
        risk_median=float(np.median(views['risk'])),
        risk_selected_mean=float(np.mean(views['risk'][indices])),
        risk_pool_mean=float(np.mean(views['risk'])),
        rel_norm_selected_mean=float(np.mean(views['r_rel'][indices])),
        phys_log_selected_mean=float(np.mean(views['r_phys'][indices])),
    )
    return SamplingResult(context.pool.ids[indices], diagnostics=diagnostics)
