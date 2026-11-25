"""Stochastic Epistemic-Weighted D^p Sampling (Arthur-Vassilvitskii D^2 and tempered D^1)."""
import numpy as np
import torch

from ...contracts import SamplingResult
from .metrics import compute_epistemic_mse_views


@torch.no_grad()
def stochastic_dp_select(pool_embed, train_embed, epistemic_energy, budget, seed, *,
                         dist_power=2.0, device='cpu'):
    """Sample sequentially with p(i) proportional to s_i * (min_{a in L U S} ||phi(x_i) - phi(a)||_2)^dist_power."""
    pool = torch.as_tensor(pool_embed, dtype=torch.float64, device=device)
    train = torch.as_tensor(train_embed, dtype=torch.float64, device=device)
    weights = torch.as_tensor(epistemic_energy, dtype=torch.float64, device=device).clamp_min(1e-12)
    n_pool = len(pool)
    budget = min(int(budget), n_pool)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    if not torch.isfinite(pool).all() or not torch.isfinite(train).all() or not torch.isfinite(weights).all():
        raise ValueError('Stochastic D^p selection requires finite embeddings and weights')

    w_med = weights.median().clamp_min(1e-12)
    weights = weights / w_med

    norms = pool.square().sum(dim=1)
    if len(train) > 0:
        distances = torch.full((n_pool,), float('inf'), dtype=torch.float64, device=device)
        for start in range(0, len(train), 256):
            centers = train[start:start + 256]
            c_norms = centers.square().sum(dim=1)
            for offset in range(0, n_pool, 8192):
                rows = pool[offset:offset + 8192]
                sq = (norms[offset:offset + len(rows), None] + c_norms[None] - 2.0 * (rows @ centers.T)).clamp_min_(0.0)
                val = sq.min(dim=1).values
                distances[offset:offset + len(rows)] = torch.minimum(distances[offset:offset + len(rows)], val)
    else:
        distances = torch.ones(n_pool, dtype=torch.float64, device=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) & 0x7FFFFFFF)
    used = torch.zeros(n_pool, dtype=torch.bool, device=device)
    selected = []
    half_p = float(dist_power) * 0.5

    for _ in range(budget):
        if half_p == 1.0:
            d_factor = distances
        elif half_p == 0.5:
            d_factor = distances.clamp_min(0.0).sqrt()
        else:
            d_factor = distances.clamp_min(0.0).pow(half_p)
        probs = torch.where(used, torch.zeros_like(weights), weights * d_factor)
        total = probs.sum()
        if not torch.isfinite(total) or total <= 0:
            probs = torch.where(used, torch.zeros_like(weights), weights)
            total = probs.sum()
        if not torch.isfinite(total) or total <= 0:
            avail = torch.where(~used)[0]
            index = int(avail[int(torch.randint(len(avail), (1,), generator=gen, device=device).item())].item())
        else:
            probs = probs / total
            index = int(torch.multinomial(probs, 1, generator=gen).item())
            if used[index]:
                avail = torch.where(~used)[0]
                index = int(avail[0].item())
        selected.append(index)
        used[index] = True
        sq = (norms + norms[index] - 2.0 * (pool @ pool[index])).clamp_min_(0.0)
        distances = torch.minimum(distances, sq)

    return np.asarray(selected, dtype=np.int64)


def _select_pool(context, budget, *, method_name, dist_power):
    views = compute_epistemic_mse_views(
        context.predictor,
        context.pool,
        context.train.inputs,
        sketch_dim=context.sketch_dim,
        gamma=0.5,
        return_latent=True,
    )
    indices = stochastic_dp_select(
        views['pool_embed'],
        views['train_embed'],
        views['energy'],
        budget,
        seed=context.seed,
        dist_power=dist_power,
        device=context.device,
    )
    energy = views['energy']
    diagnostics = dict(
        method=method_name,
        ensemble_members=len(context.predictor.members),
        dist_power=float(dist_power),
        failure_signature_gamma=0.5,
        energy_median=float(np.median(energy)),
        energy_p99_over_p50=float(np.quantile(energy, 0.99) / max(np.median(energy), 1e-12)),
        energy_selected_mean=float(np.mean(energy[indices])),
        energy_pool_mean=float(np.mean(energy)),
    )
    return SamplingResult(context.pool.ids[indices], diagnostics=diagnostics)


def select_epistemic_d2_pool(context, budget):
    """Track A main method: Power-exact MSE energy + Stochastic Weighted D^2 in joint space."""
    return _select_pool(context, budget, method_name='ours_epistemic_d2_pool', dist_power=2.0)


def select_epistemic_d1_pool(context, budget):
    """Track A tempered variant: Power-exact MSE energy + Stochastic Weighted D^1 in joint space."""
    return _select_pool(context, budget, method_name='ours_epistemic_d1_pool', dist_power=1.0)


def select_epistemic_d05_pool(context, budget):
    """Track A mild variant: Power-exact MSE energy + Stochastic Weighted D^0.5 in joint space."""
    return _select_pool(context, budget, method_name='ours_epistemic_d05_pool', dist_power=0.5)
