"""Pillar 2: Stochastic Epistemic-Weighted D^2 Sampling (Weighted K-Means++)."""
import numpy as np
import torch

from ...contracts import SamplingResult
from ...baseline.uncertainty import _power_sample_indices
from .metrics import compute_epistemic_relative_views


@torch.no_grad()
def stochastic_d2_select(pool_embed, train_embed, epistemic_energy, budget, seed, *, device='cpu'):
    """Arthur-Vassilvitskii weighted D^2 sampling: p(i) proportional to s_i * min_{a in L U S} ||phi(x_i) - phi(a)||^2."""
    pool = torch.as_tensor(pool_embed, dtype=torch.float64, device=device)
    train = torch.as_tensor(train_embed, dtype=torch.float64, device=device)
    weights = torch.as_tensor(epistemic_energy, dtype=torch.float64, device=device).clamp_min(1e-12)
    n_pool = len(pool)
    budget = min(int(budget), n_pool)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    if not torch.isfinite(pool).all() or not torch.isfinite(train).all() or not torch.isfinite(weights).all():
        raise ValueError('Stochastic D^2 selection requires finite embeddings and weights')

    # Rescale weights by median to avoid float overflow while preserving exact ratios
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

    for _ in range(budget):
        probs = torch.where(used, torch.zeros_like(weights), weights * distances)
        total = probs.sum()
        if not torch.isfinite(total) or total <= 0:
            # Fallback to epistemic weights on remaining unused items
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


def select_switch_d2_pool(context, budget):
    """Track A full method: Uncapped Epistemic Relative Variance + Stochastic Weighted D^2 Sampling."""
    views = compute_epistemic_relative_views(
        context.predictor,
        context.pool,
        context.train.inputs,
        sketch_dim=context.sketch_dim,
        eps=0.01,
        gamma=0.5,
        return_latent=True,
    )
    indices = stochastic_d2_select(
        views['pool_embed'],
        views['train_embed'],
        views['energy'],
        budget,
        seed=context.seed,
        device=context.device,
    )
    energy = views['energy']
    diagnostics = dict(
        method='ours_switch_d2_pool',
        ensemble_members=len(context.predictor.members),
        failure_signature_gamma=0.5,
        energy_median=float(np.median(energy)),
        energy_p99_over_p50=float(np.quantile(energy, 0.99) / max(np.median(energy), 1e-12)),
        energy_selected_mean=float(np.mean(energy[indices])),
        energy_pool_mean=float(np.mean(energy)),
    )
    return SamplingResult(context.pool.ids[indices], diagnostics=diagnostics)


def select_switch_power(context, budget):
    """Track A Pillar-1 ablation: Uncapped Epistemic Relative Variance + Power stochastic sampling (beta=1)."""
    views = compute_epistemic_relative_views(
        context.predictor,
        context.pool,
        None,
        sketch_dim=context.sketch_dim,
        eps=0.01,
        return_latent=False,
    )
    energy = views['energy']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(context.seed))
        indices = _power_sample_indices(energy, budget, beta=1.0).numpy()
    diagnostics = dict(
        method='ours_switch_power',
        ensemble_members=len(context.predictor.members),
        power_beta=1.0,
        energy_median=float(np.median(energy)),
        energy_p99_over_p50=float(np.quantile(energy, 0.99) / max(np.median(energy), 1e-12)),
        energy_selected_mean=float(np.mean(energy[indices])),
        energy_pool_mean=float(np.mean(energy)),
    )
    return SamplingResult(context.pool.ids[indices], diagnostics=diagnostics)
