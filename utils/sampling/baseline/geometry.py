"""Local Gaussian features and greedy MaxDist/LCMD; no reference-code dependency."""
import math
import numpy as np
import torch


def gaussian_projection(width, dimension, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return torch.randn(width, dimension) / math.sqrt(dimension)


def feature_matrices(pool, train, device):
    pool, train = torch.as_tensor(pool).to(device), torch.as_tensor(train).to(device)
    if pool.ndim != 2 or train.ndim != 2 or pool.shape[1] != train.shape[1]:
        raise ValueError('Pool/train features must have matching dimensions')
    if not torch.isfinite(pool).all() or not torch.isfinite(train).all():
        raise ValueError('Selection requires finite features')
    return pool, train


@torch.no_grad()
def distance_select(pool, train, budget, device='cpu', method='max_dist'):
    pool, train = feature_matrices(pool, train, device)
    budget = min(int(budget), len(pool))
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    if method not in ('max_dist', 'lcmd'):
        raise ValueError(f'Unknown distance selector {method}')
    pool, train = pool.double(), train.double()
    norms = pool.square().sum(1)
    distances = torch.full_like(norms, float('inf'))
    owners = torch.zeros(len(pool), device=device, dtype=torch.long)
    for start in range(0, len(train), 256):
        centers = train[start:start + 256]
        for offset in range(0, len(pool), 8192):
            rows = pool[offset:offset + 8192]
            squared = (norms[offset:offset + len(rows), None]
                       + centers.square().sum(1)[None] - 2 * rows @ centers.T).clamp_min_(0)
            value, index = squared.min(1)
            old = distances[offset:offset + len(rows)]
            better = value < old
            owners[offset:offset + len(rows)][better] = index[better] + start + 1
            old[better] = value[better]
    selected = []
    used = torch.zeros(len(pool), device=device, dtype=torch.bool)
    for step in range(budget):
        if not len(train) and step == 0:
            scores = norms.clone()
        elif method == 'lcmd':
            mass = torch.bincount(owners, weights=distances, minlength=len(train) + step + 1)
            scores = torch.where(mass[owners] == mass.max(), distances, -torch.inf)
        else:
            scores = distances.clone()
        scores[used] = -torch.inf
        index = int(scores.argmax())
        if not torch.isfinite(scores[index]):
            index = int(torch.where(~used)[0][0])
        selected.append(index)
        used[index] = True
        squared = (norms + norms[index] - 2 * (pool @ pool[index])).clamp_min_(0)
        better = squared < distances
        owners[better] = len(train) + step + 1
        distances[better] = squared[better]
    return np.asarray(selected, dtype=np.int64)
