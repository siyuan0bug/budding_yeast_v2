"""Independent BAIT trace-risk selection with rank-one information updates.

Initial training conditioning uses one solve, not a sequential candidate-matrix
whitening per training row. No full candidate-by-candidate kernel is stored.
"""
import numpy as np
import torch
from .geometry import feature_matrices


@torch.no_grad()
def bait_select(pool, train, budget, device='cpu', *, noise=1e-2, posterior=True,
                with_train=True, overselection_factor=1.):
    pool, train = feature_matrices(pool, train, device)
    budget = min(int(budget), len(pool))
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    if noise <= 0 or overselection_factor < 1:
        raise ValueError('BAIT requires positive noise and overselection >= 1')
    pool, train = pool.double(), train.double()
    eye = torch.eye(pool.shape[1], device=device, dtype=pool.dtype)
    if posterior:
        energy = train.square().sum(1).mean()
        if not torch.isfinite(energy) or energy <= 0:
            raise ValueError('BAIT training features have zero energy')
        pool, train = pool / energy.sqrt(), train / energy.sqrt()
        lower = torch.linalg.cholesky(train.T @ train + noise**2 * eye)
        pool = torch.linalg.solve_triangular(lower, pool.T, upper=False).T * noise
        train = torch.linalg.solve_triangular(lower, train.T, upper=False).T * noise
    covariance = pool.T @ pool + train.T @ train
    information = train.T @ train if with_train else torch.zeros_like(eye)
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(information + noise**2 * eye)) * noise**2
    projected = pool @ inverse
    diagonal = (projected * pool).sum(1)
    numerator = ((projected @ covariance) * projected).sum(1)
    selected, used = [], torch.zeros(len(pool), device=device, dtype=torch.bool)

    def update(index, remove=False):
        vector = projected[index].clone()
        denominator = noise**2 + (-diagonal[index] if remove else diagonal[index])
        if not torch.isfinite(denominator) or denominator <= 0:
            raise ValueError('BAIT information update lost positive definiteness')
        sign = 1 if remove else -1
        dots = pool @ vector
        cov_vector = covariance @ vector
        cross = projected @ cov_vector
        energy = vector @ cov_vector
        numerator.add_(sign * 2 * dots * cross / denominator
                       + energy * dots.square() / denominator.square())
        diagonal.add_(sign * dots.square() / denominator)
        projected.add_(sign * dots[:, None] * vector[None] / denominator)

    forward_count = min(round(overselection_factor * budget), len(pool))
    for _ in range(forward_count):
        scores = numerator.clamp_min(0) / (diagonal.clamp_min(0) + noise**2 + 1e-8)
        scores[used] = -torch.inf
        index = int(scores.argmax())
        if not torch.isfinite(scores[index]):
            raise ValueError('BAIT has no finite selection score')
        selected.append(index)
        used[index] = True
        update(index)
    for _ in range(forward_count - budget):
        ids = torch.as_tensor(selected, device=device)
        denominator = diagonal[ids] - noise**2
        scores = numerator[ids].clamp_min(0) / denominator
        scores[denominator >= 0] = -torch.inf
        position = int(scores.argmax())
        if not torch.isfinite(scores[position]):
            raise ValueError('BAIT backward step has no valid removal')
        index = selected.pop(position)
        update(index, remove=True)
        used[index] = False
    return np.asarray(selected, dtype=np.int64)
