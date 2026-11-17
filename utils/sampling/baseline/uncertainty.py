"""strategies uncertainty services extracted from the pre-package implementation."""
import numpy as np
import torch


def _al4pde_uncertainty_sample(self, trainer, pl_module, dm, device, current_epoch):
    """Official independent ensemble variance + mean aggregation + top-K."""
    if self._pool_size == 0:
        return None
    scores = self._compute_ensemble_uncertainty(device, np.arange(self._pool_size))
    n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
    # Same ordering, including ties, as official uncertainty_based.top_k.
    selected = torch.argsort(scores, descending=True)[:n_select].numpy()
    print(f'[al4pde_uncertainty] Ensemble={len(self._ensemble_members)} | top_k={n_select}')
    return self.make_selection_plan(selected, dm)


def _power_sample_indices(scores, k, beta=1.0):
    """Official power_sampling for regular inputs; explicit degenerate handling."""
    scores = torch.as_tensor(scores).detach().cpu()
    if scores.ndim != 1 or not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Power requires finite nonnegative scalar uncertainty scores')
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError('Power beta must be finite and positive')
    k = min(int(k), len(scores))
    if k <= 0:
        return torch.empty(0, dtype=torch.long)
    weights = torch.pow(scores, beta)
    if not torch.isfinite(weights).all() or not torch.isfinite(weights.sum()) or (scores.max() > 0 and weights.sum() == 0):
        weights = torch.pow(scores / scores.max(), beta)
    positive = torch.where(weights > 0)[0]
    if len(positive) >= k:
        # Exact operations from al4pde/acquisition/uncertainty_based.py.
        prob = weights / weights.sum()
        return torch.multinomial(prob, k, replacement=False)
    if len(positive):
        chosen = positive[torch.multinomial(weights[positive] / weights[positive].sum(),
                                            len(positive), replacement=False)]
    else:
        chosen = torch.empty(0, dtype=torch.long)
    zeros = torch.where(weights == 0)[0]
    remaining = k - len(chosen)
    extra = zeros[torch.multinomial(torch.ones(len(zeros)), remaining, replacement=False)]
    return torch.cat((chosen, extra))


def _al4pde_power_sample(self, trainer, pl_module, dm, device, current_epoch):
    if self._pool_size == 0:
        return None
    scores = self._compute_ensemble_uncertainty(device, np.arange(self._pool_size))
    n_select = self._al4pde_scheduled_num_add()
    selected = self._power_sample_indices(scores, n_select, self.power_beta).numpy()
    return self.make_selection_plan(selected, dm)


