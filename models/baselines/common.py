"""Shared initial-value task input contract; no learned conditioning modules."""
import torch

def joint_condition_grid(x, params, num_vars, param_dim, *, uniform=True):
    """Return (B, V+P+1, T), broadcasting only the state at the first query."""
    if x.ndim != 4 or x.shape[1:3] != (num_vars, 2) or x.shape[-1] < 2:
        raise ValueError(f"Expected input (B, {num_vars}, 2, T>=2)")
    if params.shape != (x.shape[0], param_dim):
        raise ValueError(f"Expected parameters (B, {param_dim})")
    t = x[:, :1, 1, :]
    if not torch.equal(x[:, :, 1, :], t.expand(-1, num_vars, -1)):
        raise ValueError("All state channels must share the query time grid")
    if not torch.isfinite(t).all() or not torch.all(t[..., 1:] > t[..., :-1]):
        raise ValueError("Query times must be finite and strictly increasing")
    # These FFT/DWT adaptations assume a regular temporal discretization.
    dt = t[..., 1:] - t[..., :-1]
    if uniform and not torch.allclose(dt, dt[..., :1].expand_as(dt), rtol=1e-4, atol=1e-6):
        raise ValueError("Temporal operators require uniformly spaced queries")
    initial = x[:, :, 0, :1].expand(-1, -1, x.shape[-1])
    p = params.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.cat((initial, p, t), dim=1)
