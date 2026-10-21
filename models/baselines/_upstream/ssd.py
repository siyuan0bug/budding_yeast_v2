"""Unfused SSD recurrence and gated RMS normalization used by LaMO's Hydra.

Exact discrete recurrence: S_t = exp(dt_t A) S_(t-1) + dt_t B_t x_t,
y_t = C_t S_t. This changes the numerical backend, not the SSM equations.
The sequential implementation is intended for the small latent slice sequence.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RMSNormGated(nn.Module):
    def __init__(self, width, eps=1e-5, norm_before_gate=True, device=None, dtype=None):
        super().__init__()
        self.eps, self.norm_before_gate = eps, norm_before_gate
        self.weight = nn.Parameter(torch.ones(width, device=device, dtype=dtype))

    def forward(self, x, z):
        dtype = x.dtype
        h = x.float()
        if not self.norm_before_gate:
            h = h * F.silu(z.float())
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()
        if self.norm_before_gate:
            h = h * F.silu(z.float())
        return h.to(dtype)


def scan(x, dt, A, B, C, chunk_size=None, D=None, z=None, seq_idx=None,
         initial_states=None, dt_limit=(0.0, float('inf'))):
    if seq_idx is not None:
        raise ValueError('Packed sequence identifiers are not used for ODE trajectories')
    batch, length, heads, head_dim = x.shape
    groups, state_dim = B.shape[-2:]
    if heads % groups:
        raise ValueError('SSD heads must be divisible by groups')
    dtype = x.dtype
    x, dt, A, B, C = (v.float() for v in (x, dt, A, B, C))
    dt = dt.clamp(*dt_limit)
    B = B.repeat_interleave(heads // groups, dim=2)
    C = C.repeat_interleave(heads // groups, dim=2)
    state = (x.new_zeros(batch, heads, head_dim, state_dim) if initial_states is None
             else initial_states.float())
    outputs = []
    for i in range(length):
        decay = torch.exp(dt[:, i] * A).unsqueeze(-1).unsqueeze(-1)
        injection = torch.einsum('bh,bhn,bhp->bhpn', dt[:, i], B[:, i], x[:, i])
        state = decay * state + injection
        outputs.append(torch.einsum('bhpn,bhn->bhp', state, C[:, i]))
    out = torch.stack(outputs, dim=1)
    if D is not None:
        out = out + x * D.view(1, 1, heads, -1)
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype)
