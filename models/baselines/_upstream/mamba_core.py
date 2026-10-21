# Copyright (c) 2023, Tri Dao, Albert Gu. Apache-2.0.
# Official Mamba-1 commit e9594ce1c732d97440f0332fdc43170a2294dbfa.
# __init__ retained; full-sequence forward uses equivalent torch layouts.
# No inference cache or fused inner kernel; selective scan uses official CUDA when available.
import math
import warnings
import torch
from torch import nn
from torch.nn import functional as F

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(B.float().unflatten(-1,(-1,2)))
        if is_variable_C:
            C = torch.view_as_complex(C.float().unflatten(-1,(-1,2)))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = B.repeat_interleave(dim // B.shape[1],dim=1)
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = C.repeat_interleave(dim // C.shape[1],dim=1)
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * D[:,None]
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan
except ImportError:
    _cuda_scan = None

_warned_cuda_reference = False


def selective_scan(u, *args, **kwargs):
    global _warned_cuda_reference
    if u.is_cuda and _cuda_scan is not None:
        return _cuda_scan(u, *args, **kwargs)
    if u.is_cuda and not _warned_cuda_reference:
        warnings.warn('mamba_ssm CUDA scan unavailable; using the official PyTorch reference scan. '
                      'Install a compatible mamba-ssm build for full-scale training throughput.', stacklevel=2)
        _warned_cuda_reference = True
    return selective_scan_ref(u, *args, **kwargs)


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank='auto', dt_min=0.001, dt_max=0.1, dt_init='random', dt_scale=1.0, dt_init_floor=0.0001, conv_bias=True, bias=False, use_fast_path=True, layer_idx=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs)
        self.activation = 'silu'
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** (-0.5) * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device).unsqueeze(0).expand(self.d_inner, -1).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states):
        # Full sequence B,L,D -> B,inner,L, as in the official unfused path.
        batch, length, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :length])
        projected = self.x_proj(x.transpose(1, 2))
        dt, B, C = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight).transpose(1, 2)
        B, C = B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous()
        y = selective_scan(x, dt, -torch.exp(self.A_log.float()), B, C, self.D.float(),
                           z=z, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        return self.out_proj(y.transpose(1, 2))
