"""Explicit IC layout and parameter-space contracts for biological models."""
from __future__ import annotations

import torch
from torch import nn


def extract_initial_state(x: torch.Tensor, num_vars: int) -> torch.Tensor:
    """Return (B,V,1) ICs without mistaking interleaved time channels for ICs.

    Supported layouts: the DataModule's (B,V,2,T), legacy IC-only (B,V,T),
    and explicitly interleaved flattened (B,2V,T). A block-concatenated tensor
    must not be passed as the interleaved form; callers should pass original x.
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"Expected a 3D/4D IC input, got {tuple(x.shape)}")
    if x.shape[-1] < 1:
        raise ValueError("Initial-state input must contain at least one time point")
    if x.ndim == 4 and x.shape[1:3] == (num_vars, 2):
        return x[:, :, 0, :1]
    if x.ndim == 3:
        if x.shape[1] == num_vars:
            return x[:, :, :1]
        if x.shape[1] == 2 * num_vars:
            return x[:, 0::2, :1]
    raise ValueError(f"Expected (B,{num_vars},2,T), (B,{num_vars},T), or interleaved "
                     f"(B,{2 * num_vars},T), got {tuple(x.shape)}")


class ParameterInputTransform(nn.Module):
    """Decode model parameters and form WT differences in physical units.

    Statistics describe only the parameter transform, not the Y transform.
    They must be explicitly supplied by the DataModule, or callers must opt
    into ``norm_type='physical'``. All buffers persist in new checkpoints.
    """
    _MODES = {"physical": 0, "log1p": 1, "linear": 1, "asinh_z": 2, "asinh_scale": 3}

    def __init__(self, param_dim: int):
        super().__init__()
        self.param_dim = param_dim
        self.register_buffer("p_mean", torch.zeros(1, param_dim))
        self.register_buffer("p_std", torch.ones(1, param_dim))
        self.register_buffer("norm_code", torch.tensor(-1, dtype=torch.long))

    def set_parameter_stats(self, p_mean=None, p_std=None, norm_type="log1p"):
        if norm_type not in self._MODES:
            raise ValueError(f"Unsupported parameter normalization: {norm_type!r}")
        mode = self._MODES[norm_type]
        if mode == 0:
            mean, std = torch.zeros_like(self.p_mean), torch.ones_like(self.p_std)
        else:
            if p_mean is None or p_std is None:
                raise ValueError("Normalized parameter inputs require p_mean and p_std")
            mean = torch.as_tensor(p_mean, device=self.p_mean.device, dtype=self.p_mean.dtype)
            std = torch.as_tensor(p_std, device=self.p_std.device, dtype=self.p_std.dtype)
            if mean.numel() != self.param_dim or std.numel() != self.param_dim:
                raise ValueError(f"Parameter statistics must contain {self.param_dim} coordinates")
            mean, std = mean.reshape_as(self.p_mean), std.reshape_as(self.p_std)
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise ValueError("Parameter statistics must be finite")
            if (std < 0).any() or (mode == 1 and (std == 0).any()):
                raise ValueError("Parameter scales must be nonnegative (strictly positive for Z-score)")
        with torch.no_grad():
            self.p_mean.copy_(mean)
            self.p_std.copy_(std)
            self.norm_code.fill_(mode)

    def _mode(self):
        mode = int(self.norm_code.item())
        if mode not in self._MODES.values():
            raise RuntimeError("Parameter statistics are not configured. Call set_parameter_stats("
                               "p_mean, p_std, norm_type) before forwarding normalized inputs; "
                               "for raw physical parameters explicitly use norm_type='physical'.")
        return mode

    def _stats(self, p):
        if p.ndim != 2 or p.shape[1] != self.param_dim or not p.is_floating_point():
            raise ValueError(f"Expected floating parameter tensor (B,{self.param_dim}), got {tuple(p.shape)}")
        return self.p_mean.to(p), self.p_std.to(p)

    def encode(self, physical):
        """Apply exactly the parameter transform used by YeastDataModule."""
        mode = self._mode()
        mean, std = self._stats(physical)
        if mode == 0:
            return physical
        if mode == 1:
            return (physical - mean) / std
        varying = std > 1e-5
        safe_std = torch.where(varying, std, torch.ones_like(std))
        if mode == 2:
            return torch.where(varying, torch.asinh((physical - mean) / safe_std), 0.)
        return torch.where(varying, (torch.log10(physical.clamp(min=1e-8)) - mean) / safe_std, 0.)

    def to_physical(self, p):
        mode = self._mode()
        mean, std = self._stats(p)
        if mode == 0:
            return p
        if mode == 1:
            return p * std + mean
        varying = std > 1e-5
        if mode == 2:
            safe_std = torch.where(varying, std, torch.ones_like(std))
            return torch.where(varying, torch.sinh(p) * safe_std + mean, mean).clamp(min=0.)
        transformed = torch.where(varying, p * std + mean, mean).clamp(min=-8., max=6.)
        return (10. ** transformed).clamp(min=0.)

    def intervention_features(self, p, physical_wt, eps=1e-5, logarithm="signed"):
        """Physical Delta_p/log residual; keep normalized p for learned conditioning.

        Encode WT with the same statistics and subtract in matching coordinates.
        This gives exactly zero at encoded WT despite floating-point round trips.
        For lossy transforms, both sides share the same floor/constant handling.
        """
        mode = self._mode()
        _, std = self._stats(p)
        wt = physical_wt.to(p).reshape(1, self.param_dim)
        encoded_wt = self.encode(wt)
        if mode == 0:
            delta = p - wt
        elif mode == 1:
            delta = (p - encoded_wt) * std
        else:
            delta = self.to_physical(p) - self.to_physical(encoded_wt)
        if logarithm == "signed":
            log_res = torch.sign(delta) * torch.log1p(delta.abs() / (wt.abs() + eps))
        elif logarithm == "ratio":
            # The Dynamic/Latch variants use an ordinary floored log ratio.
            # Decode both sides alike so encoded WT still has exactly zero log ratio.
            log_res = (torch.log(self.to_physical(p).clamp(min=eps))
                       - torch.log(self.to_physical(encoded_wt).clamp(min=eps)))
        else:
            raise ValueError(f"Unknown intervention logarithm: {logarithm!r}")
        return delta, log_res

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Old checkpoints have no transform buffers. Preserve already-injected
        # DataModule stats, or remain unconfigured and fail clearly on forward.
        names = ("p_mean", "p_std", "norm_code")
        if not any(prefix + name in state_dict for name in names):
            for name in names:
                state_dict[prefix + name] = getattr(self, name).detach().clone()
        elif not all(prefix + name in state_dict for name in names):
            absent = [prefix + name for name in names if prefix + name not in state_dict]
            error_msgs.append(f"Incomplete parameter-normalization state: missing {absent}")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
