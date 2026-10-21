"""Standard WNO temporal adaptation; coarsest low/detail coefficient-position kernels.
Source: Tau_BNO-main/models/WNO, using ptwt for the wavelet backend.
"""
import ptwt
import torch
from torch import nn
from torch.nn import functional as F
from .common import joint_condition_grid

class WaveletConv1d(nn.Module):
    """WNO coarsest low/detail kernels indexed by coefficient position.

    ptwt replaces pytorch_wavelets; its coefficient list is [low, coarse detail,
    ..., fine detail]. Finer details are zeroed, matching the upstream 1D WNO.
    This version intentionally supports the configured fixed resolution only.
    """
    def __init__(self, width, seq_len=500, level=3, wavelet='haar'):
        super().__init__()
        import pywt
        if width < 1 or seq_len < 2 or level < 1:
            raise ValueError("Invalid wavelet width, sequence length or level")
        if level > pywt.dwt_max_level(seq_len, pywt.Wavelet(wavelet).dec_len):
            raise ValueError("Wavelet level exceeds the boundary-free decomposition limit")
        self.seq_len, self.level, self.wavelet = seq_len, level, wavelet
        with torch.no_grad():
            coeffs = ptwt.wavedec(torch.zeros(1, 1, seq_len), wavelet, level=level, mode='symmetric')
        self.low_weight = nn.Parameter(torch.rand(width, width, coeffs[0].shape[-1]) / width**2)
        self.detail_weight = nn.Parameter(torch.rand(width, width, coeffs[1].shape[-1]) / width**2)

    def forward(self, x):
        if x.shape[-1] != self.seq_len:
            raise ValueError(f"WNO configured for {self.seq_len} queries, got {x.shape[-1]}")
        dtype = x.dtype
        z = x if dtype == torch.float64 else x.float()
        # Explicitly disable outer autocast: ptwt cannot reconstruct BF16
        # coefficients, even when its original input was promoted to FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            coeffs = ptwt.wavedec(z, self.wavelet, level=self.level, mode='symmetric')
            out = [torch.zeros_like(c) for c in coeffs]
            out[0] = torch.einsum('bim,iom->bom', coeffs[0], self.low_weight.to(z.dtype))
            out[1] = torch.einsum('bim,iom->bom', coeffs[1], self.detail_weight.to(z.dtype))
            result = ptwt.waverec(out, self.wavelet)[..., :self.seq_len]
        return result.to(dtype)

class WNO(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4,
                 seq_len=500, level=3, wavelet='haar'):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be positive")
        self.num_vars, self.param_dim = num_vars, param_dim
        self.lift = nn.Conv1d(num_vars + param_dim + 1, width, 1)
        self.wavelet = nn.ModuleList([
            WaveletConv1d(width, seq_len, level, wavelet) for _ in range(n_blocks)
        ])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_blocks)])
        self.project = nn.Sequential(nn.Conv1d(width, 256, 1), nn.GELU(), nn.Conv1d(256, num_vars, 1))

    def forward(self, x, params):
        h = self.lift(joint_condition_grid(x, params, self.num_vars, self.param_dim))
        for i, (wavelet, local) in enumerate(zip(self.wavelet, self.local)):
            h = wavelet(h) + local(h)
            if i < len(self.wavelet) - 1:
                h = F.mish(h)
        return self.project(h)
