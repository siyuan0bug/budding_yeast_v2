"""
1D Dual-Path Operator Transformer (DPOT1d) for Budding Yeast Cell Cycle.
Adapted from RealPDEBench DPOT (Wang et al., ICLR/NeurIPS 2024).

Architecture:
- 1D Patch / Token Embedding: Projects (B, 39*2, T) to (B, T, D).
- 1D AFNO (Adaptive Fourier Neural Operator) Blocks:
  - 1D Real FFT along temporal dimension.
  - Multi-block diagonal complex weight contraction with non-linear activation & thresholding.
  - 1D Inverse Real FFT back to temporal domain.
- Dual-Path Feed-Forward Transformer Sub-layers:
  - Temporal LayerNorm + GELU MLP with residual connection.
  - FiLM (Feature-wise Linear Modulation) from kinetic parameters P (141-dim).
- Projection Head: Projects back to (B, 39, T).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AFNO1D(nn.Module):
    """
    1D Adaptive Fourier Neural Operator Layer.
    Performs block-diagonal complex matrix multiplications in 1D frequency space.
    """
    def __init__(self, hidden_size=128, num_blocks=8, modes=32, sparsity_threshold=0.01, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} must be divisible by num_blocks {num_blocks}"
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.modes = modes
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size_factor = hidden_size_factor

        scale = 1.0 / (self.block_size * self.block_size * self.hidden_size_factor)
        # Complex weights: w1 (real, imag), w2 (real, imag)
        self.w1 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size, self.block_size * hidden_size_factor))
        self.b1 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size * hidden_size_factor))
        self.w2 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size * hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size))

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        x_orig = x

        # 1D RFFT along temporal dimension
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")  # (B, T//2 + 1, D)
        K = x_fft.shape[1]
        kept_modes = min(self.modes, K)

        # Reshape to multi-block representation
        x_blocks = x_fft.reshape(B, K, self.num_blocks, self.block_size)

        o1_real = torch.zeros(B, K, self.num_blocks, self.block_size * self.hidden_size_factor, device=x.device, dtype=x.dtype)
        o1_imag = torch.zeros(B, K, self.num_blocks, self.block_size * self.hidden_size_factor, device=x.device, dtype=x.dtype)

        # Block 1
        x_real_kept = x_blocks[:, :kept_modes].real
        x_imag_kept = x_blocks[:, :kept_modes].imag

        o1_real[:, :kept_modes] = F.gelu(
            torch.einsum('bkni,nio->bkno', x_real_kept, self.w1[0]) -
            torch.einsum('bkni,nio->bkno', x_imag_kept, self.w1[1]) +
            self.b1[0]
        )
        o1_imag[:, :kept_modes] = F.gelu(
            torch.einsum('bkni,nio->bkno', x_imag_kept, self.w1[0]) +
            torch.einsum('bkni,nio->bkno', x_real_kept, self.w1[1]) +
            self.b1[1]
        )

        # Block 2
        o2_real = (
            torch.einsum('bkni,nio->bkno', o1_real[:, :kept_modes], self.w2[0]) -
            torch.einsum('bkni,nio->bkno', o1_imag[:, :kept_modes], self.w2[1]) +
            self.b2[0]
        )
        o2_imag = (
            torch.einsum('bkni,nio->bkno', o1_imag[:, :kept_modes], self.w2[0]) +
            torch.einsum('bkni,nio->bkno', o1_real[:, :kept_modes], self.w2[1]) +
            self.b2[1]
        )

        out_fft = torch.zeros_like(x_blocks)
        out_fft[:, :kept_modes] = torch.complex(o2_real, o2_imag)

        # Soft shrinkage in frequency domain for sparsity
        out_fft_flat = out_fft.reshape(B, K, D)
        
        # 1D IRFFT back to time domain
        x_out = torch.fft.irfft(out_fft_flat, n=T, dim=1, norm="ortho")
        return x_orig + x_out


class DPOTBlock1D(nn.Module):
    """
    DPOT Transformer Block with AFNO1D Mixer + FiLM Modulation + MLP.
    """
    def __init__(self, hidden_size=128, num_blocks=8, modes=32, mlp_ratio=4, param_dim=141):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.afno = AFNO1D(hidden_size=hidden_size, num_blocks=num_blocks, modes=modes)
        
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size)
        )
        
        # FiLM parameter modulation: gamma, beta for both sub-layers
        self.film = nn.Sequential(
            nn.Linear(param_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * 2)
        )

    def forward(self, x, params):
        # x: (B, T, D), params: (B, 141)
        gamma, beta = self.film(params).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)  # (B, 1, D)
        beta = beta.unsqueeze(1)    # (B, 1, D)

        # AFNO sub-layer with FiLM
        x_norm = self.norm1(x)
        x_mod = (1.0 + gamma) * x_norm + beta
        x = x + self.afno(x_mod)

        # MLP sub-layer
        x = x + self.mlp(self.norm2(x))
        return x


class DPOT1d(nn.Module):
    """
    1D Dual-Path Operator Transformer (DPOT1d) Model.
    
    Args:
        num_variables (int): Number of biological species (39).
        param_dim (int): Kinetic parameter dimension (141).
        hidden_size (int): Latent embedding width (default: 128).
        modes (int): Kept Fourier modes (default: 32).
        depth (int): Number of DPOT blocks (default: 4).
        num_blocks (int): Number of frequency block diagonals (default: 8).
        mlp_ratio (int): MLP expansion ratio (default: 4).
    """
    def __init__(
        self,
        num_variables: int = 39,
        param_dim: int = 141,
        hidden_size: int = 128,
        modes: int = 32,
        depth: int = 4,
        num_blocks: int = 8,
        mlp_ratio: int = 4,
        in_features: int = 2,
        **kwargs
    ):
        super().__init__()
        self.num_variables = num_variables
        self.param_dim = param_dim
        self.hidden_size = hidden_size

        # Input projection: (B, 39*2, T) -> (B, T, hidden_size)
        in_channels = num_variables * in_features
        self.input_proj = nn.Conv1d(in_channels, hidden_size, kernel_size=1)

        # Positional Encoding for 1D time grid
        self.pos_emb = nn.Parameter(torch.zeros(1, 1000, hidden_size))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Stack of DPOT Blocks
        self.blocks = nn.ModuleList([
            DPOTBlock1D(
                hidden_size=hidden_size,
                num_blocks=num_blocks,
                modes=modes,
                mlp_ratio=mlp_ratio,
                param_dim=param_dim
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        
        # Output projection: (B, hidden_size, T) -> (B, 39, T)
        self.head = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_size, num_variables, kernel_size=1)
        )

    def forward(self, ic_time_grid: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            ic_time_grid: (B, 39, 2, T)
            params: (B, 141)
        Returns:
            pred: (B, 39, T)
        """
        B, V, F_in, T = ic_time_grid.shape
        x = ic_time_grid.reshape(B, V * F_in, T)  # (B, 78, T)

        # Project to latent width: (B, hidden_size, T) -> (B, T, hidden_size)
        x = self.input_proj(x).permute(0, 2, 1)

        # Add positional embedding
        if T <= self.pos_emb.shape[1]:
            x = x + self.pos_emb[:, :T, :]
        else:
            pos = F.interpolate(self.pos_emb.permute(0, 2, 1), size=T, mode='linear', align_corners=False).permute(0, 2, 1)
            x = x + pos

        # Apply DPOT Transformer Blocks with Parameter FiLM
        for block in self.blocks:
            x = block(x, params)

        x = self.norm(x)

        # Output projection: (B, T, hidden_size) -> (B, hidden_size, T) -> (B, 39, T)
        x = x.permute(0, 2, 1)
        out = self.head(x)
        return out
