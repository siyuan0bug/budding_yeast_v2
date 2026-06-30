"""
Unet1D 适配 wrapper.

将 al4pde 的 Unet1D (PDEArena 风格) 适配到 BY 的接口标准:
- BY 接口: forward(ic_time_grid: (B, V, 2, T), params: (B, P)) -> (B, V, T)
- Unet1D 接口: forward(x: (B, T_in, C, S), z: (B, num_params)) -> (B, T_out, C_out, S)

映射:
- ic_time_grid (B, V, 2, T) -> permute -> (B, 2, V, T) = (B, T_in=2, C=V, S=T)
- params (B, P) -> z (B, P), Unet1D 内部逐参数 fourier_embedding
- 输出 (B, 1, V, T) -> squeeze -> (B, V, T)
"""
import torch
import torch.nn as nn

from .al4pde_modules.unet_cond_1d import Unet1D


class Unet1DAdapter(nn.Module):
    """适配 al4pde Unet1D 到 BY 的 forward(ic_time_grid, params) -> (B, V, T) 接口."""

    def __init__(
        self,
        num_vars: int = 39,
        param_dim: int = 141,
        hidden_channels: int = 64,
        ch_mults=(1, 2, 2, 4),
        n_blocks: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.model_ch_mults = ch_mults

        self.model = Unet1D(
            padding_mode="zeros",
            n_input_scalar_components=num_vars,
            n_input_vector_components=0,
            n_output_scalar_components=num_vars,
            n_output_vector_components=0,
            time_history=2,   # IC 通道 + t_grid 通道
            time_future=1,    # 一次性预测整条轨迹
            hidden_channels=hidden_channels,
            activation=activation,
            ch_mults=ch_mults,
            n_blocks=n_blocks,
            param_conditioning=f"scalar_{param_dim}",
            n_dims=1,
        )

    def forward(self, ic_time_grid: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ic_time_grid: (B, V, 2, T) - 通道 0=IC 广播, 通道 1=t_grid
            params:      (B, P)

        Returns:
            (B, V, T)
        """
        T = ic_time_grid.shape[-1]
        # Unet1D 的 Downsample/Upsample 对非 2 的幂次尺寸会失配, pad 到最近的 2^N
        n_resolutions = len(self.model_ch_mults)
        min_size = 2 ** n_resolutions
        if T % min_size != 0:
            T_padded = ((T + min_size - 1) // min_size) * min_size
            pad = T_padded - T
            ic_time_grid = torch.nn.functional.pad(ic_time_grid, (0, pad))

        # (B, V, 2, T) -> (B, 2, V, T) = (B, T_in, C, S)
        x = ic_time_grid.permute(0, 2, 1, 3)

        # Unet1D forward: 只传 z (params), 不传 time
        out = self.model(x, time=None, z=params)  # (B, 1, V, T_padded)

        # (B, 1, V, T_padded) -> (B, V, T_padded) -> 裁剪回 (B, V, T)
        out = out.squeeze(1)
        if T != out.shape[-1]:
            out = out[..., :T]
        return out
