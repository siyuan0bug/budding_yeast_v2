"""
FNO1d 适配 wrapper.

将 al4pde 的 FNO1d 适配到 BY 的接口标准:
- BY 接口: forward(ic_time_grid: (B, V, 2, T), params: (B, P)) -> (B, V, T)
- FNO1d 接口: forward(x: (B, S, T_in*C), grid: (B, S, 1), params: (B, P)) -> (B, S, 1, C)

映射:
- ic_time_grid (B, V, 2, T) -> permute -> (B, T, 2, V) -> reshape -> (B, T, 2*V) = (B, S=T, T_in*C)
- t_grid 从 ic_time_grid[:, 0, 1, :] 提取 (所有变量共享同一 t_grid)
- params (B, P) 通过输入拼接注入 FNO1d 的 fc0
- 输出 (B, T, 1, V) -> squeeze -> (B, T, V) -> permute -> (B, V, T)
"""
import torch
import torch.nn as nn

from .al4pde_modules.fno import FNO1d


class FNO1DAdapter(nn.Module):
    """适配 al4pde FNO1d 到 BY 的 forward(ic_time_grid, params) -> (B, V, T) 接口."""

    def __init__(
        self,
        num_vars: int = 39,
        param_dim: int = 141,
        modes: int = 16,
        width: int = 64,
        t_steps: int = 500,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim

        self.model = FNO1d(
            num_channels=num_vars,
            modes=modes,
            width=width,
            initial_step=2,   # IC 通道 + t_grid 通道
            predict_delta=False,
            param_dim=param_dim,
        )

    def forward(self, ic_time_grid: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ic_time_grid: (B, V, 2, T) - 通道 0=IC 广播, 通道 1=t_grid
            params:      (B, P)

        Returns:
            (B, V, T)
        """
        B, V, _, T = ic_time_grid.shape

        # (B, V, 2, T) -> (B, T, 2, V) -> (B, T, 2*V) = (B, S, T_in*C)
        x = ic_time_grid.permute(0, 3, 2, 1).reshape(B, T, -1)

        # t_grid 从输入提取: (B, V, 2, T) -> 取通道 1 的任意变量 -> (B, T) -> (B, T, 1)
        grid = ic_time_grid[:, 0, 1, :].unsqueeze(-1)

        # FNO1d forward
        out = self.model(x, grid, params=params)  # (B, T, 1, V)

        # (B, T, 1, V) -> (B, T, V) -> (B, V, T)
        out = out.squeeze(-2).permute(0, 2, 1)
        return out
