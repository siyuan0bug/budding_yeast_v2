import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None


class DerivativeMLP(nn.Module):
    def __init__(self, state_dim=39, param_dim=141, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + param_dim + 1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, state_dim)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, y, params):
        if isinstance(t, (float, int)):
            t = torch.full((y.size(0), 1), t, device=y.device)
        elif t.dim() == 0:
            t = t.expand(y.size(0), 1)
        inp = torch.cat([y, params, t], dim=-1)
        return self.net(inp)


class NeuralODEModel(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, hidden_dim=128,
                 method='dopri5', rtol=1e-3, atol=1e-4):
        super().__init__()
        if odeint is None:
            raise ImportError("Please install torchdiffeq first: pip install torchdiffeq")

        self.num_vars = num_vars
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.method = method
        self.rtol = rtol
        self.atol = atol

        self.derivative_net = DerivativeMLP(
            state_dim=num_vars, param_dim=param_dim, hidden_dim=hidden_dim
        )

    def forward(self, ic_time_grid, params):
        B, V, _, T = ic_time_grid.shape

        y0 = ic_time_grid[:, :, 0, 0]

        t_grid = torch.linspace(0, 1, T, device=ic_time_grid.device)

        pred_y = odeint(
            lambda t, y: self.derivative_net(t, y, params),
            y0, t_grid,
            method=self.method, rtol=self.rtol, atol=self.atol
        )

        pred_y = pred_y.permute(1, 2, 0)
        return pred_y