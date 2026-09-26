import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet) for Yeast Biological Dynamics:
    - Branch Network: Encodes kinetic parameters and initial conditions into basis expansion coefficients.
    - Trunk Network: Learns continuous time-domain basis functions over the time grid t in [0, 1].
    - Output: One-shot tensor contraction producing the entire (B, 39, 500) trajectory.
    """
    def __init__(self, num_vars=39, param_dim=141, p_basis=128, hidden_dim=256):
        super().__init__()
        self.num_vars = num_vars
        self.p_basis = p_basis

        # Branch Network: maps (parameters + initial conditions) -> basis weights per variable
        in_branch = param_dim + num_vars
        self.branch_net = nn.Sequential(
            nn.Linear(in_branch, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_vars * p_basis)
        )

        # Trunk Network: maps time coordinate t -> continuous basis functions
        self.trunk_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, p_basis)
        )

        # Learnable per-variable bias
        self.bias = nn.Parameter(torch.zeros(num_vars))

    def forward(self, ic_time_grid, params):
        """
        ic_time_grid: (B, num_vars, 2, T) where [:, :, 0, :] is IC and [:, :, 1, :] is time grid
        params: (B, param_dim)
        """
        B, V, F_in, T_steps = ic_time_grid.shape

        # 1. Extract initial condition vector (B, 39)
        ic_vec = ic_time_grid[:, :, 0, 0] # (B, num_vars)
        branch_in = torch.cat([params, ic_vec], dim=-1) # (B, param_dim + num_vars)

        # 2. Compute Branch coefficients (B, num_vars, p_basis)
        branch_out = self.branch_net(branch_in).reshape(B, V, self.p_basis)

        # 3. Extract time coordinates: (B, T_steps, 1)
        t_coords = ic_time_grid[:, 0, 1, :].unsqueeze(-1)  # (B, T_steps, 1)

        # 4. Compute Trunk continuous basis functions: (B, T_steps, p_basis)
        trunk_out = self.trunk_net(t_coords)  # (B, T_steps, p_basis)

        # 5. One-shot contraction: B_{b, v, k} * T_{b, t, k} -> (B, V, T_steps)
        out = torch.einsum('bvk, btk -> bvt', branch_out, trunk_out) + self.bias.unsqueeze(0).unsqueeze(-1)

        return out
