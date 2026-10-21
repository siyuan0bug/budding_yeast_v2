# From Tau_BNO-main/models/OperatorFree/MambaBackbone.py, MIT.
# Uses the retained official Mamba-1 parameterization and reference scan backend.
import torch
from torch import nn
from typing import Optional
from .operatorfree_layers import get_layer_norm
from .operatorfree_nn import MLP
from .mamba_core import Mamba

class MambaBlock(nn.Module):
    """
    A single pre-norm Mamba block:
      - RMSNorm -> Mamba -> residual
      - RMSNorm -> MLP(SiLU) -> residual

    Notes:
      * Shapes are batch-first: (B, S, D)
      * `src_key_padding_mask` (B, S) with True for PAD positions:
          - Pads are zeroed before the mixer and MLP to reduce leakage.
          - Output at PAD positions is re-zeroed for convenience.
      * This block is causal by construction (from Mamba core).
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_hidden_dim: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Pre-norm + Mamba mixer
        self.pre_norm1 = get_layer_norm("rmsnorm", d_model, eps=layer_norm_eps)
        self.mixer = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop1 = nn.Dropout(dropout)

        # Pre-norm + Feedforward
        self.pre_norm2 = get_layer_norm("rmsnorm", d_model, eps=layer_norm_eps)
        hidden = mlp_hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, D)
            src_key_padding_mask: (B, S) with True for PAD positions
        """
        # Mixer branch
        h = self.pre_norm1(x)
        if src_key_padding_mask is not None:
            h = h.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        mixed = self.mixer(h)       # (B, S, D)
        x = x + self.drop1(mixed)

        # Feedforward branch
        h = self.pre_norm2(x)
        if src_key_padding_mask is not None:
            h = h.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        ff = self.ffn(h)
        x = x + self.drop2(ff)

        # Keep PAD positions at zero to simplify masked losses downstream
        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        return x

class MambaPerTokenRegressor(nn.Module):
    """
    Multi-layer Mamba encoder for per-token regression (no pooling).

    Input:
        - x: (B, S, input_dim)
        - Optional src_key_padding_mask: (B, S) True for PAD positions

    Output:
        - y: (B, S, num_outputs_per_token)

    Notes:
        - If input_dim != d_model, an input projection is applied.
        - No attention mask is needed; the model is causal by design.
        - Pads are zeroed inside blocks and re-zeroed on the output positions.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_hidden_dim: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        num_outputs_per_token: Optional[int] = None,  # default: same as input_dim
    ):
        super().__init__()
        self.num_outputs = num_outputs_per_token or input_dim

        # Input projection to model dimension
        self.input_proj = MLP(
            input_dim=input_dim,
            hidden_dims=[mlp_hidden_dim, ],
            output_dim=d_model,
            activation="silu",
            use_layernorm=True,
            layernorm_name="rmsnorm",
            dropout=dropout,
        )

        # Stacked Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])

        # Final normalization before head
        self.final_norm = get_layer_norm("rmsnorm", d_model, eps=layer_norm_eps)

        # Token-wise regression head
        self.head = MLP(
            input_dim=d_model,
            hidden_dims=[mlp_hidden_dim, ],
            output_dim=self.num_outputs,
            activation="silu",  # No activation on the head
            use_layernorm=True,
            layernorm_name="rmsnorm",
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,  # kept for API parity; ignored
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, input_dim)
            src_key_padding_mask: (B, S) True for PAD positions
            attn_mask: ignored (kept for parity with TransformerPerTokenRegressor)
        Returns:
            y: (B, S, num_outputs_per_token)
        """
        x = self.input_proj(x)  # (B, S, D)
        for block in self.layers:
            x = block(x, src_key_padding_mask=src_key_padding_mask)
        x = self.final_norm(x)
        y = self.head(x)
        if src_key_padding_mask is not None:
            y = y.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        return y
