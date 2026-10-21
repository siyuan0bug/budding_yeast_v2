# From Tau_BNO-main/models/OperatorFree/TransformerBackbone.py, MIT.
import torch
from torch import nn
import numpy as np
from typing import Optional, Literal
from .operatorfree_layers import get_layer_norm
from .operatorfree_nn import MLP, get_activation

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.channels = channels
        self.dtype_override = dtype_override

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc + tensor

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros(
            (x, self.channels),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc + tensor

class TransformerBlock(nn.Module):
    """
    A single Pre-LN Transformer encoder block:
      - LayerNorm -> MultiheadAttention -> residual
      - LayerNorm -> FeedForward -> residual
    Using batch_first=True for convenience (tensors are (B, S, D)).
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.ln1 = get_layer_norm("rmsnorm", d_model, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = get_layer_norm("rmsnorm", d_model, eps=layer_norm_eps)
        act = get_activation(activation)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, D)
            src_key_padding_mask: (B, S) with True for PAD positions to ignore
            attn_mask: (S, S) or (B * num_heads, S, S) boolean/float mask if needed
        """
        # Pre-LN + MHA
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=src_key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)

        # Pre-LN + FFN
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)

        return x

class TransformerPerTokenRegressor(nn.Module):
    """
    Multi-layer Transformer encoder for per-token regression.

    Input:
        - x: (B, S, input_dim)
        - Optional src_key_padding_mask: (B, S) True for PAD tokens
        - Optional attn_mask: (S, S) or broadcasted masks

    Output:
        - y: (B, S, num_outputs_per_token)  # prediction for every token

    Notes:
        - If input_dim != d_model, an input projection is applied.
        - The head projects each token embedding to desired output dimension.
        - No pooling is performed; predictions are token-wise.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "gelu",
        use_positional_encoding: bool = True,
        max_len: int = 2048,
        layer_norm_eps: float = 1e-5,
        num_outputs_per_token: Optional[int] = None,   # if None, defaults to input_dim
        use_position_encoding: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_outputs_per_token = num_outputs_per_token or input_dim

        # Input projection (to model dimension)
        self.input_proj = MLP(
            input_dim=input_dim,
            hidden_dims=[dim_feedforward, ],
            output_dim=d_model,
            activation=activation, 
            use_layernorm=True,
            layernorm_name="rmsnorm",
            dropout=dropout,
        )

        self.pos_enc = PositionalEncoding(d_model) if use_position_encoding else nn.Identity()
        # Encoder stack
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
            ) for _ in range(num_layers)
        ])

        # Optional final LN before head
        self.final_ln = get_layer_norm("rmsnorm", d_model, eps=layer_norm_eps)

        # Token-wise regression head
        self.head = MLP(
            input_dim=d_model,
            hidden_dims=[d_model, ],
            output_dim=self.num_outputs_per_token,
            activation=activation,
            use_layernorm=True,
            layernorm_name="rmsnorm",
            dropout=dropout,
        )

    @staticmethod
    def generate_causal_mask(S: int, device=None, dtype=torch.bool) -> torch.Tensor:
        """
        Create a causal (future-masking) attention mask of shape (S, S).
        True (bool) or -inf (float) positions will be masked by MultiheadAttention.
        """
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)
        if dtype is torch.float32 or dtype is torch.float16 or dtype is torch.bfloat16:
            # Convert to additive mask with -inf for masked positions and 0 elsewhere
            mask = mask.float()
            mask = mask.masked_fill(mask == 1, float("-inf"))
            mask = mask.masked_fill(mask == 0, 0.0)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, input_dim)
            src_key_padding_mask: (B, S) True for PAD positions
            attn_mask: (S, S) or broadcasted masks
        Returns:
            y: (B, S, num_outputs_per_token)
        """
        # Project to model dimension
        x = self.input_proj(x)  # (B, S, D)

        # Positional encodings if enabled
        x = self.pos_enc(x)

        # Pass through stacked encoder layers
        for block in self.layers:
            x = block(x, src_key_padding_mask=src_key_padding_mask, attn_mask=attn_mask)

        # Final normalization + token-wise head
        x = self.final_ln(x)               # (B, S, D)
        y = self.head(x)                   # (B, S, C)
        return y
