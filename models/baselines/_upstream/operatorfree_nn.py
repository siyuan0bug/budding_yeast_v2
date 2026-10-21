# From Tau_BNO-main/models/OperatorFree/NN.py, MIT.
from typing import List, Optional, Union, Sequence, Callable, Dict
import torch
from torch import nn
from .operatorfree_layers import get_layer_norm

def get_activation(name: Optional[Union[str, Callable[..., nn.Module]]]) -> nn.Module:
    if name is None or str(name).lower() in {"none", "identity", "linear"}:
        return nn.Identity()
    if callable(name) and not isinstance(name, str):
        return name()  # type: ignore
    name = str(name).lower()
    table: Dict[str, Callable[[], nn.Module]] = {
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True),
        "gelu": lambda: nn.GELU(),
        "selu": lambda: nn.SELU(),
        "elu": lambda: nn.ELU(alpha=1.0, inplace=True),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
        "silu": lambda: nn.SiLU(),  # aka swish
        "swish": lambda: nn.SiLU(),
    }
    if name not in table:
        raise ValueError(f"Unsupported activation: {name}")
    return table[name]()

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: Optional[int] = None,
        activation: Optional[Union[str, Callable[..., nn.Module]]] = "relu",
        activations_per_layer: Optional[Sequence[Optional[Union[str, Callable[..., nn.Module]]]]] = None,
        use_layernorm: bool = False,
        layernorm_name: str = "layernorm",
        layernorm_per_layer: Optional[Sequence[str]] = None,
        dropout: Optional[float] = None,
        dropout_per_layer: Optional[Sequence[Optional[float]]] = None,
        activation_first: bool = False,
        last_activation: bool = False,
        bias_when_bn: bool = False,
    ):
        super().__init__()

        hidden_dims = list(hidden_dims)
        n_layers = len(hidden_dims)

        if activations_per_layer is not None and len(activations_per_layer) != n_layers:
            raise ValueError("`activations_per_layer` length must match `hidden_dims` length.")
        if layernorm_per_layer is not None and len(layernorm_per_layer) != n_layers:
            raise ValueError("`layernorm_per_layer` length must match `hidden_dims` length.")
        if dropout_per_layer is not None and len(dropout_per_layer) != n_layers:
            raise ValueError("`dropout_per_layer` length must match `hidden_dims` length.")

        layers: List[nn.Module] = []
        prev = input_dim

        for i, h in enumerate(hidden_dims):
            # ---- per-layer configs ----
            act = (activations_per_layer[i] if activations_per_layer is not None else activation)
            ln_on = (layernorm_per_layer[i] if layernorm_per_layer is not None else layernorm_name)
            p_drop = (dropout_per_layer[i] if dropout_per_layer is not None else dropout)

            # using BN, Linear without bias
            linear_bias = True if (not ln_on or bias_when_bn) else False
            layers.append(nn.Linear(prev, h, bias=linear_bias))

            if activation_first:
                # Linear -> Act -> LN -> Dropout
                if act is not None:
                    layers.append(get_activation(act))
                if ln_on:
                    layers.append(get_layer_norm(ln_on, h))
            else:
                # Linear -> LN -> Act -> Dropout (Default)
                if ln_on:
                    layers.append(get_layer_norm(ln_on, h))
                if act is not None:
                    layers.append(get_activation(act))

            if p_drop is not None and p_drop > 0:
                layers.append(nn.Dropout(p_drop))

            prev = h

        # output layer
        self.has_output = output_dim is not None
        if self.has_output:
            # output layer without BN/Dropout
            self.out = nn.Linear(prev, output_dim, bias=True)
            self.out_act = get_activation(activation) if last_activation else nn.Identity()
        else:
            self.out = None
            self.out_act = nn.Identity()

        self.backbone = nn.Sequential(*layers)

        # save config
        self.config = {
            "input_dim": input_dim,
            "hidden_dims": list(hidden_dims),
            "output_dim": output_dim,
            "activation": activation,
            "use_layernorm": use_layernorm if layernorm_per_layer is None else "per-layer",
            "dropout": dropout if dropout_per_layer is None else "per-layer",
            "activation_first": activation_first,
            "last_activation": last_activation,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if self.has_output:
            x = self.out(x)
            x = self.out_act(x)
        return x
