"""
Vendored subset of pdearena (https://github.com/pdearena/pdearena, commit 22360a7).
pdearena 包未安装且 pip 不可用, 这里内联 Unet1D 所需的少量符号.
仅依赖 torch + 标准库, 无传递依赖.
"""
import math
from abc import abstractmethod

import torch
import torch.nn as nn


ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def fourier_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ConditionedBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        ...
