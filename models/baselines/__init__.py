"""
External Baseline Neural Operator and Deep Learning Architectures
for Budding Yeast Cell Cycle Benchmark.

Collection of standard literature baselines:
- Pure FNO (Li et al., ICLR 2021)
- Standard 1D FNO (RealPDEBench / PDEBench)
- Pure WNO (Gupta et al. 2021 / Tripuraneni 2022)
- Continuous Neural Operator (CNO1d, Raonić et al., ICLR 2023)
- Multiwavelet Neural Operator (MWT1d, Gupta et al., NeurIPS 2021)
- Galerkin Transformer 1D (Cao, ICLR 2021)
- Transolver 1D (Physics-informed Attention, ICML 2024 Spotlight)
- Dual-Path Operator Transformer (DPOT1d, Wang et al., ICLR/NeurIPS 2024)
- GCGNet (Graph-Consistent Generative Network, ICLR 2024)
- DeepONet (Lu et al., Nat. Mach. Intell. 2021)
- Neural ODE (Chen et al., NeurIPS 2018)
- 1D U-Net (UNet1d)
- ResNet-FiLM (Feature-wise Linear Modulation ResNet)
"""

from .pure_fno import PureFNO_ICLR2021
from .fno1d import FNO1d_Standard
from .pure_wno import PureWNO_2022
from .cno1d import CNO1d
from .mwt1d import MWT1d
from .galerkin_transformer1d import GalerkinTransformer1d
from .transolver1d import Transolver1d
from .dpot1d import DPOT1d
from .gcgnet import GCGNetYeast
from .deeponet import DeepONet
from .neural_ode import NeuralODEModel
from .unet1d import UNet1d
from .resnet_film import ResNetFiLM

__all__ = [
    'PureFNO_ICLR2021',
    'FNO1d_Standard',
    'PureWNO_2022',
    'CNO1d',
    'MWT1d',
    'GalerkinTransformer1d',
    'Transolver1d',
    'DPOT1d',
    'GCGNetYeast',
    'DeepONet',
    'NeuralODEModel',
    'UNet1d',
    'ResNetFiLM',
]
