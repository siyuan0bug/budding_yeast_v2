"""
External Baseline Neural Operator and Deep Learning Architectures
for Budding Yeast Cell Cycle Benchmark.

Collection of standard literature baselines (39 models):
Continuous Operators & Physics-Informed:
- Pure FNO (Li et al., ICLR 2021)
- Standard 1D FNO (RealPDEBench / Li et al., ICLR 2021)
- Pure WNO (Tripura & Chakraborty, 2022)
- Continuous Neural Operator (CNO1d, Raonić et al., ICLR 2023)
- Multiwavelet Neural Operator (MWT1d, Gupta et al., NeurIPS 2021)
- Galerkin Transformer 1D (Cao, ICLR 2021)
- Transolver 1D (Physics-informed Attention, ICML 2024 Spotlight)
- Transolver-3 1D (Chunk-Optimized Physics Attention, ICML 2025)
- Dual-Path Operator Transformer (DPOT1d, Wang et al., NeurIPS 2024)
- GCGNet (Global Condition Guided Network, NeurIPS 2024)
- DeepONet (Lu et al., Nat. Mach. Intell. 2021)
- Neural ODE (Chen et al., NeurIPS 2018)
- 1D U-Net (UNet1d, RealPDEBench / Ronneberger et al.)
- ResNet-FiLM (Feature-wise Linear Modulation ResNet, RealPDEBench)
- GNOT (Geometric Neural Operator Transformer, ICML 2023)
- Latent Neural Operator (LNO, NeurIPS 2024)
- LaMO (Large Mesh Operator / Mamba SSM, NeurIPS 2024 Spotlight)
- Deep Koopman Operator (DMD/Koopman, RealPDEBench / NeurIPS 2024)
- Wavelet-Driven Neural Operator (WDNO, RealPDEBench / NeurIPS 2024)
- ICON (In-Context Operator Networks, NeurIPS 2023)
- BAMPNO (Boundary-Adapted Multi-Patch Neural Operator, NeurIPS 2024)
- G-FNO (Group-Equivariant Fourier Neural Operator, ICLR 2023)
- NOBLE (Neural Operator for Biological Dynamics with NeRF Embeddings, NeurIPS 2024)

Long-Term Time-Series & Inverted Transformers:
- iTransformer (Liu et al., ICLR 2024 Spotlight)
- DLinear (Zeng et al., AAAI 2023)
- NLinear (Zeng et al., AAAI 2023)
- Linear (Zeng et al., AAAI 2023)
- Autoformer (Wu et al., NeurIPS 2021)
- Informer (Zhou et al., AAAI 2021 Best Paper)
- iInformer (Liu et al., ICLR 2024)
- Flowformer (Wu et al., ICML 2022)
- iFlowformer (Liu et al., ICLR 2024)
- Reformer (Kitaev et al., ICLR 2020)
- iReformer (Liu et al., ICLR 2024)
- Flashformer (Dao et al. / Transformer variant)
- iFlashformer (Liu et al., ICLR 2024)
- Vanilla Transformer (Vaswani et al., NeurIPS 2017)
- FEDformer (Zhou et al., ICML 2022)
- Pyraformer (Liu et al., ICLR 2022)
"""

# Existing 13 Continuous Operator Baselines
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

# Batch 1: Inverted & Linear Series Baselines
from .itransformer import iTransformer
from .dlinear import DLinear
from .nlinear import NLinear
from .linear import Linear
from .autoformer import Autoformer
from .informer import Informer
from .vanilla_transformer import VanillaTransformer

# Batch 2: Advanced Attention & Decomposed Baselines
from .fedformer import FEDformer
from .pyraformer import Pyraformer
from .flowformer import Flowformer
from .reformer import Reformer
from .flashformer import Flashformer
from .iinformer import iInformer
from .iflowformer import iFlowformer
from .ireformer import iReformer
from .iflashformer import iFlashformer

# Batch 3: Continuous, Group-Equivariant & In-Context Neural Operators
from .gnot import GNOT
from .latent_lno import LatentNeuralOperator
from .lamo import LaMO_RegularGrid
from .koopman_dmd import DeepKoopmanOperator
from .wdno import WDNO
from .transolver3_1d import Transolver3_1d
from .icon import ICONGPT
from .bampno import BAMPNO
from .gfno import G_FNO
from .noble import NOBLE

__all__ = [
    # Continuous Operators
    'PureFNO_ICLR2021',
    'FNO1d_Standard',
    'PureWNO_2022',
    'CNO1d',
    'MWT1d',
    'GalerkinTransformer1d',
    'Transolver1d',
    'Transolver3_1d',
    'DPOT1d',
    'GCGNetYeast',
    'DeepONet',
    'NeuralODEModel',
    'UNet1d',
    'ResNetFiLM',
    'GNOT',
    'LatentNeuralOperator',
    'LaMO_RegularGrid',
    'DeepKoopmanOperator',
    'WDNO',
    'ICONGPT',
    'BAMPNO',
    'G_FNO',
    'NOBLE',
    # Time-Series & Inverted Transformers
    'iTransformer',
    'DLinear',
    'NLinear',
    'Linear',
    'Autoformer',
    'Informer',
    'iInformer',
    'Flowformer',
    'iFlowformer',
    'Reformer',
    'iReformer',
    'Flashformer',
    'iFlashformer',
    'VanillaTransformer',
    'FEDformer',
    'Pyraformer',
]
