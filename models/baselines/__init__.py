"""Selected standard baselines; only initial-state/parameter/time I/O is adapted."""
from .mlp import MLP
from .resnet import ResNet
from .unet import UNet
from .kan import KAN
from .transformer import Transformer
from .mamba import Mamba
from .deeponet import DeepONet
from .fno import FNO
from .wno import WNO
from .mwt import MWT
from .gnot import GNOT
from .transolver import Transolver
from .lno import LNO
from .neural_ode import NeuralODEModel
from .sc_fno import SCFNO
from .bregman_fno import BregmanFNO
from .lamo import LaMO
from .konode import KoNODE

BASELINE_REGISTRY = {
    'mlp':MLP, 'resnet':ResNet, 'unet':UNet, 'deeponet':DeepONet,
    'kan':KAN, 'transformer':Transformer, 'mamba':Mamba,
    'fno':FNO, 'wno':WNO, 'mwt':MWT,
    'gnot':GNOT, 'transolver':Transolver, 'lno':LNO,
    'neural_ode':NeuralODEModel, 'sc_fno':SCFNO,
    'bregman_fno':BregmanFNO, 'lamo':LaMO, 'konode':KoNODE,
}


def build_baseline(name, *, num_vars, param_dim, width, n_blocks, modes,
                   wavelet='haar', seq_len=500, level=3,
                   ode_method='dopri5', ode_rtol=1e-3, ode_atol=1e-4):
    cls = BASELINE_REGISTRY[name]
    args = dict(num_vars=num_vars,param_dim=param_dim)
    if name == 'deeponet':
        args.update(hidden_dim=width,p_basis=width,n_blocks=n_blocks)
    elif name == 'neural_ode':
        args.update(hidden_dim=width,n_blocks=n_blocks,method=ode_method,rtol=ode_rtol,atol=ode_atol)
    else:
        args.update(width=width,n_blocks=n_blocks)
        if name in ('fno','sc_fno','bregman_fno','mwt'):
            args['modes'] = modes
        if name == 'wno':
            args.update(wavelet=wavelet,seq_len=seq_len,level=level)
        if name == 'konode':
            args.update(method=ode_method,rtol=ode_rtol,atol=ode_atol)
    return cls(**args)


__all__ = ['MLP','ResNet','UNet','KAN','Transformer','Mamba','DeepONet','FNO','WNO','MWT','GNOT','Transolver','LNO',
           'NeuralODEModel','SCFNO','BregmanFNO','LaMO','KoNODE',
           'BASELINE_REGISTRY','build_baseline']
