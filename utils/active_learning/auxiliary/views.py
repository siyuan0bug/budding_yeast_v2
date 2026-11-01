"""Explicit synthetic-train and unlabelled-input views for research acquisitions.

No datamodule, benchmark names, held-out arrays or experiment paths cross this
boundary. These views prevent accidental access, not malicious Python introspection.
"""
from dataclasses import dataclass
import json
import numpy as np
import torch
from budding_yeast_v2.data.normalization import normalize_y,normalize_p,denormalize_y


@dataclass(frozen=True)
class Normalization:
    mean_y: torch.Tensor
    std_y: torch.Tensor
    p_mean: torch.Tensor
    p_std: torch.Tensor
    norm_type: str
    s_c: object = None

    def decode(self,y):
        return denormalize_y(y,self.mean_y,self.std_y,self.norm_type,self.s_c)

    def encode(self,p,y0,steps,device):
        p=torch.as_tensor(np.asarray(p),dtype=torch.float32,device=device)
        y0=torch.as_tensor(np.asarray(y0),dtype=torch.float32,device=device)
        ic=normalize_y(y0,self.mean_y,self.std_y,self.norm_type,self.s_c)
        times=torch.linspace(0,1,steps,device=device,dtype=ic.dtype)[None,None,:].expand(len(ic),39,-1)
        x=torch.stack((ic[:,:,None].expand(-1,-1,steps),times),2)
        return x,normalize_p(p,self.p_mean,self.p_std,self.norm_type)


@dataclass(frozen=True)
class Inputs:
    params: np.ndarray
    initial: np.ndarray
    metadata: tuple
    overrides: tuple
    ids: np.ndarray


@dataclass(frozen=True)
class Training:
    inputs: Inputs
    trajectories: tuple


@dataclass(frozen=True)
class AcquisitionData:
    pool: Inputs
    train: Training
    normalization: Normalization
    steps: int
    t_max: float


def build_view(engine,dm):
    """The ONLY adapter that can see the legacy engine/datamodule."""
    if getattr(dm,'fit_scope',None)!='train_synthetic':
        raise ValueError('Mechanism acquisition requires train_synthetic preprocessing')
    n=len(dm.train_dataset_synth)
    provenance=getattr(engine,'_cum_synth_provenance',())
    if (not n or len(provenance)!=n or len(engine._cum_synth_data)!=n
            or len(engine._cum_synth_params)!=n):
        raise ValueError('Mechanism acquisitions require aligned current synthetic training provenance')
    # No file reads, real/val/test properties, or bound dm methods are retained.
    initial=np.asarray([r['y_init'] for r in provenance],dtype=np.float64)
    metadata=tuple(json.loads(r['candidate_metadata_json']) for r in provenance)
    overrides=tuple(json.loads(r['overrides_json']) for r in provenance)
    if any(not row.get('pairs',row.get('genes')) for row in metadata):
        raise ValueError('Synthetic training mutation metadata is required')
    train_ids=np.asarray([r['candidate_id'] for r in provenance],dtype=np.int64)
    pool_ids=engine.pool.ensure_ids().copy()
    if ((train_ids<0).any() or (pool_ids<0).any() or len(np.unique(pool_ids))!=len(pool_ids)
            or len(set(map(int,train_ids)))!=n or set(map(int,train_ids))&set(map(int,pool_ids))):
        raise ValueError('Training/pool candidate identities overlap or repeat')
    def numpy(value):return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
    pool=Inputs(numpy(engine._pool_P_real),numpy(engine._pool_initial_physical),
                tuple(engine._pool_mass_meta),tuple(engine._pool_ovr),pool_ids)
    train=Inputs(np.asarray(engine._cum_synth_params,dtype=np.float64),initial,metadata,overrides,train_ids)
    for block in (pool,train):
        if (block.params.shape!=(len(block.ids),141) or block.initial.shape!=(len(block.ids),39)
                or len(block.metadata)!=len(block.ids) or len(block.overrides)!=len(block.ids)
                or not np.isfinite(block.params).all() or not np.isfinite(block.initial).all()):
            raise ValueError('Malformed mechanism acquisition inputs')
    norm=Normalization(*(getattr(dm,key).detach().cpu().clone() for key in ('mean_y','std_y','p_mean','p_std')),
                       norm_type=dm.norm_type,s_c=getattr(dm,'s_c',None))
    steps=int(dm.train_dataset_synth.tensors[0].shape[-1])
    if steps!=500:raise ValueError('Budding mechanism acquisitions require the canonical 500-point grid')
    return AcquisitionData(pool,Training(train,tuple(engine._cum_synth_data)),norm,steps,float(dm.t_max))


def pairs(row):return tuple(sorted((str(g),str(m)) for g,m in row.get('pairs',row.get('genes',()))))


def contexts(inputs):
    from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
    mdt=PARAM_KEYS.index('mdt')
    return [(tuple(sorted({g for g,m in pairs(meta)})),float(p[mdt]),
             bool(ovr.get('is_sbf_ca',False)),bool(ovr.get('is_mutant_104',False)))
            for p,meta,ovr in zip(inputs.params,inputs.metadata,inputs.overrides)]


def feature_values(inputs,tokens):
    lookup={token:i for i,token in enumerate(tokens)}
    modes=np.zeros((len(inputs.ids),len(tokens)),dtype=np.float32)
    for i,row in enumerate(inputs.metadata):
        for token in pairs(row):
            if token not in lookup:raise ValueError(f'Unsupported mutation token: {token}')
            modes[i,lookup[token]]=1.
    return np.concatenate((inputs.params.astype(np.float32),inputs.initial.astype(np.float32),modes),axis=1)
