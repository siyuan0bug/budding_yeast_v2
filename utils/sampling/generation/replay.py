"""Bounded full-trajectory replay, with a fixed number of draws per epoch.

Raw source trajectories remain in the immutable store. This RAM buffer is only
an online training view; evicted trajectories are not deleted from that store.
All transforms use the shared initial-training normalization object.
"""
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryReplay(Dataset):
    def __init__(self,normalization,params,initial,truth,*,capacity,draws,seed):
        if min(capacity,draws)<1:raise ValueError('Positive replay capacity/draws required')
        self.normalization=normalization;self.capacity=int(capacity);self.draws=int(draws)
        self.rng=np.random.default_rng(seed);self.exposure=Counter();self.total_inserted=0
        self.params=np.empty((capacity,params.shape[1]),np.float64)
        self.initial=np.empty((capacity,initial.shape[1]),np.float64)
        self.truth=np.empty((capacity,*truth.shape[1:]),np.float32)
        self.row_ids=np.empty(capacity,np.int64);self.size=0
        self.is_reference=np.empty(capacity,dtype=bool)
        self.add(params,initial,truth)

    def __len__(self):return self.draws

    def add(self,params,initial,truth,*,is_reference=None):
        if not len(params)==len(initial)==len(truth):raise ValueError('Unaligned replay data')
        if not all(np.isfinite(v).all() for v in (params,initial,truth)):raise ValueError('Nonfinite replay inputs')
        flags=np.ones(len(params),dtype=bool) if is_reference is None else np.asarray(is_reference)
        if flags.dtype!=np.bool_ or flags.shape!=(len(params),):raise ValueError('Invalid replay reference provenance')
        for p,h,y,reference in zip(params,initial,truth,flags):
            i=self.total_inserted%self.capacity
            self.params[i]=p;self.initial[i]=h;self.truth[i]=y;self.row_ids[i]=self.total_inserted
            self.is_reference[i]=reference
            self.total_inserted+=1;self.size=min(self.capacity,self.total_inserted)

    def __getitems__(self,indices):
        ids=self.rng.integers(self.size,size=len(indices))
        x,p=self.normalization.encode(self.params[ids],self.initial[ids],self.truth.shape[-1])
        y=self.normalization.targets(self.truth[ids])
        return [dict(x=x[j],p=p[j],y=y[j],online_params=torch.from_numpy(self.params[i].copy()),
                     online_initial=torch.from_numpy(self.initial[i].copy()),
                     online_is_reference=torch.tensor(bool(self.is_reference[i])),
                     online_row=torch.tensor(self.row_ids[i])) for j,i in enumerate(ids)]

    def __getitem__(self,index):return self.__getitems__([index])[0]

    def record_exposure(self,rows):self.exposure.update(map(int,rows))

    def state(self):
        return dict(rng=self.rng.bit_generator.state,total_inserted=self.total_inserted,
                    exposure=dict(self.exposure),capacity=self.capacity,draws=self.draws,
                    reference_flags=self.is_reference[:self.size].copy())

    def restore(self,state,params,initial,truth):
        if (state['capacity'],state['draws'])!=(self.capacity,self.draws):raise ValueError('Replay settings changed')
        self.size=0;self.total_inserted=0
        self.add(params,initial,truth)
        if self.total_inserted!=state['total_inserted']:raise ValueError('Replay/source row count changed')
        flags=np.asarray(state.get('reference_flags'))
        if flags.dtype!=np.bool_ or flags.shape!=(self.size,):raise ValueError('Missing/invalid replay reference provenance')
        self.is_reference[:self.size]=flags
        self.rng.bit_generator.state=state['rng']
        self.exposure=Counter({int(k):int(v) for k,v in state['exposure'].items()})
