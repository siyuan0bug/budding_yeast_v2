"""Bounded raw resident subset, shared normalization and uniform weighted draws."""
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset


class ResidentDataset(Dataset):
    def __init__(self,normalization,inputs,reader,initial_rows,*,capacity,draws,seed):
        self.normalization=normalization;self.inputs=inputs;self.reader=reader
        self.lookup={int(row):i for i,row in enumerate(inputs.ids)}
        self.capacity=capacity;self.draws=draws;self.rng=np.random.default_rng(seed);self.exposure=Counter()
        self.rows=[];self.weights=np.ones(capacity);self.truth=None;self.slots={}
        self.set_rows(initial_rows)

    def __len__(self):return self.draws

    def set_rows(self,rows,weights=None):
        rows=list(map(int,rows))
        if not rows or len(rows)>self.capacity or len(set(rows))!=len(rows) or not set(rows)<=self.lookup.keys():
            raise ValueError('Invalid dynamic resident training rows')
        truth=self.reader(rows)['data']
        if not np.isfinite(truth).all():raise FloatingPointError('Nonfinite dynamic supervision')
        if self.truth is None:self.truth=np.empty((self.capacity,*truth.shape[1:]),np.float32)
        self.truth[:len(rows)]=truth;self.rows=rows;self.slots={row:i for i,row in enumerate(rows)}
        weights=np.ones(len(rows)) if weights is None else np.asarray(weights,dtype=float)
        if weights.shape!=(len(rows),) or (weights<0).any() or not np.isfinite(weights).all():raise ValueError('Invalid dynamic weights')
        self.weights[:len(rows)]=weights

    def append(self,rows):
        rows=list(map(int,rows))
        if len(set(rows))!=len(rows) or set(rows)&self.slots.keys() or not set(rows)<=self.lookup.keys() or len(self.rows)+len(rows)>self.capacity:
            raise ValueError('Invalid incremental dynamic rows')
        truth=self.reader(rows)['data'];start=len(self.rows)
        if not np.isfinite(truth).all():raise FloatingPointError('Nonfinite dynamic supervision')
        self.truth[start:start+len(rows)]=truth;self.weights[start:start+len(rows)]=1.
        for row in rows:self.slots[row]=len(self.rows);self.rows.append(row)

    def batch(self,rows):
        slots=np.asarray([self.slots[int(row)] for row in rows]);indices=np.asarray([self.lookup[int(row)] for row in rows])
        x,p=self.normalization.encode(self.inputs.params[indices],self.inputs.initial[indices],self.truth.shape[-1])
        return dict(x=x,p=p,y=self.normalization.targets(self.truth[slots]),
                    sample_weight=torch.tensor(self.weights[slots],dtype=torch.float32),
                    source_row=torch.tensor(rows,dtype=torch.int64))

    def __getitems__(self,indices):
        rows=np.asarray(self.rows)[self.rng.integers(len(self.rows),size=len(indices))]
        batch=self.batch(rows)
        return [{key:value[i] for key,value in batch.items()} for i in range(len(rows))]
    def __getitem__(self,index):return self.__getitems__([index])[0]

    def state(self):return dict(rows=self.rows,weights=self.weights[:len(self.rows)].copy(),
        exposure=dict(self.exposure),rng=self.rng.bit_generator.state)
    def restore(self,state):
        self.set_rows(state['rows'],state['weights']);self.exposure=Counter({int(k):int(v) for k,v in state['exposure'].items()})
        self.rng.bit_generator.state=state['rng']
