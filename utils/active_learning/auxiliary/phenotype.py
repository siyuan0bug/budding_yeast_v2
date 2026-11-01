"""Streaming physical descriptors and canonical 500-point window groups."""
import numpy as np
import torch
from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
from budding_yeast_v2.utils.runtime_logging import ProgressLog

DESCRIPTOR_NAMES=tuple(f'{stat}_{j}' for stat in ('log_peak','log_end','log_mean') for j in range(39))+('divisions','first_division_fraction')


def describe(values):
    y=np.asarray(values)
    labels=pattern_group_metadata(y);drops=np.diff(y[:,0])<-.1
    first=np.where(drops.any(1),(drops.argmax(1)+1)/(y.shape[-1]-1),1.)
    features=np.concatenate((np.log1p(np.maximum(y.max(2),0)),np.log1p(np.maximum(y[:,:,-1],0)),
        np.log1p(np.maximum(y.mean(2),0)),labels['observed_division_counts'][:,None],first[:,None]),axis=1)
    if not np.isfinite(features).all():raise ValueError('Nonfinite physical acquisition descriptor')
    return features.astype(np.float32),labels['pattern_group_ids']


def training_descriptors(view):
    features=[];labels=[]
    for start in range(0,len(view.train.inputs.ids),256):
        values=np.asarray(view.train.trajectories[start:start+256],dtype=np.float32)
        phi,c=describe(values);features.append(phi);labels.append(c)
    return np.concatenate(features),np.concatenate(labels)


def predicted_descriptors(view,members,device,batch_size=128):
    if len(members)<2:raise ValueError('Pattern acquisition requires independent ensemble members')
    count=len(view.pool.ids);features=np.empty((count,len(DESCRIPTOR_NAMES)),dtype=np.float32)
    votes=np.zeros((count,6),dtype=np.float32)
    devices=[next(m.parameters()).device for m in members];modes=[m.training for m in members]
    progress=ProgressLog('pattern descriptor inference',count)
    try:
        for member in members:member.to(device).eval()
        with torch.no_grad():
            for start in range(0,count,batch_size):
                end=min(start+batch_size,count)
                x,p=view.normalization.encode(view.pool.params[start:end],view.pool.initial[start:end],view.steps,device)
                batch=[]
                for member in members:
                    physical=view.normalization.decode(member.forward_ic_time(x,p).float()).cpu().numpy()
                    phi,c=describe(physical);batch.append(phi)
                    votes[start:end]+=np.eye(6,dtype=np.float32)[c]/len(members)
                # Average descriptors, never average trajectories before labelling.
                features[start:end]=np.mean(batch,axis=0)
                progress.update(end)
    finally:
        for member,mode,original in zip(members,modes,devices):member.to(original).train(mode)
    return features,votes
