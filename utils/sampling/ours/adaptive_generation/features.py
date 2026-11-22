"""Fixed-scale trajectory and portable directed-event descriptions."""
import numpy as np
import torch


def trajectory_features(y,scale):
    y=np.asarray(y,dtype=np.float64)
    if y.ndim!=3 or not np.isfinite(y).all():raise ValueError('Nonfinite trajectory features')
    grid=np.unique(np.linspace(0,y.shape[-1]-1,min(33,y.shape[-1])).round().astype(int))
    return np.arcsinh(y[:,:,grid]/np.asarray(scale)[None,:,None]).reshape(len(y),-1)


def event_features(y,params,system):
    """Counts, first/last crossing times and ordered transition frequencies.

    Uses only supplied trajectories, with native guard directions. No claim
    that a predicted crossing is a verified physical event/reset.
    """
    y=np.asarray(y,dtype=np.float64)
    if y.ndim!=3 or not np.isfinite(y).all():raise ValueError('Nonfinite event input')
    values,directions=system.guards(torch.tensor(y.transpose(0,2,1).copy()),torch.tensor(np.asarray(params).copy(),dtype=torch.float64))
    values=values.detach().cpu().numpy();n=len(directions);out=[]
    for v in values:
        sequence=[];f=[]
        for j,d in enumerate(directions):
            a,b=v[:-1,j],v[1:,j]
            hit=np.where((a>=0)&(b<0) if d<0 else (a<=0)&(b>0))[0]
            times=(hit+np.abs(a[hit])/(np.abs(a[hit])+np.abs(b[hit])))/(len(v)-1)
            f.extend((np.log1p(len(hit)),float(times[0]) if len(hit) else 1.,float(times[-1]) if len(hit) else 1.))
            sequence.extend((float(t),j) for t in times)
        order=[j for _,j in sorted(sequence)];transition=np.zeros((n,n))
        for a,b in zip(order,order[1:]):transition[a,b]+=1
        out.append(np.r_[f,np.log1p(transition).ravel()])
    return np.asarray(out)


def context_key(meta):
    pairs=meta.get('pairs',meta.get('genes',()))
    environment=meta.get('_effective_overrides',meta.get('original_overrides',{}))
    return '|'.join(f'{g}:{m}' for g,m in sorted(pairs))+f"|mdt={environment.get('mdt','unknown')}"
