"""Prediction-only global geometry and variable-balanced amplitude strata."""
import numpy as np
import torch


class PredictionObserver:
    """Shared streaming summaries; never reads real or held-out trajectories."""
    def __init__(self, count, seed, *, calibrated=False, width=128):
        self.count,self.width,self.seed=count,width,seed
        self.global_features=torch.empty(count,width,dtype=torch.float32)
        self.peak_log=np.empty((count,39),dtype=np.float32)
        self.physical_variance=np.empty((count,39),dtype=np.float32)
        self.calibrated=calibrated
        self.log_contrasts=None
        self._projection={}

    def sketch(self, values):
        size=values.shape[-1];device=values.device
        key=(size,str(device))
        if key not in self._projection:
            rng=np.random.RandomState(self.seed+31279)
            buckets=torch.as_tensor(rng.randint(0,self.width,size=size),device=device)
            signs=torch.as_tensor(rng.choice([-1.,1.],size=size),dtype=torch.float32,device=device)
            self._projection[key]=(buckets,signs)
        buckets,signs=self._projection[key]
        flat=values.reshape(-1,size)
        result=torch.zeros(len(flat),self.width,device=device,dtype=torch.float32)
        result.scatter_add_(1,buckets.expand(len(flat),-1),flat*signs)
        return (result*(self.width/size)**.5).reshape(*values.shape[:-1],self.width)

    def __call__(self, left, right, predictions, dm, device):
        members,n,variables,steps=predictions.shape
        times=torch.linspace(0,steps-1,min(32,steps),device=predictions.device).round().long()
        self.global_features[left:right]=self.sketch(predictions.mean(0)[...,times].flatten(1)).cpu()
        physical=dm.denormalize_y(predictions.flatten(0,1),device).reshape(members,n,variables,steps)
        if not torch.isfinite(physical).all():raise ValueError('Nonfinite hybrid prediction summary')
        mean=physical.mean(0)
        self.peak_log[left:right]=torch.log1p(mean.clamp(min=0).amax(-1)).cpu().numpy()
        self.physical_variance[left:right]=(physical-mean).square().mean((0,3)).cpu().numpy()
        if self.calibrated:
            logs=torch.log1p(physical.clamp(min=0))[...,times].flatten(2)
            logs=logs-logs.mean(0,keepdim=True)
            if members==2:logs=(logs[:1]-logs[1:])*.5
            projected=self.sketch(logs).cpu().numpy()
            if self.log_contrasts is None:
                self.log_contrasts=np.empty((len(projected),self.count,self.width),dtype=np.float32)
            self.log_contrasts[:,left:right]=projected


def select_global(engine, observer, train_features, count, selected, device):
    """Use the official MaxDist selector on a shared 128D ensemble-mean map."""
    if count<=0:return np.empty(0,dtype=np.int64)
    chosen=np.asarray(selected,dtype=np.int64)
    available=np.setdiff1d(np.arange(engine._pool_size),chosen)
    anchors=train_features
    if len(chosen):anchors=torch.cat((anchors,observer.global_features[chosen]))
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(engine.random_seed+97*getattr(engine,'_al4pde_al_iter',0)+len(chosen)))
        local=engine._bmdal_predicted_max_dist_select(observer.global_features[available],anchors,
                                                    min(count,len(available)),device)
    return available[local]


def stratified_tail(observer,pool_size,budget,seed):
    """Round-robin all variables' predicted p90-p99/p99+ strata; no rejection."""
    if budget<=0:return np.empty(0,dtype=np.int64),[]
    peaks=observer.peak_log[:pool_size];variance=observer.physical_variance[:pool_size]
    if not np.isfinite(peaks).all() or not np.isfinite(variance).all():
        raise ValueError('Tail strata require finite predictions')
    quantiles=np.quantile(peaks,[.9,.99],axis=0)
    rng=np.random.RandomState(seed);tie=rng.random(pool_size)
    regions=[]
    for variable in range(peaks.shape[1]):
        low,high=quantiles[:,variable]
        for band,mask in ((0,(peaks[:,variable]>low)&(peaks[:,variable]<=high)),
                          (1,peaks[:,variable]>high)):
            rows=np.flatnonzero(mask)
            if len(rows):
                order=np.lexsort((tie[rows],-variance[rows,variable]))
                regions.append([variable,band,rows[order],0])
    selected=[];seen=set();records=[]
    while len(selected)<budget:
        progress=False
        for region in regions:
            variable,band,rows,cursor=region
            while cursor<len(rows) and int(rows[cursor]) in seen:cursor+=1
            region[3]=cursor
            if cursor==len(rows):continue
            row=int(rows[cursor]);region[3]+=1
            seen.add(row);selected.append(row);progress=True
            records.append(dict(kind='tail',endpoints=[row],new_queries=1,
                variable=variable,amplitude_band='p90_p99' if band==0 else 'p99_plus',
                score=float(variance[row,variable])))
            if len(selected)==budget:break
        if not progress:break
    return np.asarray(selected,dtype=np.int64),records


def edge_features(point,observer,edges,distances,normalized_variance,log_variance,pool_size,
                  fixed_inputs=None,context=None):
    """Symmetric edge/model-state features, independent of real identities."""
    x=np.empty((len(edges),17),dtype=np.float32)
    for start in range(0,len(edges),8192):
        part=edges[start:start+8192];a,b=part.T
        pa,pb=observer.peak_log[a],observer.peak_log[b]
        ua,ub=point[a],point[b]
        mean_a,mean_b=pa.mean(1),pb.mean(1);max_a,max_b=pa.max(1),pb.max(1)
        if fixed_inputs is not None:
            delta=np.abs(fixed_inputs[a]-fixed_inputs[b]);input_mean,input_max=delta.mean(1),delta.max(1)
        else:input_mean=input_max=np.zeros(len(part))
        if context is None:background=np.zeros((len(part),3))
        else:background=context[a]
        x[start:start+len(part)]=np.stack((
            np.log1p(np.minimum(ua,ub)),np.log1p(np.maximum(ua,ub)),
            np.log(np.maximum(log_variance[start:start+len(part)],1e-8)),
            np.log(np.maximum(normalized_variance[start:start+len(part)],1e-8)),
            distances[start:start+len(part)],np.minimum(mean_a,mean_b),np.maximum(mean_a,mean_b),
            np.minimum(max_a,max_b),np.maximum(max_a,max_b),np.abs(pa-pb).mean(1),
            np.abs(pa-pb).max(1),(part<pool_size).mean(1),input_mean,input_max,
            background[:,0],background[:,1],background[:,2]),axis=1)
    if not np.isfinite(x).all():raise ValueError('Nonfinite edge calibration inputs')
    return x
