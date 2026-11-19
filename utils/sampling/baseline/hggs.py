"""HGGS GF Eq4 + Gaussian-mixture stratified MGS Eq5, trajectory adaptation.

Interpolation returns *desired coordinates* only. The workflow projects onto
fresh canonical proposals; the original biological model is never interpolated.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class HGGSConfig:
    coarse_multiplier: int = 2
    boundary_fraction: float = .4  # 20% of coarse bank; retained GF total is 50%.
    neighbors: int = 5
    output_sketch: int = 64
    crossover_fraction: float = .6
    mixture: bool = True
    global_steps: int = 5

    def validate(self):
        if min(self.coarse_multiplier,self.neighbors,self.output_sketch,self.global_steps)<1:raise ValueError('Invalid HGGS dimensions')
        if not 0<self.boundary_fraction<1 or not 0<self.crossover_fraction<1:raise ValueError('Invalid HGGS proportions')
        return self


def gradient_degree(inputs,outputs,neighbors=8):
    x=np.asarray(inputs,dtype=np.float64);y=np.asarray(outputs,dtype=np.float64)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid HGGS GF domain')
    nearest=NearestNeighbors(n_neighbors=min(neighbors+1,len(x))).fit(x)
    scores=np.empty(len(x))
    # Bounded neighbor queries: never construct/store an N by N distance matrix.
    for start in range(0,len(x),128):
        _,ids=nearest.kneighbors(x[start:start+128])
        for local,row in enumerate(ids):
            i=start+local;row=row[row!=i][:neighbors]
            distance=np.sum((x[row]-x[i])**2,axis=1)
            if (distance<=1e-24).any():raise ValueError('Duplicate HGGS input embeddings')
            scores[i]=np.mean(np.sum((y[row]-y[i])**2,axis=1)/distance)
    if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite HGGS gradient degree')
    return scores


class MultigridGenetic:
    def __init__(self,seed=42,config=HGGSConfig()):
        self.config=config.validate();self.rng=np.random.default_rng(seed);self.seed=seed;self.cycles=0
        self.inputs=None;self.high=None;self.middle=None;self.degeneracy=None;self.proposed=0

    def refresh(self,inputs,residuals):
        self.inputs=np.asarray(inputs,dtype=np.float64)
        residuals=np.asarray(residuals,dtype=np.float64)
        if len(residuals)<3 or residuals.shape!=(len(self.inputs),) or not np.isfinite(residuals).all():
            raise ValueError('HGGS MGS needs at least three finite revealed residuals')
        self.degeneracy=None
        if self.config.mixture and len(np.unique(residuals))>=3:
            gmm=GaussianMixture(n_components=3,random_state=self.seed+self.cycles,reg_covar=1e-8).fit(residuals[:,None])
            labels=gmm.predict(residuals[:,None]);order=np.argsort(gmm.means_[:,0])
            self.high=np.flatnonzero(labels==order[2]);self.middle=np.flatnonzero(labels==order[1])
        else:self.high=self.middle=np.array([],dtype=int)
        if len(self.high)<2 or not len(self.middle):
            # Author code also falls back from GMM to residual quantiles.
            order=np.argsort(residuals,kind='stable');cut=max(1,len(order)//3)
            self.high=order[-max(2,cut):];self.middle=order[cut:-max(2,cut)]
            if not len(self.middle):self.middle=order[:1]
            self.degeneracy='author-supported residual-quantile fallback'
        self.cycles+=1

    def propose(self,count):
        if self.inputs is None:raise RuntimeError('Refresh HGGS revealed residual strata first')
        targets=[];parents=[];crossovers=0
        fraction=self.config.crossover_fraction
        crossover_quota=int(np.floor((self.proposed+count)*fraction))-int(np.floor(self.proposed*fraction))
        for position in range(count):
            crossover=position<crossover_quota
            if crossover:a,b=self.rng.choice(self.high,2,replace=False);crossovers+=1
            else:a=self.rng.choice(self.high);b=self.rng.choice(self.middle)
            alpha=self.rng.random(self.inputs.shape[1])
            targets.append(alpha*(self.inputs[a]-self.inputs[b])+self.inputs[b]);parents.append([int(a),int(b)])
        self.proposed+=count
        return np.asarray(targets),dict(cycle=self.cycles,crossover_count=crossovers,mutation_count=count-crossovers,
            parent_indices=parents,strata_fallback=self.degeneracy,rule='HGGS MGS Eq5 high-high/high-middle boxes')

    def state_dict(self):return dict(config=self.config.__dict__,rng=self.rng.bit_generator.state,
        inputs=self.inputs,high=self.high,middle=self.middle,cycles=self.cycles,degeneracy=self.degeneracy,proposed=self.proposed)
    def load_state_dict(self,state):
        if state['config']!=self.config.__dict__:raise ValueError('HGGS settings changed')
        self.rng.bit_generator.state=state['rng']
        for key in ('inputs','high','middle','cycles','degeneracy','proposed'):setattr(self,key,state[key])
