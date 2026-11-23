"""Exact single-module backgrounds and revealed, non-degenerate path support."""
import hashlib,json
from collections import defaultdict
import numpy as np
from budding_yeast_v2.data.systems.budding.catalog import touch_set
from ..adaptive_generation.features import context_key


def axes(meta):return sorted({g for g,_ in meta.get('pairs',meta.get('genes',()))})


def path_id(p,meta,gene,param_names):
    tokens=meta.get('pairs',meta.get('genes',()))
    edits=set().union(*(touch_set(g,m) for g,m in tokens if g==gene))
    if not edits or any(edits & touch_set(g,m) for g,m in tokens if g!=gene):return None
    outside=[i for i,k in enumerate(param_names) if k not in edits]
    overrides=meta.get('_effective_overrides',meta.get('original_overrides',{}))
    fixed={k:v for k,v in overrides.items() if (k.startswith('init_') or k.startswith('is_')) and k not in edits}
    description=json.dumps([context_key(meta),gene,fixed],sort_keys=True)
    return hashlib.sha256(description.encode()+np.asarray(p[outside],dtype='<f8').tobytes()).hexdigest()


class PathIndex:
    def __init__(self,param_names,input_scale):
        self.param_names=param_names;self.scale=input_scale;self.paths=defaultdict(list)

    def add(self,index,p,y,meta,target):
        for gene in axes(meta):
            key=path_id(p,meta,gene,self.param_names)
            if key is None:continue
            self.paths[key].append(dict(index=index,gene=gene,x=np.r_[p,y]/self.scale,target=target))

    def support(self,key):
        rows=self.paths.get(key,[])
        if len(rows)<2:return False
        x=np.array([r['x'] for r in rows])
        return bool(np.max(np.linalg.norm(x-x[0],axis=1))>1e-8)

    def estimate(self,p,y,meta):
        key=meta.get('adaptive_path_id')
        if key is None or not self.support(key):return None,0
        rows=self.paths[key];x=np.r_[p,y]/self.scale
        distance=np.array([np.sum((r['x']-x)**2) for r in rows])
        chosen=[rows[i] for i in np.argsort(distance,kind='stable')[:8]]
        xs=np.array([r['x'] for r in chosen]);targets=np.array([r['target'] for r in chosen])
        dx=xs-xs[0];dy=(targets-targets[0]).reshape(len(chosen),-1)
        gram=dx@dx.T;ridge=max(float(np.trace(gram))*1e-4,1e-8)
        weights=np.linalg.solve(gram+ridge*np.eye(len(chosen)),dx@(x-xs[0]))
        result=targets[0]+(weights@dy).reshape(targets.shape[1:])
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite revealed response interpolation')
        return result.astype(np.float32),len(chosen)
