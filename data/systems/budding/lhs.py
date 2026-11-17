"""Probability-space LHS for configured mixed marginals, including exact atoms.

Only necessary initial-state/event dependencies are shared. This does not infer
genotypes or preserve the joint parameter relationships of genetic constructs.
"""
import numpy as np
from scipy.special import betainc, ndtr
from scipy.stats import qmc


def bounds(part):
    if part['family']=='point':return part['value'],part['value']
    if part['family']=='discrete':return min(part['values']),max(part['values'])
    return part['lower'],part['upper']


def mixture_cdf(x, parts):
    x=np.asarray(x,dtype=float);result=np.zeros_like(x)
    for part in parts:
        family=part['family'];lo,hi=bounds(part)
        if family=='point':v=(x>=part['value']).astype(float)
        elif family=='discrete':
            order=np.argsort(part['values']);values=np.asarray(part['values'])[order]
            cumulative=np.r_[0.,np.cumsum(np.asarray(part['probabilities'])[order])]
            v=cumulative[np.searchsorted(values,x,side='right')]
        elif family=='uniform':v=np.clip((x-lo)/(hi-lo),0.,1.)
        elif family=='log_uniform':
            v=np.clip(np.log(np.maximum(x,np.finfo(float).tiny)/lo)/np.log(hi/lo),0.,1.)
        elif family=='scaled_beta':
            v=betainc(part['a'],part['b'],np.clip(x/part['parameter_scale'],0.,1.))
        elif family=='clipped_normal_scaled':
            v=ndtr((x/part['parameter_scale']-part['alpha_mean'])/part['alpha_std'])
            v=np.where(x<lo,0.,np.where(x>=hi,1.,v))
        else:raise ValueError(f'Unknown marginal family {family}')
        result+=part['weight']*v
    return np.clip(result,0.,1.)


def mixture_atoms(parts):
    atoms={}
    def add(value,weight):atoms[value]=atoms.get(value,0.)+float(weight)
    for part in parts:
        weight=part['weight'];family=part['family']
        if family=='point':add(part['value'],weight)
        elif family=='discrete':
            for value,p in zip(part['values'],part['probabilities']):add(value,weight*p)
        elif family=='clipped_normal_scaled':
            lo,hi=bounds(part);scale=part['parameter_scale']
            add(lo,weight*ndtr((lo/scale-part['alpha_mean'])/part['alpha_std']))
            add(hi,weight*ndtr(-(hi/scale-part['alpha_mean'])/part['alpha_std']))
    return sorted(atoms.items())


def mixture_ppf(u, parts):
    """Generalized inverse of the FULL mixture, not concatenated component PPFs."""
    u=np.asarray(u,dtype=float)
    if not np.isfinite(u).all() or (u<0).any() or (u>1).any():raise ValueError('Invalid probability coordinates')
    if abs(sum(p['weight'] for p in parts)-1.)>1e-10:raise ValueError('Marginal weights must sum to one')
    if len(parts)==1 and parts[0]['family']=='uniform':
        lo,hi=bounds(parts[0]);return lo+u*(hi-lo)
    output=np.full(u.shape,np.nan)
    low=min(bounds(p)[0] for p in parts);high=max(bounds(p)[1] for p in parts)
    output[u==0]=low;output[u==1]=high
    for value,mass in mixture_atoms(parts):
        upper=float(mixture_cdf(value,parts));lower=upper-mass
        mask=(u>lower)&(u<=upper)&np.isnan(output)
        output[mask]=value
    missing=np.isnan(output)
    if missing.any():
        target=u[missing];left=np.full(target.shape,low);right=np.full(target.shape,high)
        for _ in range(54):
            middle=left+(right-left)*.5
            below=mixture_cdf(middle,parts)<target
            left=np.where(below,middle,left);right=np.where(below,right,middle)
        output[missing]=right
    if not np.isfinite(output).all():raise ValueError('Nonfinite mixture inverse')
    return output


def sample_design(spec, names, count, seed):
    """Return one latent Latin design and physical parameters; no ODE call."""
    if count<1:raise ValueError('A design needs at least one row')
    names=list(names);u=qmc.LatinHypercube(d=len(names),seed=seed).random(count)
    # Both pairs have matching zero probabilities. Sharing the full coordinate
    # preserves their marginals while preventing incompatible initial states.
    for source,target in [('init_CDH1T','init_CDH1'),('BUB2_active','init_BUB2')]:
        u[:,names.index(target)]=u[:,names.index(source)]
    params=np.empty_like(u)
    for j,name in enumerate(names):params[:,j]=mixture_ppf(u[:,j],spec['marginals'][name])
    return params,u
