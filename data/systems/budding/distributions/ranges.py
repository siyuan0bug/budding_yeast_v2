"""Approved synthetic strength domains; canonical biology/real rules are separate."""
import hashlib
import json
import numpy as np

MC_MAX = dict(CLN2=10, CLB5=10, CDC14=10, TEM1=10, CDC15=10,
              CDC6=15, BCK2=15, SIC1=20, CDC20=30, CLN3=8, CLB2=8)
MC_MIN = 2
GAL_CLB2 = (.002, .6)
GAL_MC_BASE = .12
GAL_MC_MIN, GAL_MC_MAX = 2, 10
CLB2_DB_RESIDUAL = (.017/.4, .068/.4)
MEN_TS_RESIDUAL = (.001, .004)  # TEM1/ESP1 residual activity, relative to WT.
NET1_BINDING_RESIDUAL = (.025, .1)
AMPLITUDE_MODES = frozenset({'GAL-OE','GAL-MC','MC','ts','DAmP','pLOF','db'})
GAL_MODES = frozenset({'GAL-OE','GAL-MC'})

RANGE_FINGERPRINT = hashlib.sha256(json.dumps(dict(
    mc_min=MC_MIN, mc_max=MC_MAX, gal_clb2=GAL_CLB2,
    gal_mc=(GAL_MC_BASE,GAL_MC_MIN,GAL_MC_MAX),
    clb2_db=CLB2_DB_RESIDUAL, men_ts=MEN_TS_RESIDUAL,
    net1_binding=NET1_BINDING_RESIDUAL,
    exact_zero_policy='unchanged', mass_mc='continuous_loguniform_dosage'),
    sort_keys=True).encode()).hexdigest()


def copy_support(gene, mode='MC'):
    if mode=='GAL-MC':
        if gene!='CLB2':raise ValueError('GAL-MC is defined only for CLB2')
        return np.arange(GAL_MC_MIN,GAL_MC_MAX+1,dtype=float)
    if gene not in MC_MAX:raise ValueError(f'No MC support for {gene}')
    return np.arange(MC_MIN,MC_MAX[gene]+1,dtype=float)


def copy_quantile(gene,u,mode='MC'):
    values=copy_support(gene,mode)
    weights=1./values;cdf=np.cumsum(weights/weights.sum())
    return float(values[min(int(np.searchsorted(cdf,u)),len(values)-1)])


def sample_copies(gene,rng,mode='MC'):
    values=copy_support(gene,mode);weights=1./values
    return float(rng.choice(values,p=weights/weights.sum()))


def log_uniform(rng,bounds):
    return float(10.**rng.uniform(*np.log10(bounds)))


def log_quantile(u,bounds):
    lo,hi=np.log10(bounds)
    return float(10.**(lo+u*(hi-lo)))
