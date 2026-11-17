"""ERACS Algorithm1 SNR-triggered weighted OMP, trajectory-gradient adapter.

SNR uses covariance *spectral norm* as in the appendix. The trigger is <tau,
matching Algorithm1 and the theory (the experimental 'exceeds' sentence differs).
Batch-OMP grouping is explicit; original sample gradients can be used with size1.
"""
import numpy as np
from scipy.optimize import nnls


def gradient_snr(gradients,weights=None,eps=1e-12):
    g=np.asarray(gradients,dtype=np.float64)
    if g.ndim!=2 or not len(g) or not np.isfinite(g).all():raise ValueError('Invalid SNR gradients')
    if weights is not None:g=g*np.asarray(weights)[:,None]
    mean=g.mean(0);centered=g-mean
    covariance=centered.T@centered/max(1,len(g)-1)
    variance=max(0.,float(np.linalg.eigvalsh(covariance)[-1]))
    value=float(np.linalg.norm(mean)/max(np.sqrt(variance),eps))
    if not np.isfinite(value):raise FloatingPointError('Nonfinite gradient SNR')
    return value


def weighted_omp(gradients,budget,*,group_size=100,seed=42,tolerance=1e-8,ridge=1e-6):
    g=np.asarray(gradients,dtype=np.float64);n,d=g.shape
    if not 0<budget<=n or group_size<1 or not np.isfinite(g).all():raise ValueError('Invalid OMP domain/budget')
    order=np.random.default_rng(seed).permutation(n)
    groups=[order[i:i+group_size] for i in range(0,n,group_size)]
    means=np.stack([g[rows].mean(0) for rows in groups]);target=g.mean(0)
    selected=[];blocks=[];available=np.ones(len(groups),bool);residual=target.copy();count=0
    stationary_fill=0
    while count<budget:
        norm=np.linalg.norm(means,axis=1)
        scores=(means@residual)/np.maximum(norm,1e-30);scores[~available]=-np.inf
        index=int(np.argmax(scores));available[index]=False
        if np.linalg.norm(residual)<=tolerance:stationary_fill+=1
        rows=groups[index][:budget-count];blocks.append(rows);selected.append(g[rows].mean(0));count+=len(rows)
        a=np.stack(selected).T
        augmented=np.vstack((a,np.sqrt(ridge)*np.eye(len(selected))))
        coefficient,_=nnls(augmented,np.r_[target,np.zeros(len(selected))],maxiter=30*len(selected))
        residual=target-a@coefficient
    indices=np.concatenate(blocks);weights=np.concatenate([
        np.full(len(rows),len(indices)*c/len(rows)) for rows,c in zip(blocks,coefficient)])
    if not np.isfinite(weights).all():raise FloatingPointError('Nonfinite OMP weights')
    if not weights.sum()>0:
        # Exact zero full gradient permits the zero estimator. Retain that
        # explicit estimator; never silently turn it into uniform training.
        zero=True
    else:zero=False
    return indices,weights,dict(group_size=group_size,selected_groups=len(blocks),
        gradient_residual=float(np.linalg.norm(residual)),target_gradient_norm=float(np.linalg.norm(target)),
        stationary_budget_fill_groups=stationary_fill,zero_weight_rows=int((weights==0).sum()),
        zero_gradient_estimator=zero,objective='nonnegative regularized mean-gradient matching')
