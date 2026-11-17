"""Independent VeSSAL and leverage selection from their published update rules.

Reference repositories are development evidence only. Runtime never imports,
executes, reads or requires the reference repositories.
"""
import math
import numpy as np
import torch


def source_record(name):
    return dict(algorithm=name,implementation='independent_project_implementation',
                runtime_reference_dependency=False)


@torch.no_grad()
def vessal_penultimate(features,budget,seed,device='cpu'):
    x=torch.as_tensor(np.asarray(features),dtype=torch.float32,device=device)
    budget=min(int(budget),len(x))
    d=x.shape[1]
    covariance=torch.zeros(d,d,device=device)
    inverse=100*torch.eye(d,device=device)
    rng=np.random.RandomState(seed)
    selected,skipped=[],[]
    for i,vector in enumerate(x):
        column=vector[:,None]
        gain=(column.T@inverse@column).abs().squeeze()
        covariance.mul_(i/(i+1)).add_(column@column.T,alpha=1/(i+1))
        rate=(budget-len(selected))/(len(x)-i)
        trace=torch.trace(covariance@inverse)
        probability=abs(rate/float(trace))*float(gain) if trace!=0 else float('nan')
        if rng.rand()<probability:
            selected.append(i)
            if len(selected)>=budget:break
            projected=inverse@column
            inverse.sub_(projected@projected.T/(1+gain))
        else:skipped.append(i)
    missing=budget-len(selected)
    if missing:
        remaining=np.setdiff1d(np.arange(len(x)),selected)
        selected.extend(rng.permutation(remaining)[:missing].tolist())
    return np.asarray(selected,dtype=np.int64),dict(source=source_record('VeSSAL'),
        author_branch='penultimate',early_stop=True,fill_random=True,
        random_fill_count=missing,skipped_count=len(skipped))


@torch.no_grad()
def leverage(features,budget,seed,device='cpu',block_size=1024,step_size=2):
    features=np.asarray(features,dtype=np.float32)
    blocks=[np.arange(i,min(i+block_size,len(features))) for i in range(0,len(features),block_size)]
    raw=np.asarray([len(b) for b in blocks])*budget/len(features)
    quotas=np.floor(raw).astype(int)
    quotas[np.argsort(-(raw-quotas),kind='stable')[:budget-quotas.sum()]]+=1
    selected=[]
    for block_index,(block,quota) in enumerate(zip(blocks,quotas)):
        if not quota:continue
        if quota==len(block):selected.extend(block);continue
        x=torch.as_tensor(features[block],device=device)
        step=min(step_size,quota)
        local_seed=int(np.random.SeedSequence([seed,block_index]).generate_state(1)[0])
        rng=np.random.default_rng(local_seed)
        local=rng.choice(len(block),size=step,replace=False).tolist()
        gram=x[local].T@x[local]
        regularizer=1e-5*torch.eye(x.shape[1],device=device)
        for _ in range(1,math.ceil(quota/step)):
            covariance=gram/len(local)+regularizer
            lower=torch.linalg.cholesky(covariance)
            whitened=torch.linalg.solve_triangular(lower,x.T,upper=False)
            scores=whitened.square().sum(0)
            scores[local]=-torch.inf
            next_ids=torch.argsort(-scores)[:step].tolist()
            local.extend(next_ids)
            gram.add_(x[next_ids].T@x[next_ids])
        selected.extend(block[np.asarray(local[:quota],dtype=int)])
    return np.asarray(selected,dtype=np.int64),dict(source=source_record('LevDataDiversity'),
        block_size=block_size,step_size=step_size,regularization=1e-5,exact_block_quotas=quotas.tolist())
