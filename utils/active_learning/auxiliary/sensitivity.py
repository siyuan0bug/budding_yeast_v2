"""Response consistency with the canonical ODE, away from hybrid events.

Continuous directions: finite-CDF model sensitivity vs exact RHS JVP.
Integer copies: finite secant vs the exact finite difference of the RHS.
Probe predictions never query the oracle; only original pool IDs are acquired.
"""
import numpy as np
import torch
import torch.nn.functional as F
from .canonical_rhs import rhs,SOURCE_SHA256
from .mutation_directions import probes,STEP
from .phenotype import training_descriptors
from budding_yeast_v2.utils.runtime_logging import ProgressLog

SETTINGS=dict(batch_size=64,event_radius_steps=5,minimum_valid_fraction=.25,
              exploration_fraction=.1,continuous_cdf_half_step=STEP,
              numerical_derivative='central time differences on 500 physical time points',
              scope='continuous intervals only; no saltation or true event-time sensitivity claim',
              rhs_source_sha256=SOURCE_SHA256)


def event_safe_mask(trajectories):
    """Union of guard/reset neighborhoods and event-count mismatches of probes."""
    marks=[];counts=[]
    for y in trajectories:
        guard=torch.stack((y[:,2]-.3,y[:,2]+y[:,3]-.2,y[:,33]-1.,y[:,35]-1.),1)
        left,right=guard[:,:,:-1],guard[:,:,1:]
        down=(left>0)&(right<=0);up=(left<0)&(right>=0)
        hits=torch.cat((down[:,:2],up[:,2:]),1)
        hits=F.pad(hits,(1,0));counts.append(hits.cumsum(-1))
        resets=(torch.diff(y[:,0],dim=-1)<-.1)|(torch.diff(y[:,33],dim=-1)<-.5)
        resets|=(torch.diff(y[:,34],dim=-1)<-.3)|(torch.diff(y[:,35],dim=-1)<-.3)
        resets|=(torch.diff(y[:,36:39],dim=-1).abs()>.3).any(1)
        marks.append(hits.any(1)|F.pad(resets,(1,0)))
    bad=torch.stack(marks).any(0)
    # Do not interpret a finite difference spanning shifted events as a smooth tangent.
    for value in counts[1:]:bad|=(value!=counts[0]).any(1)
    radius=SETTINGS['event_radius_steps']
    bad=F.max_pool1d(bad[:,None].float(),2*radius+1,stride=1,padding=radius)[:,0].bool()
    valid=~bad;valid[:,0]=False;valid[:,-1]=False
    return valid


def residual_scores(center,minus,plus,p,dp,dy,delta,discrete,flags,t_max,scale):
    """Physical tensors B,39,T; parameters B,141; no observed oracle outputs."""
    dtype=center.dtype;device=center.device;steps=center.shape[-1]
    times=torch.linspace(0,t_max,steps,dtype=dtype,device=device)
    sensitivity=(plus-minus)/delta[:,None,None]
    dt=t_max/(steps-1)
    ds=torch.zeros_like(sensitivity);ds[:,:,1:-1]=(sensitivity[:,:,2:]-sensitivity[:,:,:-2])/(2*dt)
    base=center.transpose(1,2);tangent=sensitivity.transpose(1,2)
    right=torch.empty_like(base)
    continuous=torch.where(~discrete)[0]
    if len(continuous):
        local_flags={k:v[continuous] for k,v in flags.items()}
        _,value=torch.autograd.functional.jvp(lambda y,parameters:rhs(y,parameters,times,local_flags),
            (base[continuous],p[continuous]),(tangent[continuous],dp[continuous]),create_graph=False)
        right[continuous]=value
    indices=torch.where(discrete)[0]
    if len(indices):
        local_flags={k:v[indices] for k,v in flags.items()}
        # Caller places endpoint parameters in flags' private numeric fields.
        fp=rhs(plus[indices].transpose(1,2),flags['_p_plus'][indices],times,local_flags)
        fm=rhs(minus[indices].transpose(1,2),flags['_p_minus'][indices],times,local_flags)
        right[indices]=(fp-fm)/delta[indices,None,None]
    right=right.transpose(1,2)
    mask=event_safe_mask((center,minus,plus))
    if flags['is_mutant_104'].any():
        mask&=~(flags['is_mutant_104'][:,None]&((times-180.).abs()<SETTINGS['event_radius_steps']*dt))
    finite=torch.isfinite(ds).all(1)&torch.isfinite(right).all(1)
    mask&=finite
    fraction=mask.float().mean(1)
    # Componentwise relative rate plus train-derived concentration/minute floor.
    denominator=right.abs()+sensitivity.abs()/t_max+scale[None,:,None]/t_max
    normalized=(ds-right)/denominator
    normalized=torch.where(mask[:,None],normalized,torch.zeros_like(normalized))
    rate=normalized.square().sum((1,2))/(mask.sum(1).clamp(min=1)*center.shape[1])
    initial=((sensitivity[:,:,0]-dy)/(.1+dy.abs())).square().mean(1)
    score=torch.sqrt(rate+initial)
    eligible=(fraction>=SETTINGS['minimum_valid_fraction'])&torch.isfinite(score)
    return torch.where(eligible,score,torch.zeros_like(score)),fraction,initial


def score_pool(view,model,device,round_index=0):
    true_phi,_=training_descriptors(view)
    scale=np.maximum(np.expm1(np.median(true_phi[:,:39],axis=0)),.1)
    scale=torch.as_tensor(scale,dtype=torch.float64,device=device)
    n=len(view.pool.ids);scores=np.zeros(n);valid=np.zeros(n);initial=np.zeros(n)
    coordinate=np.zeros(n);step=np.zeros(n);branches=np.full(n,'',dtype='<U12')
    kinds=np.full(n,'no_legal_direction',dtype='<U28');genes=np.full(n,'',dtype='<U24');modes=np.full(n,'',dtype='<U16')
    original=next(model.parameters()).device;was_training=model.training
    progress=ProgressLog('ODE sensitivity consistency',n)
    try:
        model.to(device).eval()
        for start in range(0,n,SETTINGS['batch_size']):
            end=min(start+SETTINGS['batch_size'],n);rows=probes(view.pool,range(start,end),round_index)
            if rows:
                def tensor(key):return torch.as_tensor(np.asarray([row[key] for row in rows]),device=device,dtype=torch.float64)
                def predict(p,y0):
                    x,parameters=view.normalization.encode(p,y0,view.steps,device)
                    with torch.no_grad():return view.normalization.decode(model.forward_ic_time(x,parameters).double())
                cp,cy=np.asarray([r['center_p'] for r in rows]),np.asarray([r['center_y'] for r in rows])
                pm,ym=np.asarray([r['p_minus'] for r in rows]),np.asarray([r['y_minus'] for r in rows])
                pp,yp=np.asarray([r['p_plus'] for r in rows]),np.asarray([r['y_plus'] for r in rows])
                center=predict(cp,cy);minus=predict(pm,ym);plus=predict(pp,yp)
                flags={key:torch.tensor([r['flags'][key] for r in rows],device=device,dtype=torch.bool)
                       for key in ('is_sbf_ca','is_mutant_104')}
                flags.update(_p_minus=tensor('p_minus'),_p_plus=tensor('p_plus'))
                discrete=torch.tensor([r['kind']=='integer_secant' for r in rows],device=device)
                values,ratio,ic=residual_scores(center,minus,plus,tensor('center_p'),tensor('dp'),tensor('dy'),
                    tensor('delta'),discrete,flags,view.t_max,scale)
                ids=np.asarray([r['row'] for r in rows]);scores[ids]=values.detach().cpu();valid[ids]=ratio.detach().cpu()
                initial[ids]=torch.nan_to_num(ic).detach().cpu()
                for row in rows:
                    i=row['row'];kinds[i]=row['kind'];genes[i]=row['gene'];modes[i]=row['mode']
                    coordinate[i]=row['u'];step[i]=row['delta'];branches[i]=row['branch']
            progress.update(end)
    finally:model.to(original).train(was_training)
    return scores,dict(valid_time_fraction=valid,initial_condition_error=initial,direction_kind=kinds,gene=genes,mode=modes,
                       coordinate=coordinate,coordinate_step=step,allele_branch=branches)


def select_scores(scores,budget,seed):
    if not np.isfinite(scores).all() or (scores<0).any():raise ValueError('Nonfinite sensitivity ranking')
    rng=np.random.RandomState(seed);order=rng.permutation(len(scores));order=order[np.argsort(-scores[order],kind='stable')]
    exploit=min(budget-int(budget*SETTINGS['exploration_fraction']),int((scores>0).sum()))
    chosen=order[:exploit];remaining=np.setdiff1d(np.arange(len(scores)),chosen)
    random=rng.choice(remaining,budget-len(chosen),replace=False)
    return np.r_[chosen,random].astype(np.int64),exploit
