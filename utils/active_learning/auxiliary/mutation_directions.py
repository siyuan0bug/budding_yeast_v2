"""Legal continuous and integer-copy probes derived from production mutation laws.

Probe trajectories are MODEL predictions only. No probe is simulated or added to
the pool; oracle queries always reference existing original candidate IDs.
"""
import numpy as np
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.budding.simulation import prepare_initial_conditions
from ..strategies.response import _strength_curve,_CoordinateRNG
from .views import pairs

STEP=.01
CONTINUOUS=('GAL-OE','ts','DAmP','pLOF','db','APC-A','TAB6-1')


def _inputs(override,backend):
    defaults=backend.defaults();p=np.asarray([override.get(k,defaults[k]) for k in backend.PARAM_KEYS],dtype=np.float64)
    y=prepare_initial_conditions(override)[1]
    return p,y


def probes(inputs,indices,round_index=0):
    backend=BuddingSystem();defaults=backend.defaults();lookup={k:i for i,k in enumerate(backend.PARAM_KEYS)}
    result=[]
    for i in indices:
        original=dict(inputs.overrides[i]);choices=[]
        for gene,mode in pairs(inputs.metadata[i]):
            if mode not in CONTINUOUS+('MC','GAL-MC'):continue
            branch=('synthesis' if float(inputs.params[i,lookup['ks14']])<.1*defaults['ks14'] else 'activity') if (gene,mode)==('CDC14','ts') else ''
            keys,values,cdf=_strength_curve(gene,mode,branch)
            if not keys:continue
            current=float(inputs.params[i,lookup[keys[0]]])
            if mode in ('MC','GAL-MC'):
                from budding_yeast_v2.data.systems.budding.distributions.ranges import copy_support
                copies=copy_support(gene,mode);j=int(np.argmin(abs(values-current)))
                a=max(0,j-1);b=min(len(values)-1,j+1)
                if a==b:continue
                coordinates=(float(copies[a]),float(copies[b]));delta=(copies[b]-copies[a])/max(1,copies[-1]-copies[0])
                kind='integer_secant';u=float(copies[j])
            else:
                u=float(np.interp(current,values,cdf));a=max(0,u-STEP);b=min(1,u+STEP)
                if b<=a:continue
                coordinates=(a,b);delta=b-a;kind='continuous_tangent_fd'
            ends=[]
            for coordinate in coordinates:
                alpha=coordinate if kind=='integer_secant' else backend.sample_scaling_factor_sobol(mode,gene,coordinate)
                override=original.copy();backend.apply_gene_mutation(gene,mode,alpha,defaults,defaults.copy(),override,
                    rng=_CoordinateRNG(coordinate,branch))
                ends.append(_inputs(override,backend))
            dp=(ends[1][0]-ends[0][0])/delta;dy=(ends[1][1]-ends[0][1])/delta
            if not np.any(dp) and not np.any(dy):continue
            choices.append(dict(gene=gene,mode=mode,branch=branch,kind=kind,u=u,delta=float(delta),
                p_minus=ends[0][0],p_plus=ends[1][0],y_minus=ends[0][1],y_plus=ends[1][1],dp=dp,dy=dy))
        if choices:
            row=choices[(int(inputs.ids[i])+round_index)%len(choices)]
            row['center_p'],row['center_y']=_inputs(original,backend)
            row['flags']={key:bool(original.get(key,False)) for key in ('is_sbf_ca','is_mutant_104')}
            row['row']=int(i);result.append(row)
    return result
