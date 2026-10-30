"""Compare current MA mutation controls to same-mode canonical real conditions.

Reads the actual candidate pool and cumulative training inputs. No generation,
filtering or training changes. Real min/max is an empirical reference, not a
claim about biological validity outside those finitely many conditions.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from budding_yeast_v2.data.systems.budding.catalog import GROUP_BY_NAME,touch_set
from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
from budding_yeast_v2.data.systems.budding.real_mutants import REAL_MUTANTS
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.budding.distributions.ranges import (
    MC_MAX,MC_MIN,GAL_CLB2,CLB2_DB_RESIDUAL,MEN_TS_RESIDUAL,NET1_BINDING_RESIDUAL)

GAL_KEYS=dict(CLN2='ksn2p',CLN3='C0',CLB2='ksb2p',CLB5='ksb5p',SIC1='ksc1p',CDC6='ksf6p',
              SWI5='ksswip',CDC20='ks20p',CDC14='ks14',NET1='ksnet',TEM1='Tem1T',CDC15='Cdc15T',
              PDS1='kspdsp',ESP1='ESP1T',PPX='ksppx')
MC_KEYS=dict(CLN2='ksn2pp',CLN3='C0',BCK2='B0',CLB2='ksb2p',CLB5='ksb5p',SIC1='ksc1p',
             CDC6='ksf6p',CDC20='ks20p',CDC14='ks14',TEM1='Tem1T',CDC15='Cdc15T')
DB_KEYS=dict(CLB2='kdb2pp',CLB5='kdb5pp',PDS1='kd2pdspp',SIC1='kd3c1',PPX='kdppxpp')
TS_KEYS=dict(CLN2='ksn2pp',CLB2='ksb2p',CDC20='ks20p',CDC15='kkpnetpp',APC_Core='ks20p',
             NET1='kasrent',TEM1='ka15pp',ESP1='kasesp',PPX='kppnetpp',Mcm1='kamcm')
LOF_KEYS=dict(CDC6='ksf6p',CDC20='ks20p',CDC14='ks14',NET1='ksnet',CLB2='ksb2p')
GAL_BOUNDS=dict(CLN2=(.06,.30),CLN3=(1.,5.),CLB2=(.002,1.5),CLB5=(.004,.03),
    SIC1=(.03,.36),CDC6=(.1,.6),CDC20=(.5,12.),CDC14=(.3,1.2),NET1=(.15,.6),
    TEM1=(2.,8.),CDC15=(3.,20.),ESP1=(1.5,6.),PDS1=(.02,.25),SWI5=(.02,.1),PPX=(.2,.6))
GAL_BOUNDS['CLB2']=GAL_CLB2


def support(gene,mode,branch=''):
    """Exact source bounds at the audit date; distinct from midpoint quantiles."""
    if mode=='MC':return float(MC_MIN),float(MC_MAX[gene])
    if mode=='GAL-MC':return .24,1.2
    if mode=='GAL-OE':return GAL_BOUNDS[gene]
    if mode=='DAmP':return .05,.4
    if mode=='pLOF':return .1,.5
    if mode=='APC-A':return 0.,.01
    if mode=='TAB6-1':return NET1_BINDING_RESIDUAL
    if mode=='db':return CLB2_DB_RESIDUAL if gene=='CLB2' else (0.,.01)
    if mode=='ts':
        if gene=='CDC14':return (0.,.01) if branch=='synthesis' else (.05,.35)
        if gene in ('TEM1','ESP1'):return MEN_TS_RESIDUAL
        if gene=='NET1':return NET1_BINDING_RESIDUAL
        if gene in ('CDC20','CDC15','APC_Core'):return 0.,.01
        return 0.,.2
    return None,None


def control(gene,mode,branch=''):
    if mode=='GAL-MC':return 'ksb2p','absolute_parameter'
    if mode=='GAL-OE':return GAL_KEYS[gene],'absolute_parameter'
    if mode=='MC':return MC_KEYS[gene],'copy_multiplier'
    if mode=='db':return DB_KEYS[gene],'residual_WT_fraction'
    if mode in ('DAmP','pLOF'):return LOF_KEYS[gene],'residual_WT_fraction'
    if mode=='ts':
        if gene=='CDC14':return ('ks14' if branch=='synthesis' else 'kppc1'),'residual_WT_fraction'
        return TS_KEYS[gene],'residual_WT_fraction'
    if mode=='APC-A':return 'ka20pp','residual_WT_fraction'
    if mode=='TAB6-1':return 'kasrent','residual_WT_fraction'
    return None,'fixed_assignment'


class CoordinateRNG:
    def __init__(self,u,branch=''):self.u=u;self.branch=branch
    def random(self):return .25 if self.branch=='synthesis' else .75
    def rand(self):return self.random()
    def uniform(self,low=0.,high=1.):return low+(high-low)*self.u


def reference_rows():
    """Keep named alleles apart; explicitly label parameter-equivalent aliases."""
    defaults=BuddingSystem().defaults();result=[]
    for identity,(name,override) in REAL_MUTANTS.items():
        p={**defaults,**override}
        for gene in GAL_KEYS:
            if re.search(r'GAL-'+re.escape(gene)+r'(?![A-Za-z0-9])',name) or (gene=='SIC1' and 'GAL-SIC1db' in name):
                if identity=='104':continue  # Timed induction cannot calibrate a constant GAL control.
                subtype='GAL_with_8copy_boost' if identity=='50' and gene=='CLB2' else 'ordinary_GAL'
                result.append(dict(identity=identity,name=name,gene=gene,mode='GAL-MC' if subtype=='GAL_with_8copy_boost' else 'GAL-OE',branch='',
                                   strength=p[GAL_KEYS[gene]],kind=subtype))
        match=re.search(r'multi-copy\s+(?!GAL-)([A-Z][A-Z0-9]*)',name)
        if match:
            gene=match.group(1)
            if gene in MC_KEYS:
                key=MC_KEYS[gene]
                result.append(dict(identity=identity,name=name,gene=gene,mode='MC',branch='',
                                   strength=p[key]/defaults[key],kind='named_MC'))
        for gene in ('CLB2','CLB5','PDS1','SIC1'):
            if re.search(re.escape(gene)+r'-?db',name):
                key=DB_KEYS[gene]
                result.append(dict(identity=identity,name=name,gene=gene,mode='db',branch='',
                                   strength=p[key]/defaults[key],kind='named_db'))
        for gene in TS_KEYS.keys()|{'CDC14'}:
            token='apc-ts' if gene=='APC_Core' else gene.lower()+'-ts'
            if token in name.lower():
                branch=('synthesis' if 'ks14' in override else 'activity') if gene=='CDC14' else ''
                key,_=control(gene,'ts',branch)
                result.append(dict(identity=identity,name=name,gene=gene,mode='ts',branch=branch,
                                   strength=p[key]/defaults[key],kind='named_ts'))
        if 'TAB6-1' in name:
            result.append(dict(identity=identity,name=name,gene='NET1',mode='TAB6-1',branch='',
                               strength=p['kasrent']/defaults['kasrent'],kind='named_TAB6'))
        if 'APC-A' in name:
            for gene in ('APC_Core','CDC20'):
                result.append(dict(identity=identity,name=name,gene=gene,mode='APC-A',branch='',
                    strength=p['ka20pp']/defaults['ka20pp'],kind='named_APC-A' if gene=='APC_Core' else 'parameter_equivalent_APC_proxy'))
        if 'CLB1 clb2Δ' in name:
            result.append(dict(identity=identity,name=name,gene='CLB2',mode='pLOF',branch='',
                               strength=p['ksb2p']/defaults['ksb2p'],kind='lumped_CLB1_remaining'))
    # Fixed assignments are checked numerically and not claimed to be independent
    # experimental calibrations of every synthetic functional module.
    backend=BuddingSystem()
    for gene,group in GROUP_BY_NAME.items():
        if gene=='MDT':continue
        for mode in group['modes']:
            if control(gene,mode)[0] is not None:continue
            changes={};backend.apply_gene_mutation(gene,mode,1.,defaults,defaults.copy(),changes)
            for identity,(name,override) in REAL_MUTANTS.items():
                p={**defaults,**override}
                # Vi20 is an unused legacy alias; the ODE uses Vi20_active.
                if all(np.isclose(p.get('Vi20_active' if k=='Vi20' else k,np.nan),value,rtol=1e-8,atol=1e-10) for k,value in changes.items()):
                    result.append(dict(identity=identity,name=name,gene=gene,mode=mode,branch='',strength=1.,kind='matching_fixed_parameter_assignment'))
    return result


def nominal(gene,mode,branch,profile='mutant_aware',size=4096):
    backend=BuddingSystem(profile,profile);defaults=backend.defaults();key,kind=control(gene,mode,branch)
    if key is None:return np.array([1.]),[]
    values=[]
    for u in (np.arange(size)+.5)/size:
        alpha=backend.sample_scaling_factor_sobol(mode,gene,float(u))
        overrides={}
        backend.apply_gene_mutation(gene,mode,alpha,defaults,defaults.copy(),overrides,rng=CoordinateRNG(float(u),branch))
        values.append(overrides[key] if kind=='absolute_parameter' else overrides[key]/defaults[key])
    return np.asarray(values),sorted(overrides)


def index_components(params,pairs,indices):
    groups=defaultdict(list);ks14=PARAM_KEYS.index('ks14');default=BuddingSystem().defaults()['ks14']
    for i in indices:
        for gene,mode in json.loads(str(pairs[i])):
            branch=('synthesis' if params[i,ks14]<.1*default else 'activity') if (gene,mode)==('CDC14','ts') else ''
            groups[gene,mode,branch].append(int(i))
    return {key:np.asarray(rows,dtype=int) for key,rows in groups.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--moments',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    refs=reference_rows();pd.DataFrame(refs).to_csv(args.output/'real_components.csv',index=False)
    defaults=BuddingSystem().defaults();tables=[];no_ref=[];nominal_cache={};mass_differences=[]
    specs=[]
    for gene,group in GROUP_BY_NAME.items():
        if gene=='MDT':continue
        for mode in group['modes']:
            for branch in (('synthesis','activity') if (gene,mode)==('CDC14','ts') else ('',)):
                key,unit=control(gene,mode,branch);observed=[x for x in refs if (x['gene'],x['mode'],x['branch'])==(gene,mode,branch)]
                q,changes=nominal(gene,mode,branch)
                values=sorted(set(float(x['strength']) for x in observed));nominal_cache[gene,mode,branch]=q
                row=dict(gene=gene,mode=mode,branch=branch,parameter=key,unit=unit,
                    current_support_min=support(gene,mode,branch)[0],current_support_max=support(gene,mode,branch)[1],
                    nominal_qmin=float(q.min()),nominal_qmax=float(q.max()),nominal_p50=float(np.median(q)),
                    real_values=json.dumps(values),real_min=min(values) if values else None,
                    real_max=max(values) if values else None,reference_conditions=len(observed),
                    real_ids=';'.join(x['identity'] for x in observed),reference_kinds=';'.join(sorted({x['kind'] for x in observed})),
                    nominal_fraction_above_real_max=float((q>max(values)+1e-10).mean()) if values else None,
                    nominal_fraction_below_real_min=float((q<min(values)-1e-10).mean()) if values else None,
                    upper_to_real_max=float(q.max()/max(values)) if values and max(values)>0 else None)
                specs.append(row)
                if not values:no_ref.append(dict(gene=gene,mode=mode,branch=branch))
                mass,_=nominal(gene,mode,branch,'mass')
                if not np.allclose(q,mass):mass_differences.append(dict(gene=gene,mode=mode,branch=branch,MA_min=q.min(),MA_max=q.max(),MASS_min=mass.min(),MASS_max=mass.max(),note='effective control distribution, not proposal alpha'))
    pd.DataFrame(specs).to_csv(args.output/'nominal_vs_real.csv',index=False)
    pd.DataFrame(mass_differences).to_csv(args.output/'mass_profile_differences.csv',index=False)
    with np.load(args.moments,allow_pickle=False) as z:high=z['legacy_high'];peaks=z['maxima'];train=z['train']
    aggregate=[]
    for scope,path in (('pool',args.pool),('train',args.dataset)):
        print(f'Reading {scope}: {path}',flush=True)
        with np.load(path,allow_pickle=False) as z:
            params=z['params'];pairs=z['mutation_pairs_json'];indices=np.arange(len(params)) if scope=='pool' else train
        groups=index_components(params,pairs,indices)
        outside_support=np.zeros(len(params),dtype=bool)
        any_above=np.zeros(len(params),dtype=bool);any_unreferenced=np.zeros(len(params),dtype=bool)
        expression_above=np.zeros(len(params),dtype=bool);stronger_lof=np.zeros(len(params),dtype=bool)
        for spec in specs:
            gene,mode,branch=spec['gene'],spec['mode'],spec['branch'];rows=groups.get((gene,mode,branch),np.array([],dtype=int))
            key,unit=control(gene,mode,branch)
            values=np.ones(len(rows)) if key is None else params[rows,PARAM_KEYS.index(key)]/(1. if unit=='absolute_parameter' else defaults[key])
            observed=[x for x in refs if (x['gene'],x['mode'],x['branch'])==(gene,mode,branch)]
            row=dict(scope=scope,**spec,count=len(rows),actual_min=float(values.min()) if len(rows) else None,
                     actual_max=float(values.max()) if len(rows) else None,actual_p50=float(np.median(values)) if len(rows) else None)
            lo,hi=support(gene,mode,branch)
            if lo is not None:
                outside=(values<lo-1e-10)|(values>hi+1e-10)
                if mode=='MC':outside|=np.abs(values-np.round(values))>1e-10
                outside_support[rows[outside]]=True
                row['outside_current_support']=int(outside.sum())
            if observed:
                limit=max(x['strength'] for x in observed)
                above=values>limit+1e-9
                any_above[rows[above]]=True
                if mode in ('MC','GAL-OE','GAL-MC'):expression_above[rows[above]]=True
                below=values<min(x['strength'] for x in observed)-1e-9
                if unit=='residual_WT_fraction':stronger_lof[rows[below]]=True
                row.update(above_real_max=int(above.sum()),above_real_max_fraction=float(above.mean()) if len(rows) else None)
                row.update(below_real_min=int(below.sum()),below_real_min_fraction=float(below.mean()) if len(rows) else None)
            else:any_unreferenced[rows]=True
            if scope=='train':
                row.update(high_trajectory_count=int(high[rows].sum()),high_trajectory_fraction=float(high[rows].mean()) if len(rows) else None)
            tables.append(row)
        summary=dict(scope=scope,samples=len(indices),with_component_above_real_max=int(any_above[indices].sum()),
                     outside_current_strength_support=int(outside_support[indices].sum()),
                     with_expression_above_real_max=int(expression_above[indices].sum()),
                     with_stronger_LOF_than_real_min=int(stronger_lof[indices].sum()),
                     with_unreferenced_component=int(any_unreferenced[indices].sum()))
        if scope=='train':
            summary['cross_tab_expression_above_vs_high']=pd.crosstab(pd.Series(expression_above[indices],name='above_real_expression_max'),pd.Series(high[indices],name='high_trajectory')).to_dict()
            sample_rows=pd.DataFrame(dict(row=indices,high=high[indices],above_real_parameter_max=any_above[indices],above_real_expression_max=expression_above[indices],stronger_LOF_than_real_min=stronger_lof[indices],unreferenced_component=any_unreferenced[indices]))
            sample_rows.to_csv(args.output/'sample_parameter_flags.csv',index=False)
        aggregate.append(summary)
        print(json.dumps(summary),flush=True)
    pd.DataFrame(tables).to_csv(args.output/'empirical_pool_train.csv',index=False)
    (args.output/'summary.json').write_text(json.dumps(dict(scopes=aggregate,without_same_mode_real_reference=no_ref,
        mode_rows=len(specs),note='Parameter-component associations are not causal effects; MC finite capacity/dedup and equal allele quotas can change marginal strength frequencies.'),indent=2)+'\n')


if __name__=='__main__':main()
