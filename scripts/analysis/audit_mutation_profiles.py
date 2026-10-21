"""Audit retained mutation rules and the authorized pruning of synthetic catalogs.

The archived catalogs contain intentionally removed controls. Compare amplitudes
and assignments for surviving cells; compare graphs to induced archived graphs.
Whole-function AST equality and historical candidate streams no longer apply.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.systems.budding import catalog,topology
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.budding.samplers import common


def canonical(value):
    if isinstance(value,np.ndarray):return canonical(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return sorted([(repr(k),canonical(v)) for k,v in value.items()])
    if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
    if isinstance(value,set):return sorted([canonical(v) for v in value],key=repr)
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(canonical(value),allow_nan=True).encode()).hexdigest()


def identical(a,b):return fingerprint(a)==fingerprint(b)


def audit(seeds,draws):
    results=[]
    def add(name,ok,**detail):results.append(dict(check=name,equal=bool(ok),**detail))
    cells=[(g,m) for g in catalog.GENE_NAMES for m in catalog.GROUP_BY_NAME[g]['modes']]
    legal=set(catalog.GENE_NAMES)
    add('current_catalog_size',len(legal)==20 and len(cells)==65)
    for profile,legacy in [('mass','mass_dataset_generator'),('mutant_aware','mutant_aware')]:
        old=importlib.import_module('budding_yeast_v2.data.'+legacy)
        backend=BuddingSystem(profile,profile)
        failures=[];amplitude_checks=0;assignment_checks=0
        defaults=backend.defaults()
        for seed in seeds:
            for gene,mode in cells:
                np.random.seed(seed)
                a=[old.sample_scaling_factor(mode,gene) for _ in range(draws)]
                state_a=np.random.get_state()
                np.random.seed(seed)
                b=[backend.sample_scaling_factor(mode,gene) for _ in range(draws)]
                if not identical((a,state_a),(b,np.random.get_state())):failures.append([seed,gene,mode,'draw'])
                amplitude_checks+=draws
                for alpha in (0.,1e-4,.01,.2,1.,2.,30.):
                    np.random.seed(seed);pa=defaults.copy();oa={}
                    old.apply_gene_mutation(gene,mode,alpha,defaults,pa,oa)
                    sa=np.random.get_state()
                    np.random.seed(seed);pb=defaults.copy();ob={}
                    backend.apply_gene_mutation(gene,mode,alpha,defaults,pb,ob)
                    if not identical((pa,oa,sa),(pb,ob,np.random.get_state())):failures.append([seed,gene,mode,alpha])
                    assignment_checks+=1
        add(profile+'.all_cells_draws_and_assignments',not failures,cells=len(cells),amplitude_checks=amplitude_checks,
            assignment_checks=assignment_checks,failures=failures[:10])
        # The only surviving metadata changes are canonical MA initial keys,
        # normalized gene weights and removal of the two DAmP modes.
        failures=[]
        weight_sum=sum(row['P_bio_G'] for row in old.BINDING_GROUPS.values() if row['name'] in legal)
        for key,b in catalog.BINDING_GROUPS.items():
            a=dict(old.BINDING_GROUPS[key])
            if profile=='mass' and a['name'] in ('TEM1','CDC15'):
                a['params']=a['params']+['init_Tem1' if a['name']=='TEM1' else 'init_Cdc15']
            modes={m:w for m,w in a['modes'].items() if (a['name'],m) in cells}
            if a['name'] in ('TEM1','CDC15'):
                total=sum(modes.values());modes={m:w/total for m,w in modes.items()}
            expected=dict(a,modes=modes)
            if not np.isclose(a['P_bio_G']/weight_sum,b['P_bio_G']):failures.append([key,'weight'])
            for field in expected.keys()-{'P_bio_G'}:
                if not identical(expected[field],b[field]):failures.append([key,field])
        add(profile+'.pruned_catalog',not failures,failures=failures)
    old=importlib.import_module('budding_yeast_v2.data.mutant_aware')
    add('module.induced_graph',identical(topology.MODULE_EDGES,
        [e for e in old.MODULE_EDGES if e[0] in legal and e[1] in legal]))
    add('module.retained_touch_sets',identical(catalog.TOUCH,
        {k:v for k,v in old.TOUCH.items() if k in cells}))
    add('module.retained_roles',identical(catalog.GENE_ROLE,
        {k:v for k,v in old.GENE_ROLE.items() if k in legal}))
    add('module.mode_class',identical(catalog.MODE_CLASS,
        {k:v for k,v in old.MODE_CLASS.items() if k in {m for g,m in cells}}))
    grid=list(cells);np.random.RandomState(20260831).shuffle(grid)
    add('module.current_coverage_grid',identical(common._L1_GRID,grid))
    # Run the archived selector against a local restricted catalog. This changes
    # neither the archived module nor files, and tests selector mathematics.
    original_namespace=dict(old._sample_mode.__globals__,GROUP_BY_NAME=catalog.GROUP_BY_NAME)
    sample_mode=types.FunctionType(old._sample_mode.__code__,original_namespace,
                                  argdefs=old._sample_mode.__defaults__)
    sample_mode.__kwdefaults__=old._sample_mode.__kwdefaults__
    mode_checks=0
    for frequency in ('class','flat','role','s5'):
        original_namespace['MODE_FREQ']=common.MODE_FREQ=frequency
        for seed in seeds:
            for gene in catalog.GENE_NAMES:
                np.random.seed(seed);a=[sample_mode(gene) for _ in range(draws)];sa=np.random.get_state()
                np.random.seed(seed);b=[common._sample_mode(gene) for _ in range(draws)]
                add(f'mode.{frequency}.{seed}.{gene}',identical((a,sa),(b,np.random.get_state())))
                mode_checks+=draws
    for seed in seeds:
        for explicit in (False,True):
            pairs=[('CLN2','KO'),('CLB2','MC'),('CLB2','db'),('CDC20','ts'),('CDC6','DAmP')]
            np.random.seed(seed);r1=np.random.RandomState(seed)
            a=old._resolve_conflicts(pairs,rng=r1 if explicit else None);sa=(np.random.get_state(),r1.get_state())
            np.random.seed(seed);r2=np.random.RandomState(seed)
            b=common._resolve_conflicts(pairs,rng=r2 if explicit else None)
            add(f'conflict.{seed}.{explicit}',identical((a,sa),(b,(np.random.get_state(),r2.get_state()))))
        np.random.seed(seed);a=old.QuotaPicker(['a','b','c'],[.4,.35,.25])
        aa=[a.pick([1.,3.,2.] if i%5==0 else None) for i in range(draws*4)];sa=np.random.get_state()
        np.random.seed(seed);b=common.QuotaPicker(['a','b','c'],[.4,.35,.25])
        bb=[b.pick([1.,3.,2.] if i%5==0 else None) for i in range(draws*4)]
        add(f'quota_picker.{seed}',identical((aa,a.counts,sa),(bb,b.counts,np.random.get_state())))
    for source,target in [('bams_dataset_generator','bams'),('mass_wt_blind_sampler','wt_blind')]:
        a=importlib.import_module('budding_yeast_v2.data.'+source)
        b=importlib.import_module('budding_yeast_v2.data.systems.budding.samplers.'+target)
        expected=[edge for edge in a.MODULE_EDGES if edge[0] in legal and edge[1] in legal]
        add(source+'.induced_graph',identical(expected,b.MODULE_EDGES))
        add(source+'.coverage_cells',identical(set(b._L1_GRID),set(cells)))
    a=importlib.import_module('budding_yeast_v2.data.fission_random_sampler')
    b=importlib.import_module('budding_yeast_v2.data.systems.fission.catalog')
    add('fission.catalog_unchanged',identical(a.SAMPLING_MODULES,b.SAMPLING_MODULES))
    manifest=json.loads((ROOT/'data/systems/archived_generators.json').read_text())
    add('all_31_archived_generators_unchanged',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items()))
    return dict(seeds=seeds,draws_per_cell=draws,mode_draw_checks=mode_checks,
                checks=len(results),passed=sum(r['equal'] for r in results),results=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,7,42])
    parser.add_argument('--draws',type=int,default=32)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.draws<1:parser.error('--draws must be positive')
    report=audit(args.seeds,args.draws)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    for row in report['results']:
        if not row['equal']:print(json.dumps(row,ensure_ascii=False),flush=True)
    return 0 if report['checks']==report['passed'] else 1


if __name__=='__main__':sys.exit(main())
