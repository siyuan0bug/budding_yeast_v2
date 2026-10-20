"""Audit both mutation profiles, amplitude draws and biological module catalogs."""
import argparse
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.systems.budding import catalog,topology
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.sampling import common


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


class NormalizeRNG(ast.NodeTransformer):
    """Ignore only the explicitly added optional RNG interface, not formulas."""
    def visit_Name(self,node):
        if node.id=='rng' and isinstance(node.ctx,ast.Load):
            return ast.copy_location(ast.Attribute(value=ast.Name(id='np',ctx=ast.Load()),attr='random',ctx=ast.Load()),node)
        return node


def normalized_function(fn):
    node=ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    node.body=[s for s in node.body if not (isinstance(s,ast.Assign) and len(s.targets)==1
               and isinstance(s.targets[0],ast.Name) and s.targets[0].id=='rng')]
    keep=[i for i,a in enumerate(node.args.kwonlyargs) if a.arg!='rng']
    node.args.kwonlyargs=[node.args.kwonlyargs[i] for i in keep]
    node.args.kw_defaults=[node.args.kw_defaults[i] for i in keep]
    return ast.dump(NormalizeRNG().visit(node),include_attributes=False)


def audit(seeds,draws):
    results=[]
    def add(name,ok,**detail):results.append(dict(check=name,equal=bool(ok),**detail))
    cells=[(g,m) for g in catalog.GENE_NAMES for m in catalog.GROUP_BY_NAME[g]['modes']]+[('ORI_BUD_SPN','perturb')]
    for profile,legacy in [('mass','mass_dataset_generator'),('mutant_aware','mutant_aware')]:
        old=importlib.import_module('budding_yeast_v2.data.'+legacy)
        backend=BuddingSystem(profile,profile)
        for name,new in [('sample_scaling_factor',backend._distributions.sample_scaling_factor),
                         ('apply_gene_mutation',backend._mutations.apply_gene_mutation)]:
            add(profile+'.'+name+'.normalized_ast',normalized_function(getattr(old,name))==normalized_function(new))
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
        # Native MASS metadata lacks two safe-initialization keys that MA owns.
        changes=[]
        for key,a in old.BINDING_GROUPS.items():
            b=catalog.BINDING_GROUPS[key]
            for field in set(a)|set(b):
                if not identical(a.get(field),b.get(field)):
                    changes.append(dict(group=a['name'],field=field,old=a.get(field),new=b.get(field)))
        expected=[dict(group='TEM1',field='params',old=['Tem1T','ka15pp'],new=['Tem1T','ka15pp','init_Tem1']),
                  dict(group='CDC15',field='params',old=['Cdc15T','kkpnetpp'],new=['Cdc15T','kkpnetpp','init_Cdc15'])] if profile=='mass' else []
        add(profile+'.catalog',identical(changes,expected),exact_equal=not changes,
            authorized_canonical_initialization_differences=changes)
    old=importlib.import_module('budding_yeast_v2.data.mutant_aware')
    for target,keys in [(catalog,['BINDING_GROUPS','GENE_NAMES','GROUP_BY_NAME','GENE_ROLE','MODE_CLASS','MODE_CLASS_QUOTA',
                                 'ROLE_AFFINITY','TOUCH','COMPOSITE_ELIGIBLE']),
                        (topology,['MODULE_EDGES','GENE_WEIGHT_ARM','P_BIO','ADJ','DIST','_NEG_PAIRS','_OUT','_OUT_NEG',
                                   '_OUT_ALL','CONVERGENT_PAIRS','ALL_PAIRS','PAIR_LISTS','HUBS_DEG3','STAR_LEAVES']),
                        (common,['_L1_GRID'])]:
        for key in keys:add('module.'+key,identical(getattr(old,key),getattr(target,key)))
    mode_checks=0
    for frequency in ('class','flat','role','s5'):
        old.MODE_FREQ=common.MODE_FREQ=frequency
        for seed in seeds:
            for gene in catalog.GENE_NAMES:
                np.random.seed(seed);a=[old._sample_mode(gene) for _ in range(draws)];sa=np.random.get_state()
                np.random.seed(seed);b=[common._sample_mode(gene) for _ in range(draws)]
                add(f'mode.{frequency}.{seed}.{gene}',identical((a,sa),(b,np.random.get_state())))
                mode_checks+=draws
    for seed in seeds:
        for explicit in (False,True):
            pairs=[('CLN2','KO'),('SBF_MBF','KO'),('CLB2','MC'),('CLB2','db'),('CDC28','ts'),('CDC6','PM')]
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
    # Retain each specialized graph; do not silently force WT-Blind's 46 edges to 54.
    for source,target,keys in [('bams_dataset_generator','sampling.bams',['MODULE_EDGES','ADJ','EDGES_OUT','EDGES_IN','CONVERGENT_TARGETS','ANTAG_PAIRS','_L1_GRID']),
                               ('mass_wt_blind_sampler','sampling.wt_blind',['MODULE_EDGES','P_BIO','ADJ','DIST','FOLD_MODES','ALLELE_RESIDUAL','_L1_GRID']),
                               ('fission_random_sampler','systems.fission.catalog',['SAMPLING_MODULES'])]:
        a=importlib.import_module('budding_yeast_v2.data.'+source)
        b=importlib.import_module('budding_yeast_v2.data.'+target)
        for key in keys:add(source+'.'+key,identical(getattr(a,key),getattr(b,key)))
    manifest=json.loads((ROOT/'data/generation/archived_generators.json').read_text())
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
