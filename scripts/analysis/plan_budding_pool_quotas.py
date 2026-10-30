"""Enumerate current legal module/allele combinations for AL quota design.

This planner writes tables only. Production sampling, mutation definitions,
strength distributions, and existing datasets are untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.systems.budding import catalog


from budding_yeast_v2.data.systems.budding.quotas import allele_options, enumerate_combinations, waterfill_quotas


def fixed_module_capacity(group,backend):
    """Exact capacity for finite-only module sets (currently <=180 inputs).

    Includes all discrete MC copy counts, three environments and the canonical
    real-collision predicate. Does not add or change mutation types.
    """
    from budding_yeast_v2.data.systems.budding.al_pool import finite_options
    seen=set()
    for _,row in group.iterrows():
        support=finite_options(row.to_dict(),backend)
        if support is None:raise ValueError('Expected a small finite support')
        seen.update(option[0] for option in support)
    return len(seen)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/analysis/al_pool_redesign_2026-10-28'))
    parser.add_argument('--pool-size',type=int,default=1000000);args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    rows,counts=enumerate_combinations(True);frame=pd.DataFrame(rows);frame.to_csv(args.output/'legal_allele_combinations.csv',index=False)
    count=pd.DataFrame(counts)
    for column in ('feasible_module_sets','legal_allele_combinations'):
        exact=args.pool_size*count[column].to_numpy()/count[column].sum();q=np.floor(exact).astype(int)
        q[np.argsort(-(exact-q),kind='stable')[:args.pool_size-int(q.sum())]]+=1
        count[column+'_quota_uncapped']=q
    count.to_csv(args.output/'quota_options.csv',index=False)
    from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
    backend=BuddingSystem();modules=[]
    for names,group in frame.groupby('modules',sort=True):
        cap=None if group.has_continuous_strength.any() else fixed_module_capacity(group,backend)
        modules.append(dict(modules=names,level=int(group.level.iloc[0]),legal_alleles=len(group),unique_capacity=cap))
    capacities=[np.inf if item['unique_capacity'] is None else item['unique_capacity'] for item in modules]
    quotas=waterfill_quotas(capacities,args.pool_size)
    for item,quota in zip(modules,quotas):item['quota']=int(quota)
    module_frame=pd.DataFrame(modules);module_frame.to_csv(args.output/'capacity_aware_module_quotas.csv',index=False)
    capped=module_frame.groupby('level').quota.sum().to_dict()
    _,atomic=enumerate_combinations(False);pd.DataFrame(atomic).to_csv(args.output/'atomic_combinations_counts.csv',index=False)
    summary=dict(pool_size=args.pool_size,genetic_modules=len(catalog.GENE_NAMES)-1,elementary_gene_mode_pairs=sum(len(catalog.GROUP_BY_NAME[g]['modes']) for g in catalog.GENE_NAMES if g!='MDT'),alleles_including_composites=int(count.legal_allele_combinations.iloc[0]),counts=counts,
        capacity_aware_module_quotas=capped,production_modified=False,quotas_are_uncapped_design_only=False,
        limitations=['global named-control aliases are not merged','per-module mode quotas and acquisition implementation remain design work','uncapped quota_options.csv is a comparison, not the chosen production quota'])
    (args.output/'enumeration_summary.json').write_text(json.dumps(summary,indent=2))
    print(count.to_string(index=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
