"""Legal module/allele enumeration and capacity-capped equal allocation."""
import itertools
import math
import numpy as np
from . import catalog
from .distributions.ranges import copy_support,GAL_MODES

def allele_options(gene,composites=True):
    modes=list(catalog.GROUP_BY_NAME[gene]['modes']);options=[(m,) for m in modes]
    if composites and gene in catalog.COMPOSITE_ELIGIBLE:
        synth=[m for m in ('GAL-OE','MC') if m in modes]
        structural=[m for m in ('db','PM','CA') if m in modes]
        options.extend(tuple(sorted((a,b))) for a,b in itertools.product(synth,structural)
                       if not catalog.touch_set(gene,a)&catalog.touch_set(gene,b))
    return sorted(set(options))


def enumerate_combinations(composites=True):
    genes=[g for g in catalog.GENE_NAMES if g!='MDT'];options={g:allele_options(g,composites) for g in genes}
    rows=[];counts=[]
    for level in (1,2,3):
        feasible_sets=0;legal=0;rejected=0;fixed=0
        for group in itertools.combinations(genes,level):
            for modes in itertools.product(*(options[g] for g in group)):
                used=set();conflict=False
                for gene,allele in zip(group,modes):
                    touch=set().union(*(catalog.touch_set(gene,m) for m in allele))
                    if touch&used:conflict=True;break
                    used.update(touch)
                if conflict:rejected+=1;continue
                legal+=1
                flat_modes=[m for allele in modes for m in allele]
                continuous=any(m not in ('KO','CA','CSL','CKO','PM','MC','GAL-MC') for m in flat_modes)
                capacity=None if continuous else (1 if GAL_MODES.intersection(flat_modes) else 3)*math.prod(
                    len(copy_support(g,m)) for g,allele in zip(group,modes) for m in allele if m in ('MC','GAL-MC'))
                if capacity==3:fixed+=1
                rows.append(dict(level=level,modules=';'.join(group),alleles=';'.join(f'{g}[{"+".join(a)}]' for g,a in zip(group,modes)),finite_capacity_upper=capacity,has_continuous_strength=continuous))
            # Count module sets with at least one legal allele assignment.
            if rows and rows[-1]['modules']==';'.join(group):feasible_sets+=1
        counts.append(dict(level=level,all_module_sets=math.comb(len(genes),level),feasible_module_sets=feasible_sets,legal_allele_combinations=legal,rejected_allele_combinations=rejected,fixed_only_cells=fixed))
    return rows,counts


def waterfill_quotas(capacities,total,seed=42):
    """Equal allocation up to each cell's unique-input capacity."""
    caps=np.asarray(capacities,dtype=float)
    if total<0 or (caps<0).any() or np.isnan(caps).any() or caps.sum()<total:
        raise ValueError('Invalid or insufficient capacity')
    low,high=0,total+1
    while low+1<high:
        middle=(low+high)//2
        if np.minimum(caps,middle).sum()<=total:low=middle
        else:high=middle
    quota=np.minimum(caps,low).astype(int)
    remaining=total-int(quota.sum());eligible=np.flatnonzero(caps>quota)
    order=np.random.RandomState(seed).permutation(eligible)
    quota[order[:remaining]]+=1
    assert quota.sum()==total and (quota<=caps).all()
    return quota

