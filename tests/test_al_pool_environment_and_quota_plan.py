from collections import Counter
import numpy as np
import pytest

from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.budding.samplers.sobol import sample_pure_random_candidate
from budding_yeast_v2.scripts.analysis.plan_budding_pool_quotas import allele_options,enumerate_combinations,waterfill_quotas


class FixedMode:
    def __init__(self,mode='KO'):self.mode=mode
    def choice(self,values,size=None,replace=True):
        return np.array(['CLN3']) if size is not None else self.mode


@pytest.mark.parametrize('u,expected',[(0.,90.),(.799999,90.),(.8,150.),(.949999,150.),(.95,160.),(.999999,160.)])
def test_pool_environment_boundaries(u,expected):
    q=np.full(8,.5);q[-1]=u
    ovr,_,_=sample_pure_random_candidate(1,q,FixedMode(),ma=BuddingSystem())
    assert ovr['mdt']==expected


def test_pool_environment_counts_and_gal_force():
    count=Counter()
    for u in (np.arange(1000)+.5)/1000:
        q=np.full(8,.5);q[-1]=u
        ovr,_,_=sample_pure_random_candidate(1,q,FixedMode(),ma=BuddingSystem());count[ovr['mdt']]+=1
    assert count=={90.:800,150.:150,160.:50}
    for u in [0.,.85,.98]:
        q=np.full(8,.5);q[-1]=u
        ovr,_,_=sample_pure_random_candidate(1,q,FixedMode('GAL-OE'),ma=BuddingSystem())
        assert ovr['mdt']==150.


def test_quota_enumeration_rejects_impossible_overlaps_and_canonicalizes_composites():
    opts=allele_options('PPX');assert ('GAL-OE','db') in opts and ('db','GAL-OE') not in opts
    rows,counts=enumerate_combinations()
    assert len({r['alleles'] for r in rows})==len(rows)
    assert [x['feasible_module_sets'] for x in counts]==[22,230,1520]
    assert not any({'SBF_MBF','CLN2'}.issubset(r['modules'].split(';')) for r in rows)
    assert next(r for r in rows if r['alleles']=='SBF_MBF[KO]')['finite_capacity_upper']==3


def test_capacity_capped_equal_allocation():
    q=waterfill_quotas([3,88,np.inf,np.inf],1000)
    assert q.sum()==1000 and q[0]==3 and q[1]==88
    assert abs(q[2]-q[3])<=1
    with pytest.raises(ValueError,match='capacity'):
        waterfill_quotas([3,3],7)
