from collections import Counter
from dataclasses import replace
import numpy as np
import pytest
from budding_yeast_v2.data.systems.budding.al_pool import plan_pool,_propose_group,signature,POOL_CONTRACT,validate_pool_contract
from budding_yeast_v2.data.systems.budding.catalog import CATALOG_VERSION
from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.contracts import GenerationConfig
from budding_yeast_v2.data.systems.registry import resolve_config


def test_million_module_budget_and_finite_saturation():
    groups,counts=plan_pool(1000000)
    assert [row['feasible_module_sets'] for row in counts]==[19,171,969]
    assert sum(g['quota'] for g in groups)==1000000
    infinite=[g['quota'] for g in groups if np.isinf(g['capacity'])]
    assert max(infinite)-min(infinite)<=1
    assert not any('SBF_MBF' in g['modules'].split(';') for g in groups)
    assert all(g['quota']<=g['capacity'] for g in groups)
    for name in ['BCK2','CLB2','BCK2;CLN2;CLN3']:
        group=next(g for g in groups if g['modules']==name)
        total=group['quota']
        if name=='BCK2':assert total==group['capacity']
        config=resolve_config(GenerationConfig('budding_pool',n_samples=1000000,parameter_only=True))
        _,rows,actual,caps,_=_propose_group((group,config,0,[0]*len(group['cells']),set()))
        assert len(rows)==total and sum(actual)==total
        assert len({row[3] for row in rows})==total
        active=[n for n,cap in zip(actual,caps) if n<cap]
        if active:assert max(active)-min(active)<=1
        assert all(row[4]['quota_module_set']==name for row in rows)
        assert not any(g in ('TEM1','CDC15') and m=='DAmP' for row in rows for g,m in row[4]['pairs'])


def test_global_duplicate_retry_preserves_module_and_balances_modes():
    groups,_=plan_pool(1000000);g=next(x for x in groups if x['modules']=='CLB2');g=dict(g,quota=20)
    cfg=resolve_config(GenerationConfig('budding_pool',parameter_only=True,n_samples=20))
    _,rows,_,_,_=_propose_group((g,cfg,0,[0]*len(g['cells']),set()))
    retained=rows[1:];prior=[0]*len(g['cells'])
    for row in retained:prior[row[0]]+=1
    banned={row[3] for row in rows}
    _,refill,counts,_,_=_propose_group((g,cfg,1,prior,banned))
    assert len(refill)==1 and sum(counts)==20 and not banned & {row[3] for row in refill}


def test_production_pool_contract_rejects_old_catalog_or_quota():
    with pytest.raises(ValueError,match='superseded'):
        validate_pool_contract({'artifact_kind':'parameter_pool','catalog_version':CATALOG_VERSION,'sampling_range_fingerprint':RANGE_FINGERPRINT})
    with pytest.raises(ValueError,match='catalog'):
        validate_pool_contract({'artifact_kind':'parameter_pool','pool_generation_contract':POOL_CONTRACT,'catalog_version':'old'})
    validate_pool_contract({'artifact_kind':'parameter_pool','pool_generation_contract':POOL_CONTRACT,'catalog_version':CATALOG_VERSION,'sampling_range_fingerprint':RANGE_FINGERPRINT})


def test_full_generation_exports_contract_and_actual_level_ratios(tmp_path):
    import json
    from budding_yeast_v2.data.systems.generation import generate_dataset
    path=tmp_path/'balanced.npz'
    generate_dataset(GenerationConfig('budding_pool',n_samples=17,seed=42,workers=2,output=str(path)))
    with np.load(path,allow_pickle=False) as z:
        report=json.loads(str(z['generation_report']));config=json.loads(str(z['generation_config']))
        assert str(z['pool_generation_contract'])==POOL_CONTRACT
        assert str(z['sampling_rng_contract'])==POOL_CONTRACT
        assert z['params'].shape==(17,141) and z['y_init'].shape==(17,39)
        actual=np.bincount(z['module_counts'],minlength=4)[1:]
        np.testing.assert_allclose(config['level_ratios'],actual/17)
        assert sum(row['accepted'] for row in report['module_quotas'])==17
        assert 'data' not in z and 'labels' not in z
        validate_pool_contract(z)
