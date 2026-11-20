import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from budding_yeast_v2.data.systems.budding.lhs import mixture_cdf,mixture_ppf,sample_design,mixture_atoms
from budding_yeast_v2.data.systems.budding.offline_pool import Proposals,stream_seed,generate
from budding_yeast_v2.data.systems.budding.specification import SPEC
from budding_yeast_v2.data.systems.budding.real_mutants import is_duplicate_real_mutant
from budding_yeast_v2.data.dataset_archive import open_dataset
from budding_yeast_v2.data.dataset_utils import load_yeast_dataset_universal
CONFIG=Path('/home/users/zsy/mydata/budding_yeast/analysis/offline_pool_plan_20261112_exact_zero/mixed_lhs_domain.json')

@pytest.fixture
def spec():return json.loads(CONFIG.read_text())

def test_full_mixture_inverse_with_overlapping_components():
    parts=[dict(family='point',value=1.,weight=.3),dict(family='uniform',lower=0.,upper=2.,weight=.4),dict(family='uniform',lower=.5,upper=1.5,weight=.3)]
    u=(np.arange(4096)+.5)/4096;x=mixture_ppf(u,parts)
    assert (np.diff(x)>=0).all()
    atom=x==1
    assert abs(atom.mean()-.3)<2/len(u)
    np.testing.assert_allclose(mixture_cdf(x[~atom],parts),u[~atom],atol=1e-12,rtol=0)

def test_all_marginal_inverses_preserve_atoms_and_continuous_cdf(spec):
    u=(np.arange(1024)+.5)/1024
    for name,parts in spec['marginals'].items():
        x=mixture_ppf(u,parts);assert np.isfinite(x).all() and (np.diff(x)>=-1e-12).all(),name
        continuous=np.ones(len(x),dtype=bool)
        for value,mass in mixture_atoms(parts):
            hits=x==value;continuous &= ~hits
            assert abs(hits.sum()-mass*len(u))<=2,(name,value,mass,hits.sum())
        np.testing.assert_allclose(mixture_cdf(x[continuous],parts),u[continuous],atol=2e-10,rtol=0,err_msg=name)

def test_lhs_latin_coordinates_all_parameters_and_shared_controls(spec):
    names=list(SPEC.param_names);p,u=sample_design(spec,names,128,42)
    p2,u2=sample_design(spec,names,128,42)
    np.testing.assert_array_equal(p,p2);np.testing.assert_array_equal(u,u2)
    for j in range(len(names)):np.testing.assert_array_equal(np.sort((u[:,j]*128).astype(int)),np.arange(128))
    assert (p[:,names.index('init_CDH1')]<=p[:,names.index('init_CDH1T')]+1e-12).all()
    assert np.array_equal(p[:,names.index('BUB2_active')]==0,p[:,names.index('init_BUB2')]==0)
    for key in ['kd20','kd14','kd1c1','kasbf','ksppx']:
        j=names.index(key);parts=spec['marginals'][key]
        assert parts[0]['lower']<=p[:,j].min()<=p[:,j].max()<=parts[0]['upper']

def test_mutation_proposals_have_no_sobol_and_no_MDT_genetic_module(tmp_path):
    a=Proposals('mutation',42,100000,tmp_path,None);counts={1:0,2:0,3:0}
    with patch('scipy.stats.qmc.Sobol',side_effect=AssertionError('Sobol forbidden')):
        for i in range(400):
            result,rejection=a.propose(i)
            if result is None:continue
            c,_=result;counts[c.level]+=1
            assert len(set(g for g,m in c.metadata['pairs']))==c.level
            assert all(g!='MDT' for g,m in c.metadata['pairs'])
            assert not is_duplicate_real_mutant(c.overrides,a.defaults)
            assert c.overrides['mdt'] in [90,150,160]
            if any(m in ('GAL-OE','GAL-MC') for _,m in c.metadata['pairs']):assert c.overrides['mdt']==150
            other,_=a.propose(i)
            assert other[0].overrides==c.overrides and other[1]==result[1]
    assert sum(counts.values())>390 and counts[3]>counts[2]>counts[1]

@pytest.mark.parametrize('kind',['mutation','lhs'])
def test_actual_small_pool_metadata_and_resume(tmp_path,kind):
    root=tmp_path/kind
    result=generate(kind,root,count=8,seed=91,workers=1,block_size=4,lhs_config=CONFIG if kind=='lhs' else None)
    data,params,names,dim,abc,meta=load_yeast_dataset_universal(result,return_metadata=True)
    assert data.shape==(8,39,500) and params.shape==(8,141) and dim==141
    with open_dataset(result,allow_pickle=False) as z:
        assert not z['is_real'].any() and z['y_init'].dtype==np.float64 and z['data'].dtype==np.float32
        assert len(set(z['sample_id']))==8 and len(set(z['candidate_ids']))==8
        assert len(z['labels'])==len(z['pattern_group_codes'])==8
        if kind=='lhs':
            assert not z['mutation_pairs_available'].any() and not z['mutation_available'].any()
            assert (z['levels']==-1).all() and all(json.loads(x)==[] for x in z['mutation_pairs_json'])
        else:assert z['mutation_available'].all() and np.isin(z['levels'],[1,2,3]).all()
    again=generate(kind,root,count=8,seed=91,workers=2,block_size=4,lhs_config=CONFIG if kind=='lhs' else None,resume=True)
    with open_dataset(again) as z:np.testing.assert_array_equal(z['params'],params)
    with pytest.raises(ValueError,match='changed'):
        generate(kind,root,count=8,seed=92,workers=1,block_size=4,lhs_config=CONFIG if kind=='lhs' else None,resume=True)

def test_stream_seeds_separate_kind_and_candidate():
    assert len({stream_seed(42,k,i) for k in ['mutation','lhs'] for i in range(20)})==40

def test_resume_after_committed_chunk_and_interrupted_progress_file(tmp_path):
    from budding_yeast_v2.data.systems.budding import offline_pool
    original=offline_pool._atomic_json
    def interrupt(path,payload):
        if Path(path).name=='progress.json':raise RuntimeError('simulated interruption')
        return original(path,payload)
    root=tmp_path/'interrupted'
    with patch.object(offline_pool,'_atomic_json',side_effect=interrupt):
        with pytest.raises(RuntimeError,match='simulated interruption'):
            generate('mutation',root,count=8,seed=22,workers=1,block_size=4)
    path=generate('mutation',root,count=8,seed=22,workers=2,block_size=4,resume=True)
    with open_dataset(path) as z:
        assert len(z['data'])==8 and len(set(z['candidate_ids']))==8
        np.testing.assert_array_equal(z['sample_id'],np.arange(1,9))
