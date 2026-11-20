"""Sample contents survive static serialization and AL provenance round trips."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from budding_yeast_v2.data.systems.contracts import Candidate, GenerationConfig, SimulationResult
from budding_yeast_v2.data.systems.registry import RECIPES
from budding_yeast_v2.data.systems.dataset_io import write_dataset
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem, SPEC
from budding_yeast_v2.data.systems.sampling_policy import VERSION
from budding_yeast_v2.data.sample_metadata import mutation_metadata
from budding_yeast_v2.utils.sampling.io import (
    source_provenance, provenance_fields, pool_provenance,
)
from budding_yeast_v2.utils.sampling.pool import CandidatePool


REQUIRED = {'params','param_names','y_init','variable_names','time_grid','mutant_names',
    'is_real','levels','requested_levels','module_counts','overrides_json',
    'candidate_metadata_json','mutation_descriptions','mutation_pairs_json',
    'mutation_pairs_available','candidate_ids','simulation_status',
    'simulation_diagnostics_json','initial_state_source','overrides_available',
    'sample_metadata_version'}
TRAJECTORY = {'data','labels','biological_labels_available','pattern_labels',
    'pattern_group_codes','pattern_group_labels','pattern_group_ids',
    'observed_division_counts','pattern_group_version','pattern_group_rules_json'}


def sample(parameter_only=False):
    backend = BuddingSystem()
    candidate = Candidate(17, 'MASS_RAND_CLB2[MC]+TEM1[ts]_L2_17',
        {'ksb2p':.004, 'init_Tem1':.123456789012345, 'ka15pp':.001, 'mdt':150.}, 2,
        metadata={'description':'CLB2[MC]+TEM1[ts]',
                  'pairs':[['CLB2','MC'],['TEM1','ts']],
                  'source_candidate_id':16, 'source_pool':'fixture_pool.npz'})
    row = __import__('budding_yeast_v2.data.systems.budding.adapter',fromlist=['ADAPTER']).ADAPTER.parameter_result(candidate,None)
    if parameter_only:
        return row
    row.trajectory = np.repeat(row.y_init[:,None],500,axis=1)
    row.label, row.pattern, row.status = 'Inviable: G1 Arrest', 'Pattern_C: Exponential-Arrest (==0)', 'success'
    return row


@pytest.mark.parametrize('recipe',[name for name,recipe in RECIPES.items() if recipe.system=='budding' and recipe.sampler!='pool'])
def test_every_budding_recipe_has_complete_common_export(tmp_path,recipe):
    row = sample()
    path = tmp_path/'fixture.npz'
    write_dataset(path,SPEC,[row],GenerationConfig(recipe,n_samples=1,include_real=False),
        dict(mutation_profile=RECIPES[recipe].mutation,distribution_profile=RECIPES[recipe].distribution,
             rng_contract='serialization_fixture',sampling_policy=VERSION),dtype='float32')
    with np.load(path,allow_pickle=False) as saved:
        assert REQUIRED|TRAJECTORY <= set(saved.files)
        np.testing.assert_array_equal(saved['y_init'][0],row.y_init)
        assert saved['y_init'].dtype == np.float64
        assert saved['y_init'][0,28] != saved['data'][0,28,0]  # exact IC is not float32 Y[0]
        assert json.loads(str(saved['overrides_json'][0])) == row.candidate.overrides
        assert json.loads(str(saved['mutation_pairs_json'][0])) == [['CLB2','MC'],['TEM1','ts']]
        assert saved['candidate_ids'].tolist()==[16]
        rows = source_provenance(saved)
        fields = provenance_fields(saved['mutant_names'],saved['data'],rows,saved['is_real'],'budding')
        for field in ('y_init','overrides_json','candidate_metadata_json','mutation_pairs_json','candidate_ids'):
            np.testing.assert_array_equal(saved[field],fields[field])


def test_parameter_pool_has_complete_inputs_but_no_fabricated_trajectory_labels(tmp_path):
    row=sample(True);path=tmp_path/'pool.npz'
    write_dataset(path,SPEC,[row],GenerationConfig('budding_pool',n_samples=1,include_real=False),
        dict(mutation_profile='mutant_aware',distribution_profile='mutant_aware',
             rng_contract='serialization_fixture',sampling_policy=VERSION),parameter_only=True)
    with np.load(path,allow_pickle=False) as saved:
        assert REQUIRED <= set(saved.files)
        assert not TRAJECTORY & set(saved.files)
        assert saved['simulation_status'].tolist()==['parameter_only']
        assert json.loads(str(saved['mutation_pairs_json'][0])) == [['CLB2','MC'],['TEM1','ts']]


def test_real_benchmark_keeps_genotype_without_guessing_synthetic_alleles():
    name='15_cln3Δ bck2Δ multi-copy CLN2'
    fields=mutation_metadata([name],['{}'],[True],'budding')
    assert fields['mutation_descriptions'].tolist()==[name]
    assert fields['mutation_pairs_available'].tolist()==[False]
    assert fields['mutation_pairs_json'].tolist()==['[]']


def test_pool_provenance_captured_before_reindexing():
    pool=CandidatePool();pool.size=3;pool.ensure_ids()
    states=torch.tensor(np.arange(117).reshape(3,39)+.123456789,dtype=torch.float64)
    engine=SimpleNamespace(pool=pool,_pool_mass_meta=[{'level':1,'pairs':[['TEM1','ts']]}]*3,
        _pool_initial_physical=states,_pool_ovr=[{'init_Tem1':float(i)} for i in range(3)],
        al_pool_dataset='/fixture/pool.npz')
    rows=pool_provenance(engine,[2,0])
    pool.keep(np.array([False,True,False]))
    assert [row['candidate_id'] for row in rows]==[2,0]
    np.testing.assert_array_equal(rows[0]['y_init'],states[2])
    assert json.loads(rows[0]['overrides_json'])=={'init_Tem1':2.}
    assert json.loads(rows[0]['candidate_metadata_json'])['source_candidate_id']==2


def test_source_provenance_reads_each_compressed_column_once():
    from collections import Counter
    class Archive(dict):
        def __init__(self,**arrays):
            super().__init__(arrays);self.files=list(arrays);self.reads=Counter()
        def __getitem__(self,key):
            self.reads[key]+=1
            return super().__getitem__(key)
    source=Archive(mutant_names=np.array(['a','b','c']),y_init=np.ones((3,39)),
        data=np.ones((3,39,500)),overrides_json=np.array(['{}']*3),
        candidate_metadata_json=np.array(['{}']*3),candidate_ids=np.array([5,8,10]))
    rows=source_provenance(source)
    assert [row['candidate_id'] for row in rows]==[5,8,10]
    assert source.reads['data']==0
    assert all(count==1 for count in source.reads.values())
