"""Regression checks for the modular dataset's public consumers."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.generate import main
from budding_yeast_v2.data.systems.contracts import Candidate,GenerationConfig
from budding_yeast_v2.data.systems.dataset_io import write_dataset
from budding_yeast_v2.data.systems import generation as runner
from budding_yeast_v2.data.systems.registry import RECIPES,resolve_config,output_filename
from budding_yeast_v2.data.systems.budding import api
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem,SPEC
from budding_yeast_v2.data.systems.budding.adapter import BuddingAdapter
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.data.dataset_utils import load_yeast_dataset_universal


@pytest.mark.parametrize('name',list(RECIPES))
def test_cli_resolves_recipe_defaults_and_filename(name,tmp_path,capsys):
    cfg=main(['--recipe',name,'--data_dir',str(tmp_path),'--show-config'])
    assert cfg.n_samples==RECIPES[name].default_samples
    assert cfg.seed==RECIPES[name].default_seed
    assert cfg.output==str(tmp_path/output_filename(cfg))
    assert json.loads(capsys.readouterr().out)['run']['seed']==cfg.seed


def test_original_optional_controls_remain_exposed(capsys):
    cfg=main(['--recipe','budding_bams','--no_bisection','--show-config'])
    assert not cfg.enable_bisection
    cfg=main(['--recipe','budding_wt_blind','--show-config','--nan-cap-frac','.12',
              '--backlog-thresh','.31','--max-attempts-mult','50'])
    assert (cfg.pilot_samples,cfg.nan_cap_frac,cfg.backlog_thresh,cfg.max_attempts_mult)==(20000,.12,.31,50)


def archive_rows():
    candidates=[Candidate(-1,'1_WT_Glc',{},0,True)]
    candidates += [Candidate(i,f'RANDOM_{i}',{'C0':.5+i*.02},1) for i in range(1,9)]
    rows=[]
    for candidate in candidates:
        row=BuddingAdapter().parameter_result(candidate,None)
        row.trajectory=np.repeat(row.y_init[:,None],500,axis=1)
        row.label='Viable';row.pattern='Pattern_A';row.status='success'
        rows.append(row)
    return rows


def test_old_and_new_files_produce_identical_loader_tensors_and_splits(tmp_path):
    new=tmp_path/'new_name_without_physical_time.npz'
    write_dataset(new,SPEC,archive_rows(),GenerationConfig('budding_random'),
                  {'mutation_profile':'mutant_aware','distribution_profile':'mutant_aware','rng_contract':'test'})
    with np.load(new) as saved:
        old=tmp_path/'historical_210m_500s.npz'
        np.savez_compressed(old,**{k:saved[k] for k in ('data','params','param_names','mutant_names','labels','pattern_labels')})
    legacy=load_yeast_dataset_universal(old)
    current=load_yeast_dataset_universal(new)
    for a,b in zip(legacy,current):np.testing.assert_array_equal(a,b)
    modules=[YeastDataModule(p,use_adj_matrix=False,num_workers=0) for p in (old,new)]
    for dm in modules:dm.setup('fit')
    assert modules[0].t_max==modules[1].t_max==210
    for split in ('train_dataset_synth','val_dataset_synth','test_dataset_synth','test_dataset_real'):
        for a,b in zip(getattr(modules[0],split).tensors,getattr(modules[1],split).tensors):
            assert torch.equal(a,b)


def test_no_real_flag_prevents_synthetic_samples_becoming_real(tmp_path):
    rows=archive_rows()[1:]
    for row in rows:row.candidate.name='123_'+row.candidate.name
    path=tmp_path/'no_real.npz'
    write_dataset(path,SPEC,rows,GenerationConfig('budding_random'),
                  {'mutation_profile':'mutant_aware','distribution_profile':'mutant_aware','rng_contract':'test'})
    dm=YeastDataModule(path,use_adj_matrix=False,num_workers=0);dm.setup('fit')
    assert len(dm.test_dataset_real)==0
    assert sum(len(getattr(dm,k)) for k in ('train_dataset_synth','val_dataset_synth','test_dataset_synth'))==8


def test_al_oracle_keeps_overrides_outside_exported_parameter_vector(tmp_path,monkeypatch):
    from budding_yeast_v2.data import al_pool_manager as pool
    overrides={'init_CLB2':.125,'Tem1T':0.,'init_Tem1':0.,'mdt':150.}
    row=BuddingAdapter().parameter_result(Candidate(1,'pool_candidate',overrides,1),None)
    second=BuddingAdapter().parameter_result(Candidate(2,'pool_candidate_2',overrides,1),None)
    path=tmp_path/'pool.npz'
    write_dataset(path,SPEC,[row,second],GenerationConfig('budding_pool'),
                  {'mutation_profile':'mutant_aware','distribution_profile':'mutant_aware','rng_contract':'test',
                   'pool_generation_contract':__import__('budding_yeast_v2.data.systems.budding.al_pool',fromlist=['POOL_CONTRACT']).POOL_CONTRACT},parameter_only=True)
    seen=[]
    def oracle(name,ovr,*args):
        seen.append(ovr)
        _,y,_,_=api.prepare_initial_conditions(ovr)
        return name,np.repeat(y[:,None],500,axis=1),'Viable','Pattern_A'
    monkeypatch.setattr(pool.ma,'simulate_mutant',oracle)
    monkeypatch.setattr(pool.concurrent.futures,'ProcessPoolExecutor',ThreadPoolExecutor)
    real_load=np.load
    reads={}
    class CountingArchive:
        def __init__(self,*args,**kwargs):self.archive=real_load(*args,**kwargs)
        def __contains__(self,key):return key in self.archive
        def __getitem__(self,key):
            reads[key]=reads.get(key,0)+1
            return self.archive[key]
        def close(self):self.archive.close()
    monkeypatch.setattr(pool.np,'load',CountingArchive)
    result=pool.simulate_selected_tasks([0,1],path,num_workers=1)
    assert seen==[overrides,overrides]
    assert reads['overrides_json']==1
    assert result[0]['pool_candidate'][api.nCLB2,0]==.125


def test_al_callback_reads_full_new_pool_overrides_and_initial_states(tmp_path):
    from types import SimpleNamespace
    from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
    overrides={'init_CLB2':.125,'Tem1T':0.,'init_Tem1':0.,'mdt':150.}
    row=BuddingAdapter().parameter_result(Candidate(1,'MASS_RAND_TEM1[KO]_L1_1',overrides,1),None)
    path=tmp_path/'pool.npz'
    write_dataset(path,SPEC,[row],GenerationConfig('budding_pool'),
                  {'mutation_profile':'mutant_aware','distribution_profile':'mutant_aware','rng_contract':'test',
                   'pool_generation_contract':__import__('budding_yeast_v2.data.systems.budding.al_pool',fromlist=['POOL_CONTRACT']).POOL_CONTRACT},parameter_only=True)
    dm=SimpleNamespace(mean_y=torch.zeros(1,39,1),std_y=torch.ones(1,39,1),
                       p_mean=torch.zeros(1,141),p_std=torch.ones(1,141),norm_type='linear',s_c=None)
    cb=ActiveLearningCallback(strategy='al4pde_random',al_pool_dataset=str(path))
    cb._generate_param_only_pool(dm)
    assert cb._pool_ovr==[overrides]
    np.testing.assert_allclose(cb._pool_Y_ic.numpy(),row.y_init[None,:],rtol=1e-6)


def test_al_packaging_keeps_canonical_real_order_and_explicit_flags(tmp_path,monkeypatch):
    from budding_yeast_v2.data import al_pool_manager as pool
    def oracle(task):
        name,ovr=task
        _,y,_,_=api.prepare_initial_conditions(ovr)
        return name,np.repeat(y[:,None],500,axis=1),'Viable','Pattern_A',ovr
    monkeypatch.setattr(pool,'_simulate_single_task',oracle)
    monkeypatch.setattr(pool.concurrent.futures,'ProcessPoolExecutor',ThreadPoolExecutor)
    rows=archive_rows()[1:]
    synth=({r.candidate.name:r.trajectory for r in rows},
           {r.candidate.name:r.label for r in rows},{r.candidate.name:r.pattern for r in rows},
           {r.candidate.name:r.params for r in rows},np.array(SPEC.param_names))
    path=tmp_path/'al.npz'
    pool.package_and_export_al_dataset(path,synth,num_workers=2)
    with np.load(path) as saved:
        assert saved['mutant_names'][:126].tolist()==[n for n,_ in api.real_mutants.real_tasks()]
        assert saved['is_real'].sum()==126
        assert not saved['is_real'][126:].any()


def test_incomplete_real_test_set_cannot_silently_export(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'run_simulations',lambda *args:iter(archive_rows()[1:]))
    with pytest.raises(RuntimeError,match='Incomplete real-mutant'):
        runner.generate_dataset(GenerationConfig('budding_random',n_samples=0,output=str(tmp_path/'partial.npz')))


def test_physics_parameter_defaults_use_modular_ma_and_rhs_matches():
    from budding_yeast_v2.utils import physics_eqns_torch as physics
    assert physics._defaults['ef6n3']==api.get_default_params()['ef6n3']
    y=api.initial_conds()
    expected=api.eqns(0.,y,api.get_default_params())
    actual=physics.eqns_torch(torch.tensor(y[None,:],dtype=torch.float64),api.get_default_params())
    np.testing.assert_allclose(actual.numpy()[0],expected,rtol=1e-7,atol=1e-8)
