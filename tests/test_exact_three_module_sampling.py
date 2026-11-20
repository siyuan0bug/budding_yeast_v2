"""New exact module-count policy: two purposes, all samplers and quota refills."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from budding_yeast_v2.data.systems import generation
from budding_yeast_v2.data.systems.contracts import Candidate,GenerationConfig
from budding_yeast_v2.data.systems.registry import RECIPES,resolve_config,make_sampler
from budding_yeast_v2.data.systems.sampling_policy import VERSION,STATIC_RATIOS,POOL_RATIOS,valid_candidate


@pytest.mark.parametrize('recipe',list(RECIPES))
def test_recipe_purpose_defaults_and_high_order_zero(recipe):
    spec=RECIPES[recipe]
    config=resolve_config(GenerationConfig(recipe))
    assert config.level_ratios==(POOL_RATIOS if spec.sampler=='pool' else STATIC_RATIOS)
    assert resolve_config(GenerationConfig(recipe,parameter_only=True)).level_ratios==POOL_RATIOS
    with pytest.raises(ValueError,match='higher-level'):
        resolve_config(GenerationConfig(recipe,level_ratios=(.1,.4,.4,.1)))
    assert resolve_config(GenerationConfig(recipe,level_ratios=(.1,.4,.5,0,0))).level_ratios==STATIC_RATIOS


@pytest.mark.parametrize('recipe',[k for k,v in RECIPES.items() if v.sampler not in ('twins','ko_mirror','bams_ablation')])
def test_all_sampler_algorithms_return_exact_modules_after_rescue(recipe):
    config=resolve_config(GenerationConfig(recipe,n_samples=20,seed=731,include_real=False,pilot_samples=8))
    np.random.seed(config.seed)
    sampler=make_sampler(RECIPES[recipe],config)
    report=Counter()
    rows=generation.prepare_candidates(RECIPES[recipe],config,sampler,report)
    assert rows
    for row in rows:
        assert row.level in (1,2,3)
        assert row.metadata['module_count']==row.level
        assert len(row.metadata['module_names'])==row.level
        assert row.metadata['sampling_policy']==VERSION


def test_composites_and_environment_and_real_exemption():
    good=valid_candidate(Candidate(1,'x',{},1,metadata={'description':'CLB2[GAL-OE+db]_MDT[150]'}),'budding')
    assert good.metadata['module_names']==['CLB2']
    assert valid_candidate(Candidate(1,'x',{},1,metadata={'description':'MDT[MDT]'}),'budding') is None
    row=valid_candidate(Candidate(1,'x',{},3,metadata={'description':'Wee1:ts+Rum1:OP+Ste9:KO+Growth:scale'}),'fission')
    assert row.metadata['module_count']==3 and row.metadata['environment_modules']==['Growth']
    assert valid_candidate(Candidate(-1,'42',{},0,True),'budding').is_real


def test_static_valid_quota_refills_same_level_and_preserves_real_benchmark(tmp_path,monkeypatch):
    attempted=Counter()
    original=generation.simulate_task
    def controlled(candidates,config,context):
        adapter=generation.get_adapter('budding')
        for candidate in candidates:
            row=adapter.parameter_result(candidate,context)
            attempted[candidate.level]+=1
            if candidate.level==2 and attempted[2]<=2:
                row.status='numerical_failure';yield row;continue
            row.trajectory=np.tile(row.y_init[:,None],(1,500));row.status='success'
            row.label='Viable';row.pattern='Pattern_A'
            yield row
    monkeypatch.setattr(generation,'run_simulations',controlled)
    path=tmp_path/'static.npz'
    generation.generate_dataset(GenerationConfig('budding_random',n_samples=20,seed=3,output=str(path)))
    with np.load(path) as saved:
        assert saved['is_real'].sum()==126
        counts=Counter(saved['module_counts'][~saved['is_real']])
        assert counts=={1:2,2:8,3:10}
        assert json.loads(str(saved['generation_report']))['refill_rounds']>=1
        assert len(set(saved['sample_id']))==146
        assert any(str(n).startswith('42_') for n in saved['mutant_names'])


def test_quota_exhaustion_does_not_export_partial_dataset(tmp_path,monkeypatch):
    def failed(candidates,config,context):
        for candidate in candidates:
            row=generation.get_adapter('budding').parameter_result(candidate,None)
            row.status='numerical_failure';yield row
    monkeypatch.setattr(generation,'run_simulations',failed)
    path=tmp_path/'never.npz'
    with pytest.raises(RuntimeError,match='quota not filled'):
        generation.generate_dataset(GenerationConfig('budding_random',n_samples=10,seed=8,
            include_real=False,max_attempts_mult=2,output=str(path)))
    assert not path.exists()


def test_pool_fallback_and_initial_kmeans_share_the_new_policy(tmp_path,monkeypatch):
    import shutil
    import torch
    from types import SimpleNamespace
    from torch.utils.data import TensorDataset
    from budding_yeast_v2.utils.sampling import ActiveLearningEngine
    from budding_yeast_v2.data.al_pool_manager import select_initial_kmeans
    path=tmp_path/'pool.npz'
    generation.generate_dataset(GenerationConfig('budding_pool',n_samples=2000,seed=882,
        output=str(path),include_real=False,parameter_only=True))
    with np.load(path) as saved:counts=saved['module_counts']
    indices=select_initial_kmeans(path,n_samples=20,seed=4)
    assert Counter(counts[indices])=={1:2,2:8,3:10}
    calls=[]
    def fake(config):
        calls.append(config)
        shutil.copyfile(path,config.output)
    monkeypatch.setattr(generation,'generate_dataset',fake)
    dm=SimpleNamespace(norm_type='linear',p_mean=torch.zeros(1,141),p_std=torch.ones(1,141),
        mean_y=torch.zeros(1,39,1),std_y=torch.ones(1,39,1),t_max=210,
        train_dataset_synth=TensorDataset(torch.zeros(1,39,2,500),torch.zeros(1,141),torch.zeros(1,39,500)))
    engine=ActiveLearningEngine()
    engine._generate_param_only_pool(dm)
    assert len(calls)==1 and calls[0].recipe=='budding_pool'
    assert resolve_config(calls[0]).level_ratios==POOL_RATIOS
    assert engine._pool_size==2000 and engine.al_pool_dataset is None


def test_initial_manager_replaces_failed_candidates_in_the_same_stratum(tmp_path,monkeypatch):
    from budding_yeast_v2.data import al_pool_manager as manager
    path=tmp_path/'pool.npz'
    generation.generate_dataset(GenerationConfig('budding_pool',n_samples=2000,seed=224,
        output=str(path),include_real=False,parameter_only=True))
    with np.load(path) as saved:counts=saved['module_counts'];names=saved['mutant_names']
    seen=[]
    def simulate(indices,*args):
        seen.extend(map(int,indices))
        accepted=[int(i) for i in indices]
        if len(seen)==20:
            accepted.remove(next(i for i in accepted if counts[i]==2))
        values={str(names[i]):i for i in accepted}
        return values,values.copy(),values.copy(),values.copy()
    packaged=[]
    monkeypatch.setattr(manager,'simulate_selected_tasks',simulate)
    monkeypatch.setattr(manager,'package_and_export_al_dataset',lambda output,rows,**kwargs:packaged.append(rows))
    manager.build_initial_dataset(path,tmp_path/'init.npz',20,4,1)
    assert len(seen)==21 and counts[seen[-1]]==2 and len(set(seen))==21
    mapping={str(name):int(level) for name,level in zip(names,counts)}
    assert Counter(mapping[name] for name in packaged[0][0])=={1:2,2:8,3:10}
