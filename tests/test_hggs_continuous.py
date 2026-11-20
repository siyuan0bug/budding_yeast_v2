"""HGGS author-control-flow adaptation: optimizer, reference stats and recovery."""
from dataclasses import replace
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.utils.sampling.baseline.hggs import HGGSConfig
from budding_yeast_v2.utils.sampling.benchmark.protocol import prepare_protocol,effective_signatures
from budding_yeast_v2.utils.sampling.generation.runner import run_generation
from budding_yeast_v2.utils.sampling.generation.hggs_runner import MGSProvider
from budding_yeast_v2.utils.evaluation.checkpoints import load_sampling_checkpoint
from test_sampling_benchmark import make_pool,config


@pytest.fixture(autouse=True)
def isolated(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path/'locks'))
    monkeypatch.setenv('SAMPLING_IO_ADMISSION','warn')
    monkeypatch.setenv('SAMPLING_MEMORY_PRESSURE_POLICY','available')


def setup(tmp_path):
    source,arrays=make_pool(tmp_path/'pool',count=40)
    cfg=replace(config(),budgets=(4,5,6),epochs=(2,2,3),warmup=(0,0,0),
        loss='rel_l2',validation_metric='normalized_loss')
    return prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',cfg),arrays


def options(root,tmp_path,pool='mutation'):
    kw=dict(protocol=root,pool=pool,method='hggs_projected',output=tmp_path/'runs',workers=1,
        bank_size=4,query_batch_size=1,hggs_config=dict(coarse_multiplier=2,neighbors=2,
            output_sketch=4,mixture=False),diagnostic_only=True)
    if pool=='lhs':
        lhs=Path('/home/users/zsy/mydata/budding_yeast/offline_pools/lhs_matched_100000_seed42/lhs_specification.json')
        if not lhs.exists():pytest.skip('Frozen reviewed LHS fixture unavailable')
        from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
        bundle=tmp_path/'lhs_fixture';bundle.mkdir(exist_ok=True)
        shutil.copyfile(lhs,bundle/'lhs_specification.json')
        (bundle/'inventory.json').write_text(json.dumps(dict(range_fingerprint=RANGE_FINGERPRINT)))
        kw['lhs_config']=bundle/'lhs_specification.json'
    return kw


def test_author_default_gf_ratios_and_crossover_are_explicit():
    cfg=HGGSConfig().validate();final=2000;coarse=final*cfg.coarse_multiplier
    boundary=int(final*cfg.boundary_fraction)
    assert coarse==4000 and boundary/coarse==.2 and (final-boundary)/coarse==.3
    assert cfg.global_steps==5 and cfg.neighbors==5 and cfg.crossover_fraction==.6


@pytest.mark.parametrize('pool',['mutation','lhs'])
def test_real_cpu_chain_has_continuous_moments_and_frozen_reference(tmp_path,pool):
    root,arrays=setup(tmp_path);kw=options(root,tmp_path,pool)
    output=run_generation(**kw)
    done=json.loads((output/'_state/completion.json').read_text())
    assert done['main_model_resets']==1 and done['accepted']==6 and done['newly_accepted']==2
    stages=done['stage_records']
    assert [r['optimizer_steps_total'] for r in stages]==[2,6,12]
    assert [r['adam_step'] for r in stages]==[2,6,12]
    assert [r['optimizer_steps_delta'] for r in stages]==[2,4,6]
    assert done['training_costs']['primary']['models']==1
    assert done['training_costs']['pilot']['models']==1
    gf=json.loads((output/'_state/hggs_gf.json').read_text())
    committed=json.loads((output/'_state/generation.json').read_text())
    known=np.asarray(gf['coarse_source_ids'],dtype=int)
    assert set(effective_signatures(arrays['params'][known],arrays['y_init'][known]))<=set(committed['domain']['seen'])
    assert len(gf['stages'])==3 and [r['adam_step'] for r in gf['stages']]==[2,4,6]
    assert gf['pilot_checkpoint']['policy'].startswith('live_current_model')
    with np.load(root/f'{pool}_split.npz') as z:
        reference=NormalizationStats.fit(arrays['params'][z['initial']],arrays['data'][z['initial']])
        assert not set(gf['coarse_source_ids'])&set(np.r_[z['validation'],z['test']])
    for ck in [output/'_state/primary/last_weights.ckpt',output/'_state/hggs_pilot/primary/_state/last_weights.ckpt']:
        _,stats,payload=load_sampling_checkpoint(ck)
        assert torch.equal(stats.p_std,reference.p_std)
        assert torch.equal(stats.mean_y,reference.mean_y)
        assert payload['preprocessing_contract']['dataset']['normalization_reference']['count']==4
    assert not list(output.glob('rounds/**/*.npz'))
    assert (output/'eval/per_synth_test_metrics.csv').exists()
    assert run_generation(**kw)==output


@pytest.mark.parametrize('when',['before_query','after_commit'])
def test_boundary_recovery_does_not_reset_adam_or_duplicate_ode(tmp_path,when):
    root,_=setup(tmp_path);kw=options(root,tmp_path)
    original=MGSProvider.acquire;raised=False
    def interrupted(self,target,model,norm):
        nonlocal raised
        if target==5 and not raised:
            raised=True
            if when=='after_commit':original(self,target,model,norm)
            raise RuntimeError('injected HGGS acquisition interruption')
        return original(self,target,model,norm)
    with patch.object(MGSProvider,'acquire',interrupted):
        with pytest.raises(RuntimeError,match='injected HGGS'):run_generation(**kw)
    output=run_generation(**kw);done=json.loads((output/'_state/completion.json').read_text())
    assert done['accepted']==6 and done['attempted_inputs']==2
    assert [r['adam_step'] for r in done['stage_records']]==[2,6,12]
    assert done['solver_attempts_exact'] and done['incomplete_query_attempts']==0
