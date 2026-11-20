"""Online timing, cost bounds, immutable data and a real CPU ODE/FNO workflow."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.data.dataset_archive import TrajectoryStore
from budding_yeast_v2.utils.sampling.generation.replay import TrajectoryReplay
from budding_yeast_v2.utils.sampling.generation.async_queries import AsyncQueries,query_costs
from budding_yeast_v2.utils.sampling.generation.online import batch_difficulty,OnlineConfig
from budding_yeast_v2.utils.sampling.generation.background import BackgroundOGAS
from budding_yeast_v2.utils.sampling.baseline.ogas import ConditionalOGAS,OGASConfig,density_weights_v2
from budding_yeast_v2.utils.sampling.generation.online_runner import run_online_generation
from budding_yeast_v2.utils.sampling.benchmark.protocol import prepare_protocol
from test_sampling_benchmark import make_pool,config


def test_online_signal_matches_declared_formula_not_a_clipped_loss():
    y=torch.ones(2,3,4);p=y.clone();p[0]*=3
    scaled=batch_difficulty(p,y,'scaled_mse',1e-8)
    np.testing.assert_allclose(scaled,[4/(np.sqrt(12)+1e-8),0])
    relative=batch_difficulty(p,y,'relative_l2',.01)
    np.testing.assert_allclose(relative,[4/np.hypot(2,.01),0])
    with pytest.raises(FloatingPointError):batch_difficulty(p*float('inf'),y,'scaled_mse',1e-8)


def test_latest_v2_density_recipe_is_log_clipped_capped_and_batch_normalized():
    logits=torch.tensor([-100.,0.,100.],dtype=torch.float64)
    expected=torch.tensor([np.exp(-5),1.,2.],dtype=torch.float64);expected/=expected.mean()
    torch.testing.assert_close(density_weights_v2(logits,buffer_reference_fraction=.5),expected)
    x=np.random.default_rng(2).normal(size=(6,4)).astype(np.float32)
    cfg=OGASConfig(paper_version='arxiv_v2',width=12,diffusion_steps=4,updates=1,batch_size=4)
    model=ConditionalOGAS(x,42,cfg);model.observe(x,np.arange(6)/6,is_reference=np.ones(6,dtype=bool))
    assert isinstance(model.discriminator_optimizer,torch.optim.AdamW)
    assert not model.generator.residual_modulation
    report=model.train_cycle(x);assert np.isfinite(report['diffusion_loss'])


def test_replay_is_bounded_restores_rng_and_keeps_physical_inputs():
    p=np.arange(18,dtype=float).reshape(6,3);h=np.ones((6,2));y=np.ones((6,2,5))
    norm=NormalizationStats.fit(p,y)
    a=TrajectoryReplay(norm,p[:2],h[:2],y[:2],capacity=4,draws=7,seed=42)
    a.add(p[2:],h[2:],y[2:]);assert len(a)==7 and a.size==4 and a.total_inserted==6
    state=deepcopy(a.state())
    b=TrajectoryReplay(norm,p[:2],h[:2],y[:2],capacity=4,draws=7,seed=1)
    b.restore(state,p,h,y)
    rows=a.__getitems__([0,1,2]);other=b.__getitems__([0,1,2])
    for first,second in zip(rows,other):
        for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
        assert first['online_params'].dtype==torch.float64 and first['x'].dtype==torch.float32
        i=int(first['online_row']);np.testing.assert_array_equal(first['online_params'],p[i]);assert i>=2
    a.record_exposure([2,2,5]);assert a.exposure=={2:2,5:1}


def test_async_queue_nonblocking_out_of_order_and_exact_ledger(tmp_path):
    slow=threading.Event();fast=threading.Event()
    def solve(candidate):
        if candidate.metadata['generation_proposal_index']==0:slow.wait(5)
        else:fast.set()
        return SimpleNamespace(status='success',diagnostics={'attempts':[{}]})
    def candidate(i):return SimpleNamespace(metadata={'generation_proposal_index':i},overrides={'test':i})
    with ThreadPoolExecutor(max_workers=2) as executor:
        queue=AsyncQueries(executor,tmp_path/'costs.jsonl',2,function=solve)
        try:
            queue.submit([candidate(0),candidate(1)],model_step=7)
            with pytest.raises(ValueError,match='full'):queue.submit([candidate(2)],model_step=8)
            assert fast.wait(5)
            # The unfinished first input does not hold back a completed second.
            results=queue.poll(block=True)
            assert len(results)==1 and len(queue.pending)==1
        finally:slow.set()
        while queue.pending:queue.poll(block=True)
    assert query_costs(queue.path)==dict(attempted_inputs=2,finished_inputs=2,failed_inputs=0,
        solver_attempts=2,solver_attempts_exact=True,incomplete_query_attempts=0)


def test_generator_updates_do_not_block_main_loop_and_are_recoverable():
    x=np.random.default_rng(2).normal(size=(8,4)).astype(np.float32)
    cfg=OGASConfig(paper_version='arxiv_v2',width=12,diffusion_steps=4,updates=1,batch_size=4)
    controller=ConditionalOGAS(x,42,cfg)
    original=controller.train_cycle;entered=threading.Event();release=threading.Event()
    def slow(reference):
        entered.set();release.wait(5);return original(reference)
    controller.train_cycle=slow
    worker=BackgroundOGAS(controller,4);other=None
    try:
        worker.observe(x,np.arange(8)/8,is_reference=np.arange(8)%2==0);worker.request(x)
        assert entered.wait(5)
        # A simulated surrogate step can submit feedback/read its ready cache
        # while the generator update is still deliberately blocked.
        worker.observe(x[:2],np.ones(2),is_reference=np.array([False,True]));assert worker.take(2).shape==(0,4)
        assert worker.status()['busy']
        release.set();worker.synchronize();state=worker.state_dict()
        other=BackgroundOGAS(ConditionalOGAS(x,1,cfg),4);other.load_state_dict(state)
        np.testing.assert_array_equal(worker.take(4),other.take(4))
        worker.request(x);other.request(x);worker.synchronize();other.synchronize()
        np.testing.assert_array_equal(worker.take(4),other.take(4))
    finally:
        release.set();worker.close()
        if other is not None:other.close()


def test_async_generator_failure_is_not_silently_treated_as_low_score():
    x=np.ones((2,3),np.float32)
    worker=BackgroundOGAS(ConditionalOGAS(x,42,OGASConfig(width=4,diffusion_steps=2,updates=1)),2)
    try:
        worker.observe(x,np.array([np.nan,1.]));worker.request(x)
        with pytest.raises(RuntimeError,match='generator failed'):worker.synchronize()
    finally:worker.close()


@pytest.mark.parametrize('kwargs',[dict(epochs=1),dict(generation_epochs=200),dict(replay_capacity=0),
                                   dict(signal='real_badcase'),dict(signal_eps=0)])
def test_invalid_online_protocol_rejected(kwargs):
    with pytest.raises(ValueError):OnlineConfig(**kwargs).validate()


@pytest.mark.parametrize('method',['ogas_async_projected','random_async'])
def test_actual_online_canonical_ode_fno_and_resume_after_eval_failure(tmp_path,method,monkeypatch):
    from budding_yeast_v2.utils.sampling.generation.cohorts import FrozenCohorts
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path/'locks'))
    monkeypatch.setenv('SAMPLING_IO_ADMISSION','warn')
    monkeypatch.setenv('SAMPLING_MEMORY_PRESSURE_POLICY','available')
    monkeypatch.setenv('SAMPLING_MAX_LOAD_PER_CPU','100')
    source,_=make_pool(tmp_path/'pool')
    cfg=replace(config(),budgets=(2,3),epochs=(1,1),warmup=(0,0))
    root=prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',cfg)
    kwargs=dict(workers=1,bank_size=4,diagnostic_only=True,
        online_config=dict(epochs=2,warmup=0,generation_epochs=1,steps_per_epoch=2,
                           generator_every_steps=1,bootstrap_steps=1,replay_capacity=8,mixture_warmup_steps=1),
        ogas_config=dict(width=12,diffusion_steps=4,updates=1,batch_size=4))
    original=FrozenCohorts.evaluate
    with monkeypatch.context() as local:
        local.setattr(FrozenCohorts,'evaluate',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('eval interrupted')))
        with pytest.raises(RuntimeError,match='eval interrupted'):
            run_online_generation(root,'mutation',method,tmp_path/'results',**kwargs)
    directory=run_online_generation(root,'mutation',method,tmp_path/'results',**kwargs)
    report=json.loads((directory/'_state/completion.json').read_text())
    assert report['accepted']==3 and report['newly_accepted']==1
    assert report['asynchronous_solver_training'] and report['training_exposure']=='dynamic'
    assert report['normalization']=='initial_train_frozen' and report['sources_unchanged']
    assert report['optimizer_steps']==4 and report['training_draws']==16
    assert report['attempted_inputs']>=1 and report['solver_attempts_exact']
    assert report['accepted_by_epoch'][-1]['accepted']==3
    assert report['best_checkpoint_exposure']['normalization']=='initial training frozen'
    with (directory/'stream/dataset.json').open() as f:snapshot=json.load(f)
    store=TrajectoryStore(snapshot['source']);raw=store.read(snapshot['rows'],('params','y_init','data'))
    assert raw['data'].shape==(3,39,500) and np.isfinite(raw['data']).all()
    initial=json.loads((directory/'stream/initial_dataset.json').read_text())
    assert len(initial['rows'])==2  # stats cohort stays initial; final source has all three
    assert (directory/'eval/global_metrics_denorm.json').exists()
    assert '[Epoch ' in (directory/'train.log').read_text()
    assert run_online_generation(root,'mutation',method,tmp_path/'results',**kwargs)==directory
    (directory/'stream/history.csv').write_text('corrupt')
    with pytest.raises(ValueError,match='artifact changed'):
        run_online_generation(root,'mutation',method,tmp_path/'results',**kwargs)


@pytest.mark.parametrize('method',['random_async','ogas_async_projected'])
def test_periodic_training_resume_restores_stream_sampler_and_optimizer(tmp_path,monkeypatch,method):
    from pytorch_lightning.callbacks import Callback
    from budding_yeast_v2.utils.sampling.generation import online_runner
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path/'locks'))
    monkeypatch.setenv('SAMPLING_IO_ADMISSION','warn')
    monkeypatch.setenv('SAMPLING_MEMORY_PRESSURE_POLICY','available')
    monkeypatch.setenv('SAMPLING_MAX_LOAD_PER_CPU','100')
    source,_=make_pool(tmp_path/'pool')
    root=prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',
        replace(config(),budgets=(2,3),epochs=(1,1),warmup=(0,0)))
    kwargs=dict(workers=1,bank_size=4,diagnostic_only=True,
        online_config=dict(epochs=8,warmup=0,generation_epochs=2,steps_per_epoch=1,bootstrap_steps=1),
        ogas_config=dict(width=12,diffusion_steps=4,updates=1,batch_size=4))
    original=online_runner.fit_round
    class Interrupt(Callback):
        def on_train_epoch_start(self,trainer,pl_module):
            if trainer.current_epoch==6:
                # Simulate a best produced after the epoch5 durable checkpoint.
                # Recovery must restore the matched model/exposure from epoch5.
                for path in (tmp_path/'results').glob('*/mutation/*/seed_42/stream/model.ckpt'):
                    path.write_text('uncommitted future model')
                raise RuntimeError('controlled interruption after periodic checkpoint')
    def interrupted(*args,**kw):
        kw['extra_callbacks']=tuple(kw['extra_callbacks'])+(Interrupt(),)
        return original(*args,**kw)
    with monkeypatch.context() as local:
        local.setattr(online_runner,'fit_round',interrupted)
        with pytest.raises(RuntimeError,match='controlled interruption'):
            run_online_generation(root,'mutation',method,tmp_path/'results',**kwargs)
    result=run_online_generation(root,'mutation',method,tmp_path/'results',**kwargs)
    report=json.loads((result/'_state/completion.json').read_text())
    assert report['accepted']==3 and report['optimizer_steps']==8 and report['training_draws']==32
    assert len(report['accepted_by_epoch'])==8
    assert report['attempted_inputs']==1  # committed ODE data are not regenerated
