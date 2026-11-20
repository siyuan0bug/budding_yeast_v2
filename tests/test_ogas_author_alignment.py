"""Source-derived CFG and density-cap regressions, including provenance plumbing."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from budding_yeast_v2.data.systems.contracts import Candidate
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.utils.sampling.baseline.ogas import (
    ConditionalOGAS,OGASConfig,density_weights_v2,guided_noise,
)
from budding_yeast_v2.utils.sampling.generation.background import BackgroundOGAS
from budding_yeast_v2.utils.sampling.generation.online import OnlineGeneration
from budding_yeast_v2.utils.sampling.generation.replay import TrajectoryReplay
from budding_yeast_v2.utils.sampling.contracts import PoolInputs


@pytest.mark.parametrize('scale,expected',[(0.,1.),(1.,2.),(5.,6.)])
def test_cfg_uses_author_scale_convention(scale,expected):
    torch.testing.assert_close(guided_noise(torch.tensor(2.),torch.tensor(1.),scale),torch.tensor(expected))


def test_actual_ddpm_sampling_uses_author_cfg_not_w_plus_one():
    x=np.array([[-1.,-1.],[1.,1.]],np.float32)
    a=ConditionalOGAS(x,42,OGASConfig(width=4,diffusion_steps=1,sampling_steps=1,updates=1))
    class FixedNoise(torch.nn.Module):
        def forward(self,values,t,condition,drop=None):
            return torch.full_like(values,2. if drop is None else 1.)
    a.generator=FixedNoise();a.cycles=1
    rng=torch.Generator().set_state(a.noise_rng.get_state())
    torch.rand(3,generator=rng)  # target conditions precede initial noise
    noise=torch.randn((3,2),generator=rng)
    alpha=a.cumulative[0]
    expected=(noise-(1-alpha)*6/torch.sqrt(1-alpha))/alpha.sqrt()
    actual,_=a.sample_targets(3,x,exploration=False)
    np.testing.assert_allclose(actual,expected.numpy(),rtol=2e-6,atol=2e-6)


@pytest.mark.parametrize('fraction',[0.,.25,.5,1.])
@pytest.mark.parametrize('class_prior',[.5,.25])
def test_density_cap_and_class_prior_are_independent(fraction,class_prior):
    logits=torch.tensor([-100.,0.,2.,100.],dtype=torch.float64)
    # Author sign is inverted because its positive class is buffer, not reference.
    corrected=logits+np.log((1-class_prior)/class_prior)
    expected=corrected.clamp(-5,5).exp().clamp(max=1/max(fraction,1/len(logits)))
    expected/=expected.mean()
    actual=density_weights_v2(logits,buffer_reference_fraction=fraction,
                              discriminator_reference_prior=class_prior)
    torch.testing.assert_close(actual,expected)


def test_density_cap_changes_with_observed_reference_share():
    logits=torch.tensor([-2.,0.,2.,2.],dtype=torch.float64)
    a=density_weights_v2(logits,buffer_reference_fraction=.25)
    b=density_weights_v2(logits,buffer_reference_fraction=.5)
    assert not torch.allclose(a,b)
    with pytest.raises(ValueError):density_weights_v2(logits,buffer_reference_fraction=float('nan'))


def test_feedback_flags_eviction_training_and_resume():
    x=np.random.default_rng(42).normal(size=(8,4)).astype(np.float32)
    cfg=OGASConfig(paper_version='arxiv_v2',width=8,diffusion_steps=3,updates=2,batch_size=4,capacity=5)
    a=ConditionalOGAS(x,42,cfg)
    with pytest.raises(ValueError,match='provenance'):a.observe(x,np.ones(8))
    with pytest.raises(ValueError,match='boolean'):a.observe(x,np.ones(8),is_reference=np.ones(8))
    flags=np.array([True,False,True,False,False,True,False,False])
    a.observe(x,np.arange(8),is_reference=flags)
    np.testing.assert_array_equal(a.is_reference,flags[-5:])
    state=deepcopy(a.state_dict());b=ConditionalOGAS(x,7,cfg);b.load_state_dict(state)
    assert a.train_cycle(x)==b.train_cycle(x)
    np.testing.assert_array_equal(a.sample_targets(4,x)[0],b.sample_targets(4,x)[0])
    state.pop('implementation_version')
    with pytest.raises(ValueError,match='implementation changed'):b.load_state_dict(state)


def test_background_eviction_and_pending_checkpoint_preserve_flags():
    x=np.random.default_rng(42).normal(size=(8,4)).astype(np.float32)
    cfg=OGASConfig(paper_version='arxiv_v2',width=8,diffusion_steps=2,updates=1,batch_size=4,capacity=5)
    worker=BackgroundOGAS(ConditionalOGAS(x,42,cfg),2);other=None
    try:
        flags=np.array([True,False,True,False,True,False,False,True])
        worker.observe(x[:3],np.ones(3),is_reference=flags[:3])
        worker.observe(x[3:],np.ones(5),is_reference=flags[3:])
        state=worker.state_dict()
        other=BackgroundOGAS(ConditionalOGAS(x,1,cfg),2);other.load_state_dict(state)
        worker.request(x);other.request(x);worker.synchronize();other.synchronize()
        np.testing.assert_array_equal(worker.controller.is_reference,flags[3:])
        np.testing.assert_array_equal(worker.take(2),other.take(2))
        assert worker.status()['dropped_feedback_rows']==3
    finally:
        worker.close()
        if other is not None:other.close()


def test_replay_provenance_survives_ring_overwrite_and_checkpoint():
    p=np.arange(18,dtype=float).reshape(6,3);h=np.ones((6,2));y=np.ones((6,2,5))
    norm=NormalizationStats.fit(p,y)
    a=TrajectoryReplay(norm,p[:2],h[:2],y[:2],capacity=4,draws=8,seed=42)
    a.add(p[2:],h[2:],y[2:],is_reference=np.array([False,True,False,False]))
    b=TrajectoryReplay(norm,p[:2],h[:2],y[:2],capacity=4,draws=8,seed=1)
    b.restore(deepcopy(a.state()),p,h,y)
    expected=np.array([True,True,False,True,False,False])
    for row,other in zip(a.__getitems__(range(8)),b.__getitems__(range(8))):
        assert bool(row['online_is_reference'])==expected[int(row['online_row'])]
        assert row['online_is_reference']==other['online_is_reference']


def test_online_fallback_and_guided_query_provenance(tmp_path,monkeypatch):
    from budding_yeast_v2.utils.sampling.generation import online
    monkeypatch.setattr(online,'resource_slot',lambda *a,**k:nullcontext())
    view=PoolInputs(np.arange(3),np.arange(3.)[:,None],np.zeros((3,1)))
    candidates=[Candidate(i,str(i),{'keep':i},1,metadata={'generation_proposal_index':i}) for i in range(3)]
    submitted=[];consumed=[]
    (tmp_path/'_state').mkdir()
    queue=SimpleNamespace(limit=3,pending=[],submitted=0,submit=lambda rows,**kw:submitted.extend(rows))
    rng=SimpleNamespace(random=lambda count:np.zeros(count),
                        choice=lambda available,count,**kw:np.asarray(available)[:count])
    fake=SimpleNamespace(queue=queue,target=6,accepted=3,initial=SimpleNamespace(ids=np.arange(3)),
        config=SimpleNamespace(max_attempts_mult=30,bootstrap_steps=1,mixture_warmup_steps=1),
        domain=SimpleNamespace(bank=lambda count:(candidates,['a','b','c'],view),consume=consumed.append),
        bank_size=3,controller=SimpleNamespace(config=SimpleNamespace(exploration=.3),mean=np.zeros(2),scale=np.ones(2)),
        background=SimpleNamespace(take=lambda count:np.array([[0.,0.]]),status=lambda:{}),
        rng=rng,steps=2,sampler_not_ready_queries=0,exploration_queries=0,directory=tmp_path)
    OnlineGeneration._fill(fake)
    assert [c.metadata['ogas_is_reference'] for c in submitted]==[False,True,True]
    assert [c.overrides for c in submitted]==[c.overrides for c in candidates]
    assert fake.sampler_not_ready_queries==2
    # Completion order and a failed query must not change surviving flags.
    added=[]
    fake.controller=True;fake.replay=SimpleNamespace(add=lambda *a,**kw:added.append(kw['is_reference']))
    fake.pending=[];fake.config.flush_rows=100
    results=[SimpleNamespace(status=status,candidate=submitted[i],params=np.ones(1),y_init=np.ones(1),trajectory=np.ones((1,2)))
             for i,status in [(2,'success'),(1,'failed'),(0,'success')]]
    OnlineGeneration._accept(fake,results)
    np.testing.assert_array_equal(added[0],[True,False])
