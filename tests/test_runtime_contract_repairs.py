"""Authorized 2026-10-19 repairs: artificial/public-WT fixtures, no real tuning."""
import ast
import copy
from concurrent.futures import ProcessPoolExecutor, Future
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from budding_yeast_v2.data.normalization import normalize_y, normalize_p, denormalize_y, denormalize_p
from budding_yeast_v2.data.preprocessing_contract import validate_preprocessing_state, resolve_checkpoint_norm
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.models.ours.hano import HANO
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.utils.training.losses import LOSS_REGISTRY
from budding_yeast_v2.data import mutant_aware as ma, mass_dataset_generator as mass, simulation_guards as guards

torch.set_num_threads(1)
MODES = ['log1p', 'linear', 'asinh_z', 'asinh_scale']


def stats(mode):
    return SimpleNamespace(
        mean_y=torch.full((1, 39, 1), 0.2), std_y=torch.full((1, 39, 1), 0.7),
        p_mean=torch.full((1, 141), 0.2), p_std=torch.full((1, 141), 0.7),
        norm_type=mode, s_c=torch.full((1, 39, 1), 2.0) if mode == 'asinh_scale' else None,
        t_max=210.,
    )


def make_dm(tmp_path, mode='log1p', *, state=None, filename='artificial_210min.npz', fit_scope=None, seed=42):
    path = tmp_path / filename
    if not path.exists():
        rng = np.random.RandomState(17)
        raw = rng.uniform(0.1, 3., (62, 39, 16)).astype(np.float32)
        params = rng.uniform(0.1, 2., (62, 141))
        params[:, 0] = 0.5
        np.savez(path, data=raw, params=params, param_names=np.array([f'p{i}' for i in range(141)]),
                 mutant_names=np.array(['1_artificial', '2_artificial'] + [f'RANDOM_{i}' for i in range(60)]),
                 pattern_labels=np.array(['Pattern_B: Single-Division (==1)'] * 62))
    dm = YeastDataModule(dataset_path=str(path), norm_type=mode, num_workers=0,
                         use_adj_matrix=False, preprocessing_state=state, fit_scope=fit_scope, seed=seed)
    dm.setup('fit')
    return dm


@pytest.mark.parametrize('mode', MODES)
def test_independent_transform_formulas_and_roundtrip(mode):
    s = stats(mode)
    y = torch.full((2, 39, 8), 10.)
    transformed = {'log1p': np.log1p(10.), 'linear': 10., 'asinh_z': np.log1p(10.),
                   'asinh_scale': np.arcsinh(10./4.)}[mode]
    z = normalize_y(y, s.mean_y, s.std_y, mode, s.s_c)
    torch.testing.assert_close(z, torch.full_like(z, (transformed - .2)/.7))
    torch.testing.assert_close(denormalize_y(z, s.mean_y, s.std_y, mode, s.s_c), y)
    p = torch.full((2,141), 3.)
    encoded = {'log1p': (3.-.2)/.7, 'linear': (3.-.2)/.7,
               'asinh_z': np.arcsinh((3.-.2)/.7), 'asinh_scale': (np.log10(3.)-.2)/.7}[mode]
    zp = normalize_p(p, s.p_mean, s.p_std, mode)
    torch.testing.assert_close(zp, torch.full_like(zp, encoded))
    torch.testing.assert_close(denormalize_p(zp, s.p_mean, s.p_std, mode), p)
    s.p_std[:, 0] = 0.
    if mode in ('asinh_z', 'asinh_scale'):
        assert (normalize_p(p, s.p_mean, s.p_std, mode)[:, 0] == 0).all()


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('fit_scope', [None, 'all_synthetic_legacy'])
def test_explicit_fit_scope_and_equations(tmp_path, mode, fit_scope):
    dm = make_dm(tmp_path, mode, fit_scope=fit_scope)
    synth = (list(dm._train_idx_synth) if fit_scope is None else
             [i for i, name in enumerate(dm._mutant_names) if name.startswith('RANDOM_')])
    y = dm._Y
    if mode == 'asinh_scale':
        sc = np.maximum(np.quantile(np.maximum(y[synth].numpy(), 0), .9, axis=(0,2), keepdims=True), .01)
        torch.testing.assert_close(dm.s_c, torch.tensor(sc,dtype=torch.float32))
        yt = torch.asinh(y.clamp(min=0)/(2*dm.s_c))
    else:
        yt = y if mode == 'linear' else torch.log1p(y.clamp(min=0))
    assert torch.equal(dm.mean_y, yt[synth].mean((0,2),keepdim=True))
    assert torch.equal(dm.std_y, yt[synth].std((0,2),keepdim=True)+1e-6)
    p = torch.tensor(dm._param_conds,dtype=torch.float32)
    pt = torch.log10(p.clamp(min=1e-8)) if mode == 'asinh_scale' else p
    assert torch.equal(dm.p_mean, pt[synth].mean(0,keepdim=True))
    expected_std=pt[synth].std(0,keepdim=True)+(1e-6 if mode in ('log1p','linear') else 0)
    assert torch.equal(dm.p_std,expected_std)
    assert dm.preprocessing_state()['fit_scope'] == (fit_scope or 'train_synthetic')
    assert dm.hparams.fit_scope == dm.fit_scope


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('split', ['val', 'test', 'real'])
def test_holdout_values_cannot_change_fitted_statistics_or_training_tensors(tmp_path, mode, split):
    dm = make_dm(tmp_path, mode)
    indices = getattr(dm, {'val': '_val_idx_synth', 'test': '_test_idx_synth', 'real': '_test_idx_real'}[split])
    assert len(indices) > 0
    with np.load(dm.dataset_path) as archive:
        values = {key: archive[key] for key in archive.files}
    values['data'][indices] = values['data'][indices] * 1000 + 100
    values['params'][indices] = values['params'][indices] * 1000 + 100
    path = tmp_path / f'changed_{split}_210min.npz'
    np.savez(path, **values)
    changed = make_dm(tmp_path, mode, filename=path.name)
    for key in ('mean_y', 'std_y', 'p_mean', 'p_std', 's_c'):
        a, b = getattr(dm, key), getattr(changed, key)
        assert (a is None and b is None) or torch.equal(a, b), key
    assert dm._train_idx_synth == changed._train_idx_synth
    for a, b in zip(dm.train_dataset_synth.tensors, changed.train_dataset_synth.tensors):
        assert torch.equal(a, b)


@pytest.mark.parametrize('mode', MODES)
def test_training_values_do_control_fitted_statistics(tmp_path, mode):
    dm = make_dm(tmp_path, mode)
    with np.load(dm.dataset_path) as archive:
        values = {key: archive[key] for key in archive.files}
    values['data'][dm._train_idx_synth] += 20
    values['params'][dm._train_idx_synth] += 20
    path = tmp_path / 'changed_train_210min.npz'
    np.savez(path, **values)
    changed = make_dm(tmp_path, mode, filename=path.name)
    assert not torch.equal(dm.mean_y, changed.mean_y)
    assert not torch.equal(dm.p_mean, changed.p_mean)
    if mode == 'asinh_scale':
        assert not torch.equal(dm.s_c, changed.s_c)


@pytest.mark.parametrize('mode', MODES)
def test_legacy_payload_keeps_original_scope_and_exact_statistics(tmp_path, mode):
    old = make_dm(tmp_path, mode, fit_scope='all_synthetic_legacy')
    state = old.preprocessing_state()
    restored = make_dm(tmp_path, mode, state=state)
    assert restored.fit_scope == 'all_synthetic_legacy'
    for a, b in zip(old.train_dataset_synth.tensors, restored.train_dataset_synth.tensors):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match='fit_scope'):
        make_dm(tmp_path, mode, state=state, fit_scope='train_synthetic')
    new = make_dm(tmp_path, mode)
    lit = YeastLitModule(model_name='fno', width=8, modes=2, n_blocks=1, norm_type=mode)
    lit.bind_datamodule_stats(old)
    checkpoint = {'hyper_parameters': dict(lit.hparams), 'state_dict': lit.state_dict()}
    lit.on_save_checkpoint(checkpoint)
    loaded = YeastLitModule(model_name='fno', width=8, modes=2, n_blocks=1, norm_type=mode)
    loaded.on_load_checkpoint(checkpoint)
    loaded.bind_datamodule_stats(restored)
    with pytest.raises(ValueError, match='fit_scope'):
        loaded.bind_datamodule_stats(new)


@pytest.mark.parametrize('mode', MODES)
def test_payload_free_checkpoint_cannot_silently_bind_new_statistics(tmp_path, mode):
    from budding_yeast_v2.eval import load_model_from_checkpoint
    from budding_yeast_v2.data.preprocessing_contract import resolve_checkpoint_fit_scope
    old = make_dm(tmp_path, mode, fit_scope='all_synthetic_legacy')
    lit = YeastLitModule(model_name='fno', width=8, modes=2, n_blocks=1, norm_type=mode)
    checkpoint = {'hyper_parameters': dict(lit.hparams), 'state_dict': lit.state_dict(),
                  'datamodule_hyper_parameters': {'norm_type': mode, 'seed': 42}}
    path = tmp_path / 'legacy_without_preprocessing.ckpt'
    torch.save(checkpoint, path)
    with pytest.warns(UserWarning, match='Legacy checkpoint'):
        loaded, hp = load_model_from_checkpoint(path)
    assert hp['normalization_fit_scope'] == resolve_checkpoint_fit_scope(checkpoint) == 'all_synthetic_legacy'
    new = make_dm(tmp_path, mode)
    with pytest.raises(ValueError, match='fit_scope'):
        loaded.bind_datamodule_stats(new)
    restored = make_dm(tmp_path, hp['norm_type'], state=loaded._preprocessing_state,
                       fit_scope=hp['normalization_fit_scope'])
    loaded.bind_datamodule_stats(restored)
    assert torch.equal(loaded.mean_y, old.mean_y)
    assert torch.equal(loaded.p_mean, old.p_mean)


def test_scope_validation_and_run_tags(tmp_path):
    from budding_yeast_v2.utils.experiments.paths import generate_hparams_signature
    from budding_yeast_v2.data.preprocessing_contract import resolve_checkpoint_fit_scope
    dm = make_dm(tmp_path)
    state = dm.preprocessing_state()
    for scope in (None, 'unknown', ''):
        bad = copy.deepcopy(state)
        bad['fit_scope'] = scope
        with pytest.raises(ValueError, match='fit_scope'):
            validate_preprocessing_state(bad)
    assert resolve_checkpoint_fit_scope({'preprocessing_contract': state}) == 'train_synthetic'
    base = dict(width=8, modes=2, batch_size=4, lr=.001)
    new = generate_hparams_signature(**base, normalization_fit_scope='train_synthetic')
    old = generate_hparams_signature(**base)
    assert new != old and new.endswith('trainfit1')


@pytest.mark.parametrize('mode', MODES)
def test_saved_payload_replays_nondefault_source_split_seed_without_datamodule_hparams(tmp_path, mode):
    from budding_yeast_v2.eval import load_model_from_checkpoint
    dm = make_dm(tmp_path, mode, seed=7)
    lit = YeastLitModule(model_name='fno', width=8, modes=2, n_blocks=1, norm_type=mode)
    lit.bind_datamodule_stats(dm)
    ck = {'hyper_parameters': dict(lit.hparams), 'state_dict': lit.state_dict()}
    lit.on_save_checkpoint(ck)
    path = tmp_path / 'seed7.ckpt'
    torch.save(ck, path)
    loaded, hp = load_model_from_checkpoint(path)
    assert hp['data_seed'] == 7
    restored = make_dm(tmp_path, mode, seed=hp['data_seed'], state=loaded._preprocessing_state,
                       fit_scope=hp['normalization_fit_scope'])
    assert restored._train_idx_synth == dm._train_idx_synth
    loaded.bind_datamodule_stats(restored)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('loss_name', ['pinn_residual','pino','hybrid_pino_colloc','hybrid_all'])
def test_loss_startup_and_physical_decoders(mode, loss_name):
    s=stats(mode)
    lit=YeastLitModule(model_name='fno',loss_type=loss_name,width=8,modes=2,n_blocks=1,norm_type=mode)
    lit._trainer=SimpleNamespace(datamodule=s)
    lit.on_train_start()
    loss=lit.criterion
    assert loss.norm_type == mode
    y=torch.full((2,39,8),10.)
    z=normalize_y(y,s.mean_y,s.std_y,mode,s.s_c)
    torch.testing.assert_close(loss._denormalize_y(z),y)
    p=torch.full((2,141),3.)
    zp=normalize_p(p,s.p_mean,s.p_std,mode)
    physical=loss._denormalize_p(zp)
    assert all(torch.isfinite(value).all() for value in physical.values())
    assert abs(float(physical['B0'][0])-3.) < 1e-5
    pred=(z+.01).detach().requires_grad_(True)
    result=loss(pred,z,zp,torch.ones(2))
    assert torch.isfinite(result)
    result.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('loss_name', ['pinn_residual','pino','hybrid_pino_colloc','hybrid_all'])
def test_physics_terms_after_warmup_are_finite(mode,loss_name):
    s=stats(mode)
    loss=LOSS_REGISTRY[loss_name](s.mean_y,s.std_y,s.p_mean,s.p_std,
                                norm_type=mode,s_c=s.s_c)
    defaults=ma.get_default_params()
    defaults.update({'init_CDH1T':1.,'init_CDH1':.9304992,'init_BUB2':.2})
    p=torch.tensor([[defaults[key] for key in ma.PARAM_KEYS]],dtype=torch.float32)
    y=torch.tensor(ma.initial_conds(),dtype=torch.float32)[None,:,None].expand(1,39,12)
    target=normalize_y(y,s.mean_y,s.std_y,mode,s.s_c)
    pred=(target+1e-4).detach().requires_grad_(True)
    loss.set_epoch(60)
    value=loss(pred,target,normalize_p(p,s.p_mean,s.p_std,mode),torch.ones(1))
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize('mode', MODES)
def test_hano_and_lightning_share_inverse(mode):
    s=stats(mode)
    lit=YeastLitModule(model_name='hano',loss_type='mse_only',width=8,n_blocks=1)
    lit.bind_datamodule_stats(s)
    y=torch.full((2,39,8),10.)
    z=normalize_y(y,s.mean_y,s.std_y,mode,s.s_c)
    torch.testing.assert_close(lit.model._denorm_y(z),y)
    torch.testing.assert_close(lit.model._denorm_y(z[:,:,0]),y[:,:,0])
    torch.testing.assert_close(lit.model._norm_y(y),z)
    torch.testing.assert_close(lit.denormalize_y(z),y)
    p=torch.full((2,141),3.)
    torch.testing.assert_close(lit.model._denorm_p(normalize_p(p,s.p_mean,s.p_std,mode)),p)


@pytest.mark.parametrize('mode', MODES)
def test_hpo_uses_physical_mae_and_keeps_real_objective(mode):
    from budding_yeast_v2.tune import evaluate_real_metrics
    dm=YeastDataModule(norm_type=mode,use_adj_matrix=False)
    for key,val in vars(stats(mode)).items(): setattr(dm,key,val)
    target=dm.normalize_y(torch.full((1,39,8),10.))
    expected=dm.normalize_y(torch.full_like(target,20.))
    dm.test_dataset_real=TensorDataset(torch.zeros(1,39,2,8),torch.zeros(1,141),target)
    class Probe:
        def bind_datamodule_stats(self,dm): self.bound=True
        def forward_ic_time(self,x,p): return expected
    probe=Probe()
    metrics=evaluate_real_metrics(probe,dm,'cpu')
    assert probe.bound and metrics['MAE'] == pytest.approx(10.,abs=1e-5)
    assert metrics['MSE'] == pytest.approx(100.,abs=1e-3)
    source=ast.parse((ROOT/'tune.py').read_text())
    assert any(isinstance(node,ast.Return) and isinstance(node.value,ast.Name) and node.value.id=='real_mae'
               for node in ast.walk(source))


@pytest.mark.parametrize('recorded,old_trials,allowed',[(None,False,True),(None,True,False),
                                                     ('real_physical_mae_v1',True,True),('old',True,False)])
def test_hpo_objective_versions_are_not_mixed(recorded,old_trials,allowed):
    from budding_yeast_v2.tune import bind_objective_contract
    class Study:
        user_attrs={} if recorded is None else {'objective_contract':recorded}
        trials=[object()] if old_trials else []
        def set_user_attr(self,key,value):self.user_attrs[key]=value
    study=Study()
    if recorded == 'real_physical_mae_v1':
        study.user_attrs['normalization_fit_scope'] = 'train_synthetic'
    if allowed:
        bind_objective_contract(study)
        assert study.user_attrs['objective_contract']=='real_physical_mae_v1'
        assert study.user_attrs['normalization_fit_scope']=='train_synthetic'
    else:
        with pytest.raises(ValueError,match='NEW study_name'):
            bind_objective_contract(study)


@pytest.mark.parametrize('scope', [None, 'all_synthetic_legacy'])
def test_hpo_rejects_prior_fit_protocol_without_changing_objective(scope):
    from budding_yeast_v2.tune import bind_objective_contract
    class Study:
        user_attrs = {'objective_contract': 'real_physical_mae_v1'}
        trials = [object()]
        def set_user_attr(self, key, value): self.user_attrs[key] = value
    study = Study()
    if scope is not None:
        study.user_attrs['normalization_fit_scope'] = scope
    with pytest.raises(ValueError, match='NEW study_name'):
        bind_objective_contract(study)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('model_name',['fno','hano'])
def test_checkpoint_contract_roundtrip_and_dataset_mismatch(tmp_path,mode,model_name):
    from budding_yeast_v2.eval import load_model_from_checkpoint
    dm=make_dm(tmp_path,mode)
    lit=YeastLitModule(model_name=model_name,loss_type='mse_only',width=8,modes=2,n_blocks=1,norm_type=mode)
    lit.bind_datamodule_stats(dm)
    checkpoint={'hyper_parameters':dict(lit.hparams),'state_dict':lit.state_dict()}
    lit.on_save_checkpoint(checkpoint)
    path=tmp_path/'checkpoint.ckpt';torch.save(checkpoint,path)
    loaded,hp=load_model_from_checkpoint(path)
    torch.testing.assert_close(loaded.denormalize_y(dm.test_dataset_real.tensors[2]),
                               dm.denormalize_y(dm.test_dataset_real.tensors[2]))
    restored=make_dm(tmp_path,mode,state=loaded._preprocessing_state)
    assert torch.equal(restored.mean_y,dm.mean_y)
    assert restored._train_idx_synth == dm._train_idx_synth
    loaded.bind_datamodule_stats(restored)
    if model_name=='hano':
        assert loaded.model.norm_type==mode
        torch.testing.assert_close(loaded.model._denorm_y(dm.test_dataset_real.tensors[2]),
                                   dm.denormalize_y(dm.test_dataset_real.tensors[2]))
    # A same-shaped but different dataset must not silently refit preprocessing.
    with np.load(dm.dataset_path) as z: values={key:z[key] for key in z.files}
    values['data']=values['data'].copy();values['data'][2,0,0]+=0.1
    changed=tmp_path/'changed_210min.npz';np.savez(changed,**values)
    with pytest.raises(ValueError,match='dataset'):
        make_dm(tmp_path,mode,state=loaded._preprocessing_state,filename=changed.name)


def test_checkpoint_rejects_bad_state_and_conflicting_norm(tmp_path):
    dm=make_dm(tmp_path)
    state=dm.preprocessing_state()
    for key in ['mean_y','std_y','p_mean','p_std']:
        bad=copy.deepcopy(state);bad[key].flatten()[0]=float('nan')
        with pytest.raises(ValueError):validate_preprocessing_state(bad)
    bad=copy.deepcopy(state);bad['norm_type']='asinh_scale'
    with pytest.raises(ValueError,match='s_c'):validate_preprocessing_state(bad)
    with pytest.raises(ValueError,match='contradicts'):
        resolve_checkpoint_norm({'preprocessing_contract':state},'linear')


@pytest.mark.parametrize('loss_name',['pinn_residual','pino','hybrid_pino_colloc','hybrid_all','t_smooth'])
def test_lightning_strict_checkpoint_restores_physics_stats(tmp_path,loss_name):
    import pytorch_lightning as pl
    dm=make_dm(tmp_path,'asinh_scale')
    lit=YeastLitModule(model_name='fno',loss_type=loss_name,width=8,modes=2,n_blocks=1,
                       norm_type='asinh_scale')
    lit.bind_datamodule_stats(dm)
    ck={'hyper_parameters':dict(lit.hparams),'state_dict':lit.state_dict(),
        'pytorch-lightning_version':pl.__version__}
    lit.on_save_checkpoint(ck)
    path=tmp_path/'strict.ckpt';torch.save(ck,path)
    loaded=YeastLitModule.load_from_checkpoint(path,strict=True)
    torch.testing.assert_close(loaded.denormalize_y(dm.test_dataset_real.tensors[2]),
                               dm.denormalize_y(dm.test_dataset_real.tensors[2]))
    if hasattr(loaded.criterion,'_denormalize_y'):
        torch.testing.assert_close(loaded.criterion._denormalize_y(dm.test_dataset_real.tensors[2]),
                                   dm.denormalize_y(dm.test_dataset_real.tensors[2]))
    bad=copy.deepcopy(ck)
    if 'criterion.mean_y' in bad['state_dict']:
        bad['state_dict']['criterion.mean_y'].fill_(0)
        with pytest.raises(ValueError,match='contradict'):
            loaded.on_load_checkpoint(bad)






def test_al_mutable_training_indices_do_not_change_source_contract(tmp_path):
    dm=make_dm(tmp_path)
    original=copy.deepcopy(dm.preprocessing_state()['split'])
    original_stats={key:getattr(dm,key).clone() for key in ('mean_y','std_y','p_mean','p_std')}
    dm._train_idx_synth=np.array([1000,1001])
    dm.setup('fit')
    assert dm.preprocessing_state()['split']==original
    assert all(torch.equal(getattr(dm,key),value) for key,value in original_stats.items())




@pytest.mark.parametrize('name',['biom_refiner_plus','biom_refiner_bifurcate','hano','hano_dual'])
def test_rollout_training_equals_inference_and_has_gradients(name):
    lit=YeastLitModule(model_name=name,width=8,n_blocks=1,modes=2,loss_type='mse_only',rollout_type='autoregressive')
    lit.bind_datamodule_stats(stats('log1p'))
    lit.log=lambda *args,**kwargs:None
    x=torch.zeros(2,39,2,6);x[:,:,1,:]=torch.linspace(0,1,6)
    p=torch.zeros(2,141);y=torch.zeros(2,39,6)
    captured=[]
    class RecordingLoss(nn.Module):
        def forward(self,pred,target,*args,**kwargs):
            captured.append(pred)
            return (pred-target).square().mean()
    lit.criterion=RecordingLoss()
    count=[0]
    hook=lit.model.register_forward_hook(lambda *args:count.__setitem__(0,count[0]+1))
    lit.train();loss=lit.training_step((x,p,y),0)
    assert count[0]==5
    loss.backward()
    assert any(param.grad is not None for param in lit.parameters())
    assert all(torch.isfinite(param.grad).all() for param in lit.parameters() if param.grad is not None)
    lit.eval();count[0]=0
    with torch.no_grad():pred=lit.forward_ic_time(x,p)
    hook.remove()
    assert count[0]==5
    torch.testing.assert_close(captured[0],pred)
    torch.testing.assert_close(pred[:,:,0],x[:,:,0,0])


@pytest.mark.parametrize('module',[ma,mass])
@pytest.mark.parametrize('fault',['failed','truncated','nonfinite','stuck_event'])
def test_solver_rejects_incomplete_or_invalid_segments(monkeypatch,module,fault):
    def fake(fun,span,y0,**kwargs):
        t=np.array([span[0],span[0]+1.])
        y=np.stack([y0,y0],axis=1)
        if fault=='stuck_event':
            return SimpleNamespace(t=t[:1],y=y[:,:1],success=True,status=1,
                                   t_events=[np.array([span[0]])]+[np.array([])]*3)
        if fault=='nonfinite':y[0,-1]=np.nan
        return SimpleNamespace(t=t,y=y,success=fault!='failed',status=-1 if fault=='failed' else 0)
    monkeypatch.setattr(module,'solve_ivp',fake)
    result=module.simulate_mutant('artificial_fault',{},210,16)
    assert result[1] is None and result[2]=='Inviable: Numerical Failure'


@pytest.mark.parametrize('module',[ma,mass])
def test_solver_preserves_successful_trajectory(monkeypatch,module):
    def fake(fun,span,y0,**kwargs):
        return SimpleNamespace(t=np.array(span),y=np.stack([y0,y0],axis=1),success=True,status=0)
    monkeypatch.setattr(module,'solve_ivp',fake)
    result=module.simulate_mutant('artificial_ok',{},210,16)
    np.testing.assert_array_equal(result[1],np.repeat(module.initial_conds()[:,None],16,axis=1))


def test_worker_timeout_returns_failure_and_restores_alarm(monkeypatch):
    import signal
    monkeypatch.setattr(guards,'DEFAULT_SIM_TIMEOUT',.02)
    def slow(*args,**kwargs):time.sleep(.2)
    monkeypatch.setattr(ma,'solve_ivp',slow)
    old=signal.getsignal(signal.SIGALRM)
    started=time.monotonic()
    result=ma.simulate_mutant('artificial_slow',{},210,16)
    assert result[1] is None and time.monotonic()-started<.15
    assert signal.getsignal(signal.SIGALRM)==old
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0


def _blocked_worker():
    time.sleep(3.)


def test_parent_watchdog_terminates_stalled_pool():
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=1) as executor:
        future=executor.submit(_blocked_worker)
        with pytest.raises(TimeoutError,match='generation aborted'):
            list(guards.bounded_completed(executor,[future],stall_timeout=.05))
    assert time.monotonic()-started<2.


def test_watchdog_ordered_results_do_not_reorder_candidates():
    class Fake:
        def submit(self,fn,task):
            future=Future();future.set_result(fn(task));return future
    assert list(guards.bounded_ordered_map(Fake(),lambda i:i*2,[9,3,1]))==[18,6,2]


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('quota',[False,True])
def test_al4pde_pool_and_ingest_use_identical_transforms(tmp_path,mode,quota,monkeypatch):
    from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
    dm=YeastDataModule(norm_type=mode,use_adj_matrix=False)
    for key,value in vars(stats(mode)).items():setattr(dm,key,value)
    dm._Y=torch.full((1,39,500),100.)
    dm._test_idx_real=[0]
    params=np.full((2,141),3.)
    from budding_yeast_v2.data.systems.budding.api import prepare_initial_conditions,PARAM_KEYS
    ics=np.array([prepare_initial_conditions(dict(zip(PARAM_KEYS,row)))[1] for row in params])
    path=tmp_path/'pool.npz'
    from budding_yeast_v2.data.systems.budding.al_pool import POOL_CONTRACT
    from budding_yeast_v2.data.systems.budding.catalog import CATALOG_VERSION
    from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
    np.savez(path,params=params,y_init=ics,mutant_names=np.array(['MASS_RAND_CLB2[MC]_L1_1','MASS_RAND_CLB2[MC]_L1_2']),
        pool_generation_contract=POOL_CONTRACT,catalog_version=CATALOG_VERSION,
        sampling_range_fingerprint=RANGE_FINGERPRINT)
    cb=ActiveLearningCallback(strategy='al4pde_random',al_pool_dataset=str(path))
    cb._generate_param_only_pool(dm)
    torch.testing.assert_close(cb._pool_P,dm.normalize_p(torch.tensor(params,dtype=torch.float32)))
    torch.testing.assert_close(cb._pool_Y_ic,dm.normalize_y(torch.tensor(ics,dtype=torch.float32)))
    class FakeExecutor:
        def __init__(self,*args,**kwargs):self.index=0
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,task):
            future=Future()
            y=np.full((39,500),10.);y[:,0]=ics[self.index]
            future.set_result((None,y,params[self.index],'Pattern_B'))
            self.index+=1
            return future
    with patch('budding_yeast_v2.utils.sampling.oracle.ProcessPoolExecutor',FakeExecutor):
        result=cb._build_broadcast_from_pool(np.array([0,1]),dm,target_valid_num=1 if quota else None)
    count=1 if quota else 2
    torch.testing.assert_close(result['P'],dm.normalize_p(torch.tensor(params[:count],dtype=torch.float32)))
    expected_y=torch.full((count,39,500),10.);expected_y[:,:,0]=torch.tensor(ics[:count],dtype=torch.float32)
    torch.testing.assert_close(result['Y'],dm.normalize_y(expected_y))
    torch.testing.assert_close(result['X'][:,:,0,:],result['Y'][:,:,0:1].expand(-1,-1,500))
