"""Large relative errors must remain visible and differentiable by default."""
import math
import pytest
import torch

from budding_yeast_v2.utils.training.losses import RelativeL2Loss, HybridMSERelLoss, GradNormHybridLoss
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.utils.training.runner import make_trainer


def test_zero_target_large_loss_retains_gradient_and_forward_per_sample_agree():
    y=torch.zeros(2,39,500)
    p=torch.ones_like(y,requires_grad=True)
    criterion=RelativeL2Loss()
    value=criterion(p,y)
    assert value.item()==pytest.approx(math.sqrt(500)/.01)
    assert value>10 and criterion.max_clamp is None
    torch.testing.assert_close(value,criterion.per_sample(p,y).mean())
    value.backward()
    assert torch.isfinite(p.grad).all() and (p.grad>0).all()
    exact=torch.zeros_like(y,requires_grad=True)
    perfect=criterion(exact,y);perfect.backward()
    assert perfect==0 and torch.isfinite(exact.grad).all()


def test_registered_model_and_hybrid_defaults_do_not_reintroduce_cap():
    model=YeastLitModule(model_name='fno',width=4,modes=2,n_blocks=1,loss_type='rel_l2')
    assert model.criterion.max_clamp is None
    checkpoint={};model.on_save_checkpoint(checkpoint)
    assert checkpoint['training_objective_definition']['max_clamp'] is None
    assert checkpoint['training_objective_definition']['eps']==.01
    assert checkpoint['training_objective_definition']['denominator_rule']=='sqrt_target_time_l2_squared_plus_eps_squared'
    assert HybridMSERelLoss().rel_loss.max_clamp is None
    assert HybridMSERelLoss().rel_loss.eps==.01
    assert GradNormHybridLoss().max_clamp is None
    assert GradNormHybridLoss().eps==.01
    trainer=make_trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False)
    assert trainer.gradient_clip_val==1.


@pytest.mark.parametrize('eps',[0.,-1.,float('nan'),float('inf')])
def test_denominator_protection_cannot_be_disabled_by_invalid_eps(eps):
    with pytest.raises(ValueError,match='finite positive'):
        RelativeL2Loss(eps=eps)


@pytest.mark.parametrize('eps',[.01,.03,.1,.3])
def test_smooth_denominator_and_prediction_gradients_match_exact_formula(eps):
    norms=torch.tensor([0.,eps/2,eps,eps*2,10.],dtype=torch.float64)
    y=norms[:,None,None]
    p=(y+.2).clone().requires_grad_()
    criterion=RelativeL2Loss(eps=eps)
    denominator=(norms.square()+eps**2).sqrt()
    expected=.2/denominator
    torch.testing.assert_close(criterion.per_sample(p,y),expected)
    from budding_yeast_v2.utils.evaluation.metrics import sample_channel_relative_errors
    torch.testing.assert_close(sample_channel_relative_errors(p,y,eps).mean(),criterion(p,y))
    criterion(p,y).backward()
    torch.testing.assert_close(p.grad[:,0,0],1/(denominator*len(norms)))
    assert torch.isfinite(p.grad).all() and (p.grad>0).all()


def test_checkpoint_retains_nondefault_epsilon_without_restoring_a_loss_cap():
    model=YeastLitModule(model_name='fno',width=4,modes=2,n_blocks=1,loss_type='rel_l2')
    model.criterion.eps=.03
    checkpoint={};model.on_save_checkpoint(checkpoint)
    restored=YeastLitModule(model_name='fno',width=4,modes=2,n_blocks=1,loss_type='rel_l2')
    with pytest.warns(UserWarning,match='Legacy checkpoint'):
        restored.on_load_checkpoint(checkpoint)
    assert restored.criterion.eps==.03 and restored.criterion.max_clamp is None
    assert restored.criterion.denominator_rule=='sqrt_target_time_l2_squared_plus_eps_squared'


def test_explicit_historical_floor_checkpoint_keeps_its_original_objective():
    model=YeastLitModule(model_name='fno',width=4,modes=2,n_blocks=1,loss_type='rel_l2')
    checkpoint=dict(training_objective_definition=dict(name='relative_l2_sample_channel',eps=.1,
        denominator_rule='max_target_time_l2_eps'))
    with pytest.warns(UserWarning,match='Legacy checkpoint'):
        model.on_load_checkpoint(checkpoint)
    y=torch.tensor([[[.1]]]);p=y+.2
    assert model.criterion.eps==.1
    torch.testing.assert_close(model.criterion(p,y),torch.tensor(2.))


def test_epsilon_choice_cannot_use_real_or_test_metrics():
    from budding_yeast_v2.scripts.analysis.probe_relative_l2_epsilon import choose_epsilon
    arms=[dict(eps=.03,last_validation=dict(sample_channel_rel_l2=2.),real=0.),
          dict(eps=.1,last_validation=dict(sample_channel_rel_l2=1.),real=100.),
          dict(eps=.3,last_validation=dict(sample_channel_rel_l2=3.),real=0.)]
    assert choose_epsilon(arms)==.1


def test_epsilon_probe_gradient_norm_includes_fourier_imaginary_part(tmp_path):
    from types import SimpleNamespace
    from budding_yeast_v2.scripts.analysis.probe_relative_l2_epsilon import GradientDiagnostics
    p=torch.nn.Parameter(torch.zeros(1,dtype=torch.complex64))
    p.grad=torch.tensor([3+4j],dtype=torch.complex64)
    model=SimpleNamespace(parameters=lambda:[p])
    diagnostic=GradientDiagnostics(tmp_path)
    diagnostic.on_train_epoch_start(None,model)
    diagnostic.on_after_backward(None,model)
    diagnostic.on_train_epoch_end(SimpleNamespace(current_epoch=0),model)
    assert diagnostic.rows[0]['gradient_norm_max']==pytest.approx(5.)
