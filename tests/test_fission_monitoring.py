"""Native Fission labels through supervision, monitoring and experiment records."""
from types import SimpleNamespace

import numpy as np
import optuna
import pytest
import torch
import torch.nn.functional as F

from budding_yeast_v2.data.label_metadata import dataset_label_summary
from budding_yeast_v2.data.systems.fission.constants import VAR_NAMES
from budding_yeast_v2.data.systems.fission.parameters import WT_parameters, param_vector
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.utils.training.losses import HybridBifurcateLoss, HybridAllLoss, _masked_pattern_cross_entropy


@pytest.fixture
def fission_dm(tmp_path):
    # Repeat every category enough times for the existing train/val/test split.
    labels = np.array(['Viable', 'Viable', 'Inviable: G2 Arrest', 'Inviable: M-phase Arrest',
                       'Inviable: Too Small', 'Inviable: cut (premature division)'] * 11)
    patterns = np.array(['Pattern_A', 'Pattern_B', 'None', 'None', 'None', 'None'] * 11)
    data = np.ones((66, 16, 64), dtype=np.float32) * .2
    data[:, 0] = 1.
    data[:, 1] = 1.
    path = tmp_path/'native_fission.npz'
    np.savez_compressed(path, data=data, params=np.tile(param_vector(WT_parameters()), (66, 1)),
        mutant_names=np.array([f'FISSION_{i}' for i in range(66)]), labels=labels, pattern_labels=patterns,
        is_real=np.arange(66) < 6, system_id=np.array('fission'), variable_names=np.array(VAR_NAMES),
        time_grid=np.linspace(0, 300, 64))
    dm = YeastDataModule(path, num_workers=0, use_adj_matrix=False)
    dm.setup('fit')
    return dm


def test_absent_fission_patterns_are_unlabelled_targets_and_raw_labels_unchanged(fission_dm):
    dm = fission_dm
    actual = dm.train_dataset_synth.tensors[3].numpy()
    expected = [dict(Pattern_A=0, Pattern_B=1, None_=-1)[
        'None_' if dm.pattern_labels[idx] == 'None' else str(dm.pattern_labels[idx])]
        for idx in dm._train_idx_synth]
    np.testing.assert_array_equal(actual, expected)
    assert (actual == -1).any() and (dm.pattern_labels == 'None').any()
    assert dm.pattern_group_codes is None


def test_fission_monitors_cover_actual_biology_and_patterns_without_budding_fallback(fission_dm):
    dm = fission_dm
    for split, attr in [('synth_train', '_train_idx_synth'), ('synth_test', '_test_idx_synth'), ('real_mutants', '_test_idx_real')]:
        indices = getattr(dm, attr)
        expected = {(str(dm.biological_labels[i]), str(dm.pattern_labels[i])) for i in indices}
        samples = [(key, value) for key, value in dm.fixed_samples_for_plot.items() if key.startswith(split+'/')]
        observed = [(dm.fixed_sample_label_metadata[key]['biological_label'], str(value[4])) for key, value in samples]
        assert set(observed) == expected and len(observed) == len(expected)
        assert all(dm.fixed_sample_label_metadata[key]['pattern_group'] is None for key, _ in samples)
    assert not any(key.startswith(('real_typical/', 'real_badcases/')) or 'Pattern_C' in key
                   for key in dm.fixed_samples_for_plot)
    summary = dataset_label_summary(dm)
    assert summary['system_id'] == 'fission' and summary['horizon_minutes'] == 300
    assert summary['variable_names'] == VAR_NAMES
    assert summary['missing_pattern_target'] == -1
    assert summary['splits']['real']['pattern_labels']['None'] == 4
    assert all('pattern_groups' not in split for split in summary['splits'].values())


@pytest.mark.parametrize('targets', [[-1], [-1, -1], [0, -1, 1], [0, 1, 2]])
def test_masked_classification_ignores_absent_patterns_and_preserves_existing_abc(targets):
    logits = torch.randn(len(targets), 3, requires_grad=True)
    target = torch.tensor(targets)
    loss = _masked_pattern_cross_entropy(logits, target[:, None])
    valid = target >= 0
    expected = F.cross_entropy(logits[valid], target[valid]) if valid.any() else logits.sum()*0
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[~valid] == 0).all()


@pytest.mark.parametrize('criterion', ['hybrid_bifurcate', 'hybrid_all'])
@pytest.mark.parametrize('target', [-1, 0, 1, 2])
def test_pattern_loss_accepts_batch_one_and_column_targets(criterion, target):
    pred = torch.ones(1, 16, 12, requires_grad=True)
    truth = torch.ones_like(pred)
    logits = torch.randn(1, 3, requires_grad=True)
    if criterion == 'hybrid_bifurcate':
        loss = HybridBifurcateLoss()(pred, truth, pattern_logits=logits, pattern_targets=torch.tensor([[target]]))
    else:
        obj = HybridAllLoss(torch.zeros(1, 16, 1), torch.ones(1, 16, 1),
                            torch.zeros(1, 58), torch.ones(1, 58),
                            lambda_phys=0., lambda_jump=0., lambda_penalty=0.)
        obj.system_id = 'fission'
        loss = obj(pred, truth, torch.zeros(1, 58), pattern_logits=logits, pattern_targets=torch.tensor([[target]]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and torch.isfinite(logits.grad).all()
    if target == -1:
        assert loss.item() == 0 and logits.grad.abs().sum().item() == 0


@pytest.mark.parametrize('training', [True, False])
def test_visualizer_logs_native_biology_16_variables_and_300_minutes(fission_dm, monkeypatch, training):
    from budding_yeast_v2 import train
    captured = []
    class Logger:
        experiment = SimpleNamespace(log=lambda payload: captured.append(payload))
    class Predictor(torch.nn.Module):
        device = torch.device('cpu')
        def forward_ic_time(self, x, p): return x[:, :, 0]
    dm = fission_dm
    key = next(k for k, v in dm.fixed_samples_for_plot.items() if str(v[4]) == 'None')
    dm.fixed_samples_for_plot = {key: dm.fixed_samples_for_plot[key]}
    monkeypatch.setattr(train, 'WandbLogger', Logger)
    def image(fig):
        return dict(title=fig._suptitle.get_text(), variables=[ax.get_title() for ax in fig.axes],
                    end_time=float(fig.axes[0].lines[0].get_xdata()[-1]))
    monkeypatch.setattr(train.wandb, 'Image', image)
    trainer = SimpleNamespace(current_epoch=0, sanity_checking=False, is_global_zero=True,
                              datamodule=dm, logger=Logger())
    model = Predictor().train(training)
    train.SimToRealVisualizerCallback(1).on_validation_epoch_end(trainer, model)
    assert model.training == training
    payload = captured[0][key]
    assert 'fission | BIO: Inviable:' in payload['title'] and 'Pattern: None' in payload['title']
    assert 'Group:' not in payload['title']
    assert payload['variables'] == [f'{i}: {name}' for i, name in enumerate(VAR_NAMES)]
    assert payload['end_time'] == 300


def test_logger_tags_identify_resolved_system_without_network(tmp_path, monkeypatch):
    import pytorch_lightning.loggers
    from budding_yeast_v2.utils.experiments.paths import setup_wandb_logger
    class Logger:
        def __init__(self, **kwargs): self.kwargs = kwargs
        def log_hyperparams(self, values): self.values = values
    monkeypatch.setattr(pytorch_lightning.loggers, 'WandbLogger', Logger)
    args = SimpleNamespace(system='auto', system_id='fission', dataset_name='fission.npz')
    logger = setup_wandb_logger(args, str(tmp_path), project='fission_yeast_v2')
    assert 'system:fission' in logger.kwargs['tags']
    assert logger.values['system_id'] == 'fission'


def test_hpo_study_rejects_system_or_dataset_mixing(fission_dm):
    from budding_yeast_v2.tune import bind_study_dataset
    study = optuna.create_study()
    bind_study_dataset(study, fission_dm)
    bind_study_dataset(study, fission_dm)
    wrong = SimpleNamespace(system_id='budding', _dataset_contract=fission_dm._dataset_contract)
    with pytest.raises(ValueError, match='mismatch'):
        bind_study_dataset(study, wrong)
    wrong = SimpleNamespace(system_id='fission', _dataset_contract={})
    with pytest.raises(ValueError, match='mismatch'):
        bind_study_dataset(study, wrong)


def test_hpo_does_not_assign_a_system_to_unknown_completed_trials(fission_dm):
    from budding_yeast_v2.tune import bind_study_dataset
    study = optuna.create_study()
    study.add_trial(optuna.trial.create_trial(value=.1))
    with pytest.raises(ValueError, match='without a biological dataset contract'):
        bind_study_dataset(study, fission_dm)
