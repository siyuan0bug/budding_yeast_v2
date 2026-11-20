"""Normalized validation selects weights; each new cohort refits train-only stats."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_sampling_benchmark import config
from test_shared_training_cleanup import selected_fixture
from budding_yeast_v2.data.dataset_archive import TrajectoryStore
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig
from budding_yeast_v2.utils.sampling.benchmark.rounds import fit_round
from budding_yeast_v2.utils.sampling.contracts import PoolInputs, QueriedTraining


def test_normalized_validation_uses_training_relative_objective_and_keeps_physical_diagnostic():
    assert BenchmarkConfig().validate().validation_metric == 'normalized_loss'
    model = YeastLitModule(model_name='fno', width=4, modes=2, n_blocks=1, loss_type='rel_l2')
    model._trainer = SimpleNamespace(datamodule=SimpleNamespace(validation_metric='normalized_loss'), logger=None)
    logs = {}
    model.log = lambda key, value, **kwargs: logs.update({key: float(value)})
    def forward(x, p):
        assert not torch.is_autocast_enabled(x.device.type)
        return x[:, :, 0, :]
    model.forward_ic_time = forward
    model.denormalize_y = lambda y: y*100
    prediction = torch.full((3, 39, 11), .4)
    normalized_truth = torch.full_like(prediction, .3)
    physical_truth = torch.full_like(prediction, 200.)
    model.on_validation_epoch_start()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        model.validation_step((prediction[:, :, None, :], torch.zeros(3, 141),
                               normalized_truth, physical_truth), 0)
    model.on_validation_epoch_end()
    assert logs['val_loss_synth'] == pytest.approx(float(model.criterion(prediction, normalized_truth)))
    assert logs['val_sample_channel_rel_l2'] != pytest.approx(logs['val_loss_synth'])


def test_new_round_refits_stats_reencodes_fixed_validation_and_selects_normalized_best(tmp_path, monkeypatch):
    source, values, snapshot, training, validation = selected_fixture(tmp_path)
    cfg = replace(config(), epochs=(3,)*4, batch_size=2)
    encoded_validation = []
    # Deliberately opposite minima expose accidental use of the physical monitor.
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            stats = NormalizationStats.from_datamodule(self.trainer.datamodule)
            torch.testing.assert_close(batch[2], stats.targets(batch[3]), rtol=0, atol=0)
            if self.current_epoch == 0:
                encoded_validation.append(batch[2].detach().cpu().clone())
        self.log('val_loss_synth', self.device_tensor([3., 1., 2.][self.current_epoch]),
                 on_step=False, on_epoch=True, batch_size=len(batch[0]))
        self.log('val_sample_channel_rel_l2', self.device_tensor([1., 2., 3.][self.current_epoch]),
                 on_step=False, on_epoch=True, batch_size=len(batch[0]))
    monkeypatch.setattr(YeastLitModule, 'device_tensor', lambda self, value: torch.tensor(value, device=self.device), raising=False)
    monkeypatch.setattr(YeastLitModule, 'validation_step', validation_step)
    ids = np.arange(8)
    expanded_snapshot = tmp_path/'expanded.json'
    TrajectoryStore(source).snapshot(expanded_snapshot, ids)
    expanded = QueriedTraining(PoolInputs(ids, values['params'][ids], values['y_init'][ids]), values['data'][ids])
    stats_by_round = []
    for i, cohort, path in [(0, training, snapshot), (1, expanded, expanded_snapshot)]:
        output = tmp_path/f'round_{i}'
        model, stats, history = fit_round(cfg, i, cohort, validation, output, 77+i, dataset_path=path)
        saved = torch.load(output/'model.ckpt', map_location='cpu', weights_only=False)
        assert saved['epoch']+1 == 2
        assert [r['val_error'] for r in history] == [3., 1., 2.]
        assert [r['val_sample_channel_rel_l2'] for r in history] == [1., 2., 3.]
        assert all(r['validation_metric'] == 'val_loss_synth' for r in history)
        assert json.loads((output/'_state/training.json').read_text())['validation_metric'] == 'val_loss_synth'
        expected = NormalizationStats.fit(cohort.inputs.params, cohort.trajectories, 'log1p')
        restored = NormalizationStats.from_datamodule(model)
        for name in ('mean_y', 'std_y', 'p_mean', 'p_std'):
            torch.testing.assert_close(getattr(stats, name), getattr(expected, name), rtol=0, atol=0)
            torch.testing.assert_close(getattr(restored, name), getattr(expected, name), rtol=0, atol=0)
        stats_by_round.append(stats)
    assert not torch.equal(stats_by_round[0].mean_y, stats_by_round[1].mean_y)
    assert not torch.equal(stats_by_round[0].p_mean, stats_by_round[1].p_mean)
    assert not torch.equal(encoded_validation[0], encoded_validation[1])
