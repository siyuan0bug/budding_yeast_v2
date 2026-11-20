"""Extra real diagnostics must not change training or checkpoint selection."""
from dataclasses import replace
import json
import torch

from test_shared_training_cleanup import selected_fixture
from test_sampling_benchmark import config
from budding_yeast_v2.scripts.analysis.replay_sampling_final_round import RealEpochTrace
from budding_yeast_v2.utils.sampling.benchmark.rounds import fit_round


def test_real_trace_preserves_training_weights_history_and_synthetic_best(tmp_path):
    _, values, snapshot, training, validation = selected_fixture(tmp_path)
    cfg = replace(config(), epochs=(3, 3, 3, 3), batch_size=2, validation_metric='sample_channel_rel_l2')
    plain, stats, history = fit_round(cfg, 0, training, validation, tmp_path/'plain', 77,
                                     dataset_path=snapshot)
    output = tmp_path/'observed'
    output.mkdir()
    cohort = {k: values[k][20:23] for k in ('params', 'y_init', 'data')}
    observer = RealEpochTrace(output, cohort, 2, [1, 3], stats, 'no_original_snapshot')
    observed, _, after = fit_round(cfg, 0, training, validation, output, 77,
                                  dataset_path=snapshot, extra_callbacks=[observer])
    assert history == after
    for name, value in plain.state_dict().items():
        torch.testing.assert_close(value, observed.state_dict()[name], rtol=0, atol=0)
    assert [r['epoch'] for r in observer.rows] == [1, 2, 3]
    assert all(r['metric_space'] == 'physical' for r in observer.rows)
    assert (output/'epoch_001.ckpt').is_file() and (output/'epoch_003.ckpt').is_file()
    assert not (output/'epoch_002.ckpt').exists()
    assert all(json.loads((output/'normalization_audit.json').read_text()).values())
    assert json.loads((output/'_state/training.json').read_text())['validation_metric'] == 'val_sample_channel_rel_l2'
