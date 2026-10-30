"""Check a real checkpoint against its exact reconstructed AL round dataset."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from budding_yeast_v2.data.dataset_archive import _atomic_json, resolve_dataset_path
from budding_yeast_v2.data.preprocessing_contract import assert_compatible
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.eval import load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, help='Default: adjacent dataset; old .npz aliases also accepted')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    path = args.dataset or args.checkpoint.parent/'dataset.npz'
    model, hp = load_model_from_checkpoint(str(args.checkpoint), device='cpu')
    state = model._preprocessing_state
    dm = YeastDataModule(str(path), num_workers=0, use_adj_matrix=False,
        norm_type=state['norm_type'], fit_scope=state['fit_scope'], preprocessing_state=state)
    dm.setup('test')
    assert_compatible(state, dm.preprocessing_state())
    x, p = dm.test_dataset_real.tensors[:2]
    model.eval()
    with torch.no_grad():
        prediction = model.forward_ic_time(x[:2], p[:2])
    if not torch.isfinite(prediction).all():
        raise ValueError('Nonfinite checkpoint prediction')
    report = dict(checkpoint=str(args.checkpoint), dataset=str(resolve_dataset_path(path)),
        preprocessing='exact_dataset_hashes_splits_and_statistics_match',
        train_samples=len(dm.train_dataset_synth), val_samples=len(dm.val_dataset_synth),
        synthetic_test_samples=len(dm.test_dataset_synth), real_samples=len(dm.test_dataset_real),
        input_dtype=str(x.dtype), parameter_dtype=str(p.dtype), prediction_shape=list(prediction.shape),
        dataset_contract=state['dataset'])
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
