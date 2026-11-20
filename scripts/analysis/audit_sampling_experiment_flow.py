#!/usr/bin/env python3
"""Audit frozen inputs, splits, checkpoint choices and input scaling without training.

Uses input caches and existing checkpoint statistics, never full-pool trajectories.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from budding_yeast_v2.data.dataset_archive import atomic_json, file_sha256
from budding_yeast_v2.data.preprocessing_contract import validate_preprocessing_state
from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
from budding_yeast_v2.utils.sampling.benchmark.protocol import effective_signatures


def scale_report(params, mean, std):
    maximum = np.zeros(params.shape[1])
    total = large = very_large = 0
    for start in range(0, len(params), 4096):
        z = np.abs((params[start:start+4096].astype(np.float32)-mean)/std)
        maximum = np.maximum(maximum, z.max(0))
        total += z.size
        large += int((z > 100).sum())
        very_large += int((z > 1e4).sum())
    order = np.argsort(-maximum)[:5]
    return dict(rows=len(params), maximum_abs_z=float(maximum.max()),
        entries_over_100=large, entries_over_10000=very_large, entries=total,
        top_parameters=[dict(name=PARAM_KEYS[i], maximum_abs_z=float(maximum[i])) for i in order])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--source-results', type=Path, required=True)
    parser.add_argument('--real-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.protocol/'protocol.json').read_text())
    hashes = {name: file_sha256(args.protocol/name) == digest
              for name, digest in protocol['artifacts'].items()}
    if not all(hashes.values()):
        raise ValueError('Frozen protocol artifact changed')
    with np.load(args.real_reference, allow_pickle=False) as a:
        real_params, real_initial = a['params'], a['y_init']
    real_signatures = set(effective_signatures(real_params, real_initial))
    pools, signatures, source_steps = {}, {}, []
    for name in sorted(protocol['pools']):
        print(f'Auditing cached {name} inputs and split', flush=True)
        with np.load(args.protocol/f'{name}_inputs.npz', allow_pickle=False) as a:
            params, initial, is_real = a['params'], a['y_init'], a['is_real']
        with np.load(args.protocol/f'{name}_split.npz', allow_pickle=False) as a:
            split = {k: a[k] for k in a.files}
        roles = [split[k] for k in ('candidates', 'validation', 'test')]
        joined = np.concatenate(roles)
        assert len(joined) == len(params) and np.array_equal(np.sort(joined), np.arange(len(params)))
        assert set(split['initial']) <= set(split['candidates'])
        assert len(set(split['initial'])) == protocol['config']['budgets'][0]
        assert not is_real.any() and np.isfinite(params).all() and np.isfinite(initial).all()
        sig = effective_signatures(params, initial)
        signatures[name] = set(sig)
        group_roles = {}
        for role, ids in zip(('candidates', 'validation', 'test'), roles):
            for code in sig[ids]:
                previous = group_roles.setdefault(code, role)
                assert previous == role, 'Effective duplicate crosses holdout boundary'
        cp = args.source_results/name/'random/seed_42/rounds/round_00/model.ckpt'
        saved = torch.load(cp, map_location='cpu', weights_only=False)
        state = validate_preprocessing_state(saved['preprocessing_contract'], 39, len(PARAM_KEYS))
        mean, std = state['p_mean'].numpy(), state['p_std'].numpy()
        initial_params = params[split['initial']].astype(np.float32)
        expected_mean = torch.tensor(initial_params).mean(0, keepdim=True)
        expected_std = torch.tensor(initial_params).std(0, keepdim=True)+1e-6
        assert torch.equal(state['p_mean'], expected_mean) and torch.equal(state['p_std'], expected_std)
        vary = np.ptp(params, axis=0) > 0
        absent = (std.reshape(-1) < 1e-5) & vary
        pools[name] = dict(rows=len(params), split={k:len(split[k]) for k in ('candidates','initial','validation','test')},
            effective_duplicates=len(sig)-len(signatures[name]), real_exact_collisions=len(signatures[name] & real_signatures),
            initial_parameter_stats_exact=True, variable_in_pool_but_initial_std_small=[PARAM_KEYS[i] for i in np.flatnonzero(absent)],
            input_z={role:scale_report(params[ids], mean, std) for role, ids in zip(('candidates','validation','test'),roles)},
            real_input_z=scale_report(real_params, mean, std),
            y_std_min=float(state['std_y'].min()))
    names = sorted(signatures)
    overlap = len(signatures[names[0]] & signatures[names[1]]) if len(names)==2 else None
    for path in sorted(args.source_results.glob('*/*/seed_42/rounds/round_*/history.csv')):
        rows = list(csv.DictReader(path.open()))
        a = min(rows, key=lambda r:float(r['val_error']))
        b = min(rows, key=lambda r:float(r['val_loss_synth']))
        source_steps.append(dict(path=str(path), rows=len(rows), prior_best_epoch=int(a['epoch']),
            normalized_best_epoch=int(b['epoch']), min_normalized_validation=float(b['val_loss_synth'])))
    result = dict(protocol=protocol['protocol_id'], config=protocol['config'], cache_hashes=hashes,
        pools=pools, cross_pool_effective_overlap=overlap, existing_histories=source_steps,
        trajectory_blocks_read=False, formal_training_started=False)
    atomic_json(args.output/'input_and_split_audit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='existing_histories'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
