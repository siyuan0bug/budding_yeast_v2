"""Versioned preprocessing with explicit train-only/legacy fitting policies."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .normalization import check_norm_type

TRAIN_FIT_SCOPE = 'train_synthetic'
LEGACY_FIT_SCOPE = 'all_synthetic_legacy'
FIT_SCOPES = (TRAIN_FIT_SCOPE, LEGACY_FIT_SCOPE)


def check_fit_scope(scope):
    if scope not in FIT_SCOPES:
        raise ValueError(f'Invalid normalization fit_scope {scope!r}; expected one of {FIT_SCOPES}')
    return scope


def _array_hash(values):
    arr = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(str(arr.shape).encode())
    # Hash the actual arrays already loaded by the DataModule, not another NPZ
    # decompression. A view avoids making a second multi-GB trajectory copy.
    if arr.dtype.hasobject:
        digest.update(json.dumps(arr.tolist(), ensure_ascii=False, default=str).encode())
    else:
        digest.update(memoryview(arr).cast('B'))
    return digest.hexdigest()


def dataset_contract(raw, params, names, patterns, dataset_path, t_max):
    schema = {'param_names': None, 'variable_names': None}
    if Path(dataset_path).is_file():
        with np.load(dataset_path, allow_pickle=True) as archive:
            for key in schema:
                if key in archive:
                    schema[key] = [str(x) for x in archive[key].tolist()]
    return {
        'shape': list(raw.shape), 'param_dim': int(params.shape[1]),
        't_max': float(t_max), **schema,
        'data_sha256': _array_hash(raw), 'params_sha256': _array_hash(params),
        'names_sha256': _array_hash(np.asarray(names)),
        'patterns_sha256': _array_hash(np.asarray(patterns)),
    }


def validate_preprocessing_state(state, num_vars=None, param_dim=None):
    if not isinstance(state, dict) or state.get('schema_version') != 1:
        raise ValueError('Invalid checkpoint preprocessing contract/version')
    result = copy.deepcopy(state)
    nt = check_norm_type(result.get('norm_type'))
    check_fit_scope(result.get('fit_scope'))
    for key in ('mean_y', 'std_y', 'p_mean', 'p_std'):
        value = result.get(key)
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f'Invalid preprocessing statistic: {key}')
        result[key] = value.detach().cpu().clone()
    v = result['mean_y'].numel()
    d = result['p_mean'].numel()
    if (num_vars is not None and v != num_vars) or (param_dim is not None and d != param_dim):
        raise ValueError('Checkpoint preprocessing dimensions disagree with model/data')
    if result['mean_y'].shape != (1, v, 1) or result['std_y'].shape != (1, v, 1):
        raise ValueError('Expected preprocessing Y statistics with shape (1,V,1)')
    if result['p_mean'].shape != (1, d) or result['p_std'].shape != (1, d):
        raise ValueError('Expected preprocessing P statistics with shape (1,D)')
    if (result['std_y'] <= 0).any() or (result['p_std'] < 0).any():
        raise ValueError('Invalid preprocessing standard deviations')
    if nt in ('log1p', 'linear') and (result['p_std'] <= 0).any():
        raise ValueError('Affine parameter standard deviations must be positive')
    sc = result.get('s_c')
    if nt == 'asinh_scale':
        if not isinstance(sc, torch.Tensor) or sc.shape != (1, v, 1) or not torch.isfinite(sc).all() or (sc <= 0).any():
            raise ValueError('asinh_scale requires finite positive checkpoint s_c')
        result['s_c'] = sc.detach().cpu().clone()
    if not np.isfinite(result.get('t_max', float('nan'))) or result['t_max'] <= 0:
        raise ValueError('Invalid preprocessing physical horizon')
    return result


def make_preprocessing_state(dm):
    return validate_preprocessing_state({
        'schema_version': 1, 'norm_type': dm.norm_type,
        'mean_y': dm.mean_y, 'std_y': dm.std_y,
        'p_mean': dm.p_mean, 'p_std': dm.p_std, 's_c': dm.s_c,
        't_max': float(dm.t_max),
        'fit_scope': dm.fit_scope,
        'dataset': getattr(dm, '_dataset_contract', None),
        'split': getattr(dm, '_preprocessing_split', None) or {
            'seed': dm.seed,
            'train': list(dm._train_idx_synth), 'val': list(dm._val_idx_synth),
            'test': list(dm._test_idx_synth), 'real': list(dm._test_idx_real),
        },
    })


def assert_compatible(saved, current, *, check_stats=True):
    saved = validate_preprocessing_state(saved)
    current = validate_preprocessing_state(current)
    for key in ('norm_type', 't_max', 'dataset', 'split', 'fit_scope'):
        if saved.get(key) != current.get(key):
            raise ValueError(f'Checkpoint preprocessing mismatch: {key}; refusing silent refit/reinterpretation')
    if check_stats:
        for key in ('mean_y', 'std_y', 'p_mean', 'p_std', 's_c'):
            a, b = saved.get(key), current.get(key)
            if a is None and b is None:
                continue
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not torch.equal(a.cpu(), b.cpu()):
                raise ValueError(f'Checkpoint preprocessing mismatch: {key}')


def resolve_checkpoint_norm(checkpoint, requested=None):
    """New payload > model hparams > legacy DataModule hparams > explicit/default."""
    state = checkpoint.get('preprocessing_contract')
    nt = validate_preprocessing_state(state)['norm_type'] if state is not None else None
    hp = checkpoint.get('hyper_parameters', {})
    dhp = checkpoint.get('datamodule_hyper_parameters', {})
    recorded = nt or hp.get('norm_type') or dhp.get('norm_type')
    if nt and hp.get('norm_type', nt) != nt:
        raise ValueError('Checkpoint model norm_type contradicts preprocessing contract')
    if requested is not None and recorded is not None and requested != recorded:
        raise ValueError(f'Requested normalization {requested!r} contradicts checkpoint {recorded!r}')
    return check_norm_type(recorded or requested or 'log1p')


def resolve_checkpoint_fit_scope(checkpoint):
    """Never reinterpret old normalized weights using the new train-only fit."""
    state = checkpoint.get('preprocessing_contract')
    if state is not None:
        return validate_preprocessing_state(state)['fit_scope']
    # Payload-free historical checkpoints fitted all synthetic rows. Honor an
    # explicit DataModule policy if present, otherwise reconstruct legacy only.
    return check_fit_scope(checkpoint.get('datamodule_hyper_parameters', {}).get(
        'fit_scope', LEGACY_FIT_SCOPE
    ))
