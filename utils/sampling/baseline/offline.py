"""Fixed-pool selectors with explicit input-only or labelled-oracle contracts."""
import numpy as np
import torch

from ..contracts import SamplingResult
from ..backends import _bmdal_distance_select, _bmdal_bait_select
from .authors import vessal_penultimate, leverage
from .uncertainty import _power_sample_indices


def select(context, budget, name):
    if name.startswith('clustering_sensitivity'):
        from .clustering_sensitivity import select as select_sensitivity
        return select_sensitivity(context, budget, name)
    if name.startswith('physics_pre_'):
        from .physics_pre import select as select_physics
        return select_physics(context, budget, name)
    rng = np.random.default_rng(context.seed)
    pool = context.pool
    diagnostics = dict(implementation='independent_project_implementation')
    if name == 'random':
        indices = rng.choice(len(pool.ids), budget, replace=False)
    elif name in ('topk', 'power'):
        scores = context.predictor.uncertainty(pool)
        if name == 'topk':
            indices = torch.argsort(torch.as_tensor(scores), descending=True)[:budget].numpy()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(context.seed)
                indices = _power_sample_indices(scores, budget, beta=1.)
        diagnostics = dict(ensemble_members=len(context.predictor.members),
                           uncertainty='mean MSE disagreement', power_beta=1.)
    elif name.startswith(('max_dist_', 'coreset_', 'lcmd_', 'bait_')):
        latent = name.endswith('_ll')
        candidate = context.predictor.features(pool, context.sketch_dim, latent)
        anchors = context.predictor.features(context.train.inputs, context.sketch_dim, latent)
        if name.startswith('bait_'):
            indices = _bmdal_bait_select(torch.from_numpy(candidate), torch.from_numpy(anchors), budget, context.device)
        else:
            method = 'lcmd' if name.startswith('lcmd_') else 'max_dist'
            indices = _bmdal_distance_select(torch.from_numpy(candidate), torch.from_numpy(anchors),
                                            budget, context.device, method=method)
        diagnostics = dict(representation='last_hidden_layer' if latent else 'predicted_trajectory',
                           sketch_dimension=context.sketch_dim, posterior=name.startswith('bait_'))
    elif name == 'vessal_penultimate':
        features = context.predictor.features(pool, context.sketch_dim, latent=True)
        # The author method is order-dependent; pool storage order is not a stream.
        order = rng.permutation(len(pool.ids))
        local, diagnostics = vessal_penultimate(features[order], budget, context.seed, context.device)
        indices = order[local]
    elif name == 'lev_input':
        full = np.concatenate((pool.params, pool.initial), axis=1)
        scale = full.std(0)
        scale[scale < 1e-8] = 1.
        features = (full - full.mean(0)) / scale
        order = rng.permutation(len(pool.ids))
        local, diagnostics = leverage(features[order], budget, context.seed, context.device)
        indices = order[local]
        diagnostics['representation'] = 'candidate-input Z-score (no output labels)'
    elif name == 'residual_topk':
        if context.label_reader is None:
            raise PermissionError('True-error selection belongs to labelled track B')
        scores = np.empty(len(pool.ids), dtype=np.float32)
        # Only B has a callable true-label reader; read and score bounded batches.
        for start, prediction in context.predictor.batches(pool):
            ids = pool.ids[start:start + len(prediction)]
            truth = context.label_reader(ids)['data']
            target = context.predictor.normalization.targets(truth)
            scores[start:start + len(prediction)] = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1).numpy()
        indices = np.argsort(-scores, kind='stable')[:budget]
        diagnostics['score'] = 'HGGS true prediction residual, normalized trajectory regression adaptation'
    elif name == 'true_pattern_random':
        if context.label_reader is None:
            raise PermissionError('True-Pattern stratification belongs to labelled track B')
        labels = np.concatenate([context.label_reader(pool.ids[start:start + 256])['pattern_group_ids']
                                 for start in range(0, len(pool.ids), 256)])
        counts = np.bincount(labels.astype(int), minlength=6)
        selected = []
        # Water-fill the available classes; never overrun a finite class capacity.
        buckets = [list(rng.permutation(np.flatnonzero(labels == k))) for k in range(6)]
        while len(selected) < budget:
            for bucket in buckets:
                if bucket and len(selected) < budget:
                    selected.append(bucket.pop())
        indices = np.asarray(selected)
        diagnostics['true_pool_pattern_counts'] = counts.tolist()
    else:
        raise ValueError(f'Unsupported implemented offline baseline: {name}')
    return SamplingResult(pool.ids[np.asarray(indices, dtype=int)], diagnostics=diagnostics)
