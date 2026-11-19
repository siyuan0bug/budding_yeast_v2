"""Runnable pattern-free acquisitions for complete pools and fresh legal banks."""
from collections import defaultdict
import os

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ...contracts import PoolInputs, SamplingResult
from . import METHOD_NAMES
from .systems import BiologicalODE
from .transport import TransportConfig
from .parallel import score_rows
from .actions import pairs, rectangles, choose_faces
from budding_yeast_v2.utils.experiments.logging import ProgressLog


def _physical(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _scale(truth):
    values = np.asarray(truth)
    total = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, len(values), 64):
        block = np.asarray(values[start:start + 64], dtype=np.float64)
        total += np.einsum('bvt,bvt->v', block, block)
    rms = np.sqrt(total / (len(values) * values.shape[-1]))
    positive = rms[rms > 0]
    floor = max(float(np.median(positive)) * 1e-3 if len(positive) else 1e-3, 1e-12)
    return np.maximum(rms, floor)


def _metadata(inputs):
    return inputs.metadata if inputs.metadata else tuple({} for _ in inputs.ids)


def _context_key(meta, p, system):
    environment = (float(p[system.param_names.index('mdt')]),) if system.name == 'budding' else ()
    flags = tuple(bool(meta.get('_effective_overrides', meta.get('original_overrides', {})).get(k, False))
                  for k in ('is_sbf_ca', 'is_mutant_104'))
    return pairs(meta), environment, flags


def _nearest(pool, train, system):
    """Training-only parameter geometry; prefer matching intervention contexts."""
    scale = np.asarray(train.params).std(0)
    scale[scale < 1e-8] = 1.
    tree = NearestNeighbors(n_neighbors=1).fit(train.params / scale)
    nearest = tree.kneighbors(pool.params / scale, return_distance=False)[:, 0]
    groups = defaultdict(list)
    for j, (meta, p) in enumerate(zip(_metadata(train), train.params)):
        groups[_context_key(meta, p, system)].append(j)
    query_groups = defaultdict(list)
    for j, (meta, p) in enumerate(zip(_metadata(pool), pool.params)):
        query_groups[_context_key(meta, p, system)].append(j)
    for key, rows in query_groups.items():
        anchors = groups.get(key)
        if anchors:
            local = NearestNeighbors(n_neighbors=1).fit(train.params[anchors] / scale)
            ids = local.kneighbors(pool.params[rows] / scale, return_distance=False)[:, 0]
            nearest[rows] = np.asarray(anchors)[ids]
    return nearest


def prepare_training(train, system):
    """Reuse train-only reductions while one generation round scans many banks."""
    certificates = {}
    metas = _metadata(train.inputs)
    for j, (p, y0, meta) in enumerate(zip(train.inputs.params, train.inputs.initial, metas)):
        key = system.equivalence_key(p, y0, meta)
        if key is not None:
            certificates.setdefault(key, j)
    return dict(training=train, system=system.name, rhs_sha256=system.source_sha,
                scale=_scale(train.trajectories), certificates=certificates, metadata=metas)


def mechanism_scores(context, name, *, transport_config=TransportConfig(), training_preparation=None):
    if name not in METHOD_NAMES:
        raise ValueError('Unknown portable mechanism acquisition')
    system = BiologicalODE(getattr(context, 'system_id', 'budding'))
    pool, train = context.pool, context.train
    if not len(train.inputs.ids):
        raise ValueError('Mechanism acquisition requires queried synthetic training anchors')
    if train.trajectories.shape[1] != system.n:
        raise ValueError('Training trajectories do not match declared biological system')
    prepared = prepare_training(train, system) if training_preparation is None else training_preparation
    if (prepared['training'] is not train or prepared['system'] != system.name
            or prepared['rhs_sha256'] != system.source_sha):
        raise ValueError('Mechanism training preparation belongs to a different queried cohort/system')
    scale = prepared['scale']
    metas = _metadata(pool)
    anchors = _nearest(pool, train.inputs, system) if name=='ours_mechanism_witness' else None
    certificates = prepared['certificates']
    count, width = len(pool.ids), system.n * 8
    scores = np.empty(count)
    features = np.empty((count, width), dtype=np.float64)
    report = dict(version='pattern_free_mechanism_v1', method=name, system=system.name,
        rhs_sha256=system.source_sha, pattern_labels_used=False, candidate_truth_used=False,
        certified_equivalence_rows=0, linearized_response_rows=0, transported_defect_rows=0,
        grazing_updates=0, unresolved_event_intervals=0,
        train_channel_scale=scale.tolist(), transport_config=vars(transport_config),
        exact_certificate_enabled=system.certificate_supported,
        score_space='physical trajectory / train-only channel RMS; not the training loss')
    workers = int(os.environ.get('SAMPLING_MECHANISM_WORKERS', '4'))
    if not 1 <= workers <= min(16, len(os.sched_getaffinity(0))):
        raise ValueError('Mechanism workers must fit the explicit 1..16 CPU bound')
    report['execution'] = dict(backend='ordered_spawn_cpu_v1' if workers > 1 else 'serial_cpu',
                               workers=workers, torch_threads_per_worker=1,
                               max_inflight_rows=context.prediction_batch_size)
    times = getattr(context, 'time_grid', None)
    seen = np.zeros(count, dtype=bool)
    progress = ProgressLog('mechanism candidate scoring',count,method=name)
    for start, predictions in context.predictor.batches(pool, physical=True):
        predictions = _physical(predictions)
        if (predictions.ndim != 3 or predictions.shape[1] != system.n
                or not 0 <= start < count or start + len(predictions) > count
                or seen[start:start + len(predictions)].any() or not np.isfinite(predictions).all()):
            raise ValueError('Unaligned, duplicate or nonfinite predictor batch in mechanism acquisition')
        seen[start:start + len(predictions)] = True
        requests = []
        train_metas = prepared['metadata']
        for local, predicted in enumerate(predictions):
            i = start + local
            p, y0, meta = pool.params[i], pool.initial[i], metas[i]
            key = system.equivalence_key(p, y0, meta)
            witness = certificates.get(key) if key is not None else None
            exact = name == 'ours_mechanism_witness' and witness is not None
            j = (int(witness) if exact else int(anchors[i])) if name == 'ours_mechanism_witness' else None
            requests.append((predicted, p, y0, meta,
                train.trajectories[j] if j is not None else None,
                train.inputs.params[j] if j is not None else None,
                train.inputs.initial[j] if j is not None else None,
                train_metas[j] if j is not None else None, exact, scale, times, transport_config))
        for local, (score, feature, counts) in enumerate(score_rows(system, requests, workers)):
            i = start + local
            scores[i], features[i] = score, feature
            for key, value in counts.items():
                report[key] += value
        progress.update(start+len(predictions))
    if not seen.all():
        raise ValueError('Predictor did not score every candidate input')
    return scores, features, report


def select(context, budget, name, *, training_preparation=None):
    if context.label_reader is not None:
        raise PermissionError('Portable ours methods must not receive candidate labels')
    if not 0 < budget <= len(context.pool.ids):
        raise ValueError('Invalid mechanism acquisition budget')
    if name == 'ours_intervention_closure' and any(not pairs(m) for m in _metadata(context.pool)):
        raise ValueError('Intervention closure requires structured legal mutation controls')
    scores, features, report = mechanism_scores(context, name, training_preparation=training_preparation)
    if name == 'ours_intervention_closure':
        pool, train = context.pool, context.train.inputs
        combined = PoolInputs(np.r_[pool.ids, train.ids], np.r_[pool.params, train.params],
            np.r_[pool.initial, train.initial], tuple(_metadata(pool)) + tuple(_metadata(train)))
        faces = rectangles(combined, BiologicalODE(getattr(context, 'system_id', 'budding')), len(pool.ids))
        indices, details = choose_faces(faces, np.r_[features, np.zeros((len(train.ids), features.shape[1]))],
                                       len(pool.ids), budget, scores, context.seed)
        for record in details['completed_faces']:
            record['endpoints'] = [dict(kind='pool' if i < len(pool.ids) else 'queried_train',
                candidate_id=int(combined.ids[i]),
                source_candidate_id=int(combined.metadata[i].get('source_candidate_id', combined.ids[i])))
                for i in record['corners']]
            record['new_candidate_ids'] = [int(pool.ids[i]) for i in record['new_endpoints']]
        report.update(details)
    else:
        rng = np.random.default_rng(context.seed)
        order = rng.permutation(len(scores))
        order = order[np.argsort(-scores[order], kind='stable')]
        exploit = budget - int(.1 * budget)
        chosen = order[:exploit]
        rest = np.setdiff1d(np.arange(len(scores)), chosen)
        indices = np.r_[chosen, rng.choice(rest, budget - exploit, replace=False)]
        report.update(mechanism_queries=exploit, random_exploration=budget - exploit)
    report['selected_scores'] = scores[indices].tolist()
    report['selected_input_ids'] = context.pool.ids[indices].tolist()
    return SamplingResult(context.pool.ids[indices], diagnostics=report)


def select_witness(context, budget):
    return select(context, budget, 'ours_mechanism_witness')


def select_transport(context, budget):
    return select(context, budget, 'ours_dynamics_transport')


def select_closure(context, budget):
    return select(context, budget, 'ours_intervention_closure')
