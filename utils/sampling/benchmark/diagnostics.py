"""Shared adapters for small training-side sampling diagnostics, not a sampler.

Feedback labels may rank trial batches; an independently reserved audit cohort
must never participate in that ranking. Full-pool label access is accounted for
explicitly by the diagnostic workflow.
"""
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from budding_yeast_v2.utils.evaluation.metrics import summarize, sample_channel_relative_errors
from budding_yeast_v2.utils.training.losses import RelativeL2Loss
from ..generation.replay import TrajectoryReplay


def context_key(metadata, mdt):
    pairs = metadata.get('pairs', metadata.get('genes', ()))
    return json.dumps([sorted(map(list, pairs)), float(mdt)], separators=(',', ':'))


class FrozenReplayModule(pl.LightningDataModule):
    """Existing replay encoder, with explicitly frozen checkpoint statistics."""
    def __init__(self, norm, params, initial, truth, steps, batch, seed):
        super().__init__()
        norm.bind(self)
        self.norm_type, self.system_id, self.t_max = norm.norm_type, 'budding', 210.
        self.replay = TrajectoryReplay(norm, params, initial, truth,
            capacity=len(params), draws=steps*batch, seed=seed)
        self.batch = batch

    def train_dataloader(self):
        return DataLoader(self.replay, batch_size=self.batch, shuffle=False,
            num_workers=0, pin_memory=True)


def evaluate_probe(predictor, inputs, truth, *, per_sample=False, exclude_channels=()):
    values = np.concatenate([value.numpy() for _, value in predictor.batches(inputs)])
    y = predictor.normalization.targets(truth)
    errors = RelativeL2Loss(eps=.01).per_sample(torch.as_tensor(values), y).numpy()
    physical = predictor.normalization.decode(torch.as_tensor(values)).numpy()
    result = dict(normalized_relative=float(errors.mean()), physical=summarize(physical, truth))
    if exclude_channels:
        keep = [j for j in range(truth.shape[1]) if j not in set(exclude_channels)]
        if not keep: raise ValueError('Cannot exclude all variables')
        result['coupled_normalized_relative'] = float(RelativeL2Loss(eps=.01)(
            torch.as_tensor(values[:, keep]), y[:, keep]))
        result['coupled_MAE'] = float(np.abs(physical[:, keep]-truth[:, keep]).mean())
    if per_sample:
        delta = physical.astype(np.float64)-np.asarray(truth, dtype=np.float64)
        result['per_sample'] = dict(normalized_relative=errors.tolist(),
            physical_mae=np.abs(delta).mean((1, 2)).tolist(),
            physical_mse=np.square(delta).mean((1, 2)).tolist(),
            physical_relative=sample_channel_relative_errors(physical, truth).mean(1).tolist())
    return result


def evaluate_reference_cohort(predictor, inputs, truth, common_norm, directory, *,
                              variable_names=None, sample_names=None, real_plots=False,
                              common_reference='original_10k_train_fitted_stats', sample_ids=None,
                              return_per_sample=False):
    """Physical and common-coordinate evaluation, shared by reference workflows."""
    from budding_yeast_v2.data.dataset_archive import atomic_json
    from budding_yeast_v2.utils.evaluation.metrics import relative_error_decomposition
    from budding_yeast_v2.utils.evaluation.reporting import save_real_evaluation
    import csv
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = torch.cat([value for _, value in predictor.batches(inputs)])
    prediction = predictor.normalization.decode(raw).numpy()
    if sample_ids is not None:
        from budding_yeast_v2.utils.evaluation.reporting import export_sample_metrics
        export_sample_metrics(prediction, truth, [f'pool_row_{int(i)}' for i in sample_ids],
            directory/'per_sample_metrics.csv', source='synthetic_probe_or_test',
            pool_rows=sample_ids, candidate_ids=sample_ids)
    common_value = RelativeL2Loss(eps=.01).per_sample(common_norm.targets(prediction), common_norm.targets(truth))
    own = RelativeL2Loss(eps=.01).per_sample(raw, predictor.normalization.targets(truth))
    physical = (summarize(prediction, truth) if sample_names is None else
        save_real_evaluation(prediction, truth, sample_names, directory, plots=real_plots,
                             write_summary=True, plot_workers=2))
    metrics = dict(physical=physical, common_reference_normalized_relative=float(common_value.mean()),
        common_reference=common_reference,
        own_normalized_relative_not_cross_model_comparable=float(own.mean()))
    if return_per_sample:
        delta = prediction.astype(np.float64)-np.asarray(truth, dtype=np.float64)
        metrics['per_sample'] = dict(normalized_relative=common_value.tolist(),
            physical_mae=np.abs(delta).mean((1, 2)).tolist(),
            physical_mse=np.square(delta).mean((1, 2)).tolist(),
            physical_relative=sample_channel_relative_errors(prediction, truth).mean(1).tolist())
    atomic_json(directory/'metrics.json', metrics)
    decomposition = relative_error_decomposition(prediction, truth, variable_names=variable_names)
    atomic_json(directory/'relative_decomposition.json', decomposition)
    with (directory/'per_variable.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decomposition['per_variable'][0]))
        writer.writeheader(); writer.writerows(decomposition['per_variable'])
    return metrics


def match_update_schedule(training_rows, reference_rows, batch_size, reference_epochs, reference_warmup):
    """Nearest integer epoch schedule; report remaining step differences explicitly."""
    if min(training_rows, reference_rows, batch_size, reference_epochs) < 1 or not 0 <= reference_warmup < reference_epochs:
        raise ValueError('Invalid data/update budget')
    steps = math.ceil(training_rows/batch_size)
    ref_steps = math.ceil(reference_rows/batch_size)
    epochs = max(1, round(reference_epochs*ref_steps/steps))
    warmup = min(epochs-1, round(reference_warmup*ref_steps/steps))
    return dict(epochs=epochs, warmup=warmup, steps_per_epoch=steps,
        reference_steps_per_epoch=ref_steps, optimizer_steps=steps*epochs,
        reference_optimizer_steps=ref_steps*reference_epochs,
        warmup_steps=steps*warmup, reference_warmup_steps=ref_steps*reference_warmup)


def reserve_training_roles(row_ids, keys, initial_rows, *, shortlist_size, probe_size, seed):
    """Input-only grouping: feedback/audit backgrounds never enter train trials."""
    rows = np.asarray(row_ids, dtype=np.int64)
    initial = set(map(int, initial_rows))
    if len(set(rows)) != len(rows) or len(keys) != len(rows) or not initial <= set(rows):
        raise ValueError('Initial rows must be unique members of the training side')
    known = {key for row, key in zip(rows, keys) if int(row) in initial}
    by_context = defaultdict(list)
    for row, key in zip(rows, keys):
        if key not in known:
            by_context[key].append(int(row))
    rng = np.random.default_rng(seed)
    contexts = sorted(by_context)
    rng.shuffle(contexts)
    reserved = set()
    roles = {}
    cursor = 0
    for role in ('feedback', 'audit'):
        selected, selected_contexts = [], []
        while len(selected) < probe_size and cursor < len(contexts):
            key = contexts[cursor]
            cursor += 1
            count = min(2, len(by_context[key]), probe_size-len(selected))
            selected.extend(map(int, rng.choice(by_context[key], count, replace=False)))
            selected_contexts.append(key)
        if len(selected) != probe_size:
            raise ValueError('Insufficient unseen training-side backgrounds for two probes')
        reserved.update(selected_contexts)
        roles[role] = np.array(selected, dtype=np.int64)
        roles[role+'_contexts'] = selected_contexts
    available = [int(row) for row, key in zip(rows, keys) if int(row) not in initial and key not in reserved]
    if len(available) < shortlist_size:
        raise ValueError('Insufficient unreserved training-side shortlist')
    roles['shortlist'] = np.sort(rng.choice(available, shortlist_size, replace=False))
    return roles


def make_trial_batches(scores, budget, *, trials, seed):
    """One random batch, one true-error batch, then independent random batches."""
    scores = np.asarray(scores)
    if (scores.ndim != 1 or not np.isfinite(scores).all() or not 1 <= budget <= len(scores)
            or trials < 3):
        raise ValueError('Invalid finite shortlist scores/batch budget')
    rng = np.random.default_rng(seed)
    result = {'random': np.sort(rng.choice(len(scores), budget, replace=False)),
              'true_error': np.sort(np.argsort(-scores, kind='stable')[:budget])}
    for i in range(trials-2):
        result[f'random_trial_{i:02d}'] = np.sort(rng.choice(len(scores), budget, replace=False))
    return result


def best_feedback_trial(feedback_metrics):
    """No audit metrics/reader are accepted by this selection interface."""
    if not feedback_metrics or not all(np.isfinite(x['normalized_relative']) for x in feedback_metrics.values()):
        raise ValueError('Trial feedback must be finite and nonempty')
    return min(feedback_metrics, key=lambda name: (feedback_metrics[name]['normalized_relative'], name))


def reserve_volume_roles(row_ids, keys, existing_rows, *, total_rows, random_probe_size,
                         unseen_probe_size, seed, protected_rows=()):
    """Nested random training sets plus random-input and unseen-context probes."""
    rows = np.asarray(row_ids, dtype=np.int64)
    existing = np.asarray(existing_rows, dtype=np.int64)
    if (len(set(rows)) != len(rows) or len(keys) != len(rows)
            or len(set(existing)) != len(existing) or not set(existing) <= set(rows)
            or total_rows <= len(existing) or min(random_probe_size, unseen_probe_size) < 1):
        raise ValueError('Invalid nested data-volume comparison')
    rng = np.random.default_rng(seed)
    known, protected = set(map(int, existing)), set(map(int, protected_rows))
    excluded = known | protected
    lookup = dict(zip(map(int, rows), keys))
    available = [int(row) for row in rows if int(row) not in excluded]
    if len(available) < random_probe_size:
        raise ValueError('Insufficient random probe rows')
    random_probe = np.asarray(rng.choice(available, random_probe_size, replace=False), dtype=np.int64)
    avoid_contexts = {lookup[int(row)] for row in np.r_[existing, random_probe]}
    by_context = defaultdict(list)
    for row in available:
        if lookup[row] not in avoid_contexts:
            by_context[lookup[row]].append(row)
    contexts = sorted(by_context)
    rng.shuffle(contexts)
    unseen, reserved = [], []
    for key in contexts:
        take = min(2, len(by_context[key]), unseen_probe_size-len(unseen))
        unseen.extend(map(int, rng.choice(by_context[key], take, replace=False)))
        reserved.append(key)
        if len(unseen) == unseen_probe_size:
            break
    if len(unseen) != unseen_probe_size:
        raise ValueError('Insufficient unseen contexts for secondary probe')
    unavailable = set(map(int, random_probe)) | set(unseen)
    reserved = set(reserved)
    candidates = [row for row in available if row not in unavailable and lookup[row] not in reserved]
    if len(candidates) < total_rows-len(existing):
        raise ValueError('Insufficient candidates after internal probe reservations')
    added = np.asarray(rng.choice(candidates, total_rows-len(existing), replace=False), dtype=np.int64)
    return dict(training=np.r_[existing, added], random_probe=random_probe,
        unseen_probe=np.asarray(unseen, dtype=np.int64), unseen_contexts=sorted(reserved))
