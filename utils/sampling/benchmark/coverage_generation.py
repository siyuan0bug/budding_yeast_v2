"""Legal generative extension of conditional coverage, with a paired control.

Only common initial labels are supplied to this policy. Unqueried proposals are
screened using nearest-initial INPUT anchors' state paths: these are explicitly
proxy states, not secretly queried proposal trajectories. Every actual ODE
attempt is charged, failures supplemented, successful trajectories written once.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch

from budding_yeast_v2.data.dataset_archive import TrajectoryStore, atomic_json
from budding_yeast_v2.data.simulation_guards import bounded_ordered_map
from budding_yeast_v2.data.systems.budding.offline_pool import simulate
from budding_yeast_v2.data.systems.budding.specification import SPEC
from budding_yeast_v2.utils.experiments.logging import log_event, log_stage
from budding_yeast_v2.utils.experiments.resources import resource_slot
from ..generation.cohorts import verify_frozen_rules
from ..generation.domain import FrozenBuddingDomain
from ..generation.runner import worker_lifecycle
from ..generation.storage import GeneratedTrainingStore
from ..ours.conditional_coverage.features import parse_metadata
from ..ours.conditional_coverage.selection import conditional_vectors, facility_select
from .diagnostics import context_key
from .protocol import effective_signatures, child_seed


def input_anchors(params, initial_params, *, device='cpu'):
    """Nearest observed INPUT, not nearest hidden candidate output."""
    mean = initial_params.mean(0)
    scale = np.maximum(initial_params.std(0), 1e-6)
    base = torch.as_tensor((initial_params-mean)/scale, dtype=torch.float32, device=device)
    output = []
    for start in range(0, len(params), 256):
        value = torch.as_tensor((params[start:start+256]-mean)/scale, dtype=torch.float32, device=device)
        output.extend(torch.cdist(value, base).argmin(1).cpu().tolist())
    return np.asarray(output)


def prepare_generation(root, work, source, cfg, roles, initial, encoder, *, device='cpu', workers=8):
    from .coverage_study import write_selection
    methods = ('random_gap_generate_control_v1', 'ours_mechanism_gap_generate_v1')
    if all((work/'selections'/f'{method}.json').exists() for method in methods):
        return
    verify_frozen_rules(source, 'mutation', None)
    forbidden = set()
    protocol = json.loads((root/'protocol.json').read_text())
    # Both pools' heldout inputs are excluded, without reading their labels.
    for name in protocol['pools']:
        with np.load(root/f'{name}_inputs.npz') as inputs, np.load(root/f'{name}_split.npz') as split:
            held = np.r_[split['validation'], split['test']]
            forbidden.update(effective_signatures(inputs['params'][held], inputs['y_init'][held]))
    protected = np.r_[roles['feedback'], roles['audit'], roles['random_probe']]
    values = source.read(protected, ['params', 'y_init'])
    forbidden.update(effective_signatures(values['params'], values['y_init']))
    seen = set(effective_signatures(initial['params'], initial['y_init']))
    domain = FrozenBuddingDomain('mutation', child_seed(cfg.seed, 'mechanism_gap_proposals'),
        work/'generation_proposals', forbidden=forbidden, seen=seen)
    candidates, signatures, params, initials = [], [], [], []
    excluded_contexts = set(roles['feedback_contexts']) | set(roles['audit_contexts'])
    mdt = SPEC.param_names.index('mdt')
    with log_stage('canonical generative proposal bank', proposals=20000, seed=cfg.seed):
        while len(candidates) < 20000:
            batch, hashes, view = domain.bank(min(512, 20000-len(candidates)))
            for c, signature, p, h in zip(batch, hashes, view.params, view.initial):
                if context_key(c.metadata, p[mdt]) in excluded_contexts:
                    continue
                domain.consume(signature)
                candidates.append(c); signatures.append(signature); params.append(p); initials.append(h)
    params, initials = np.asarray(params), np.asarray(initials)
    meta = tuple(dict(c.metadata, _effective_overrides=dict(c.overrides)) for c in candidates)
    anchors = input_anchors(params, initial['params'], device=device)
    grid = np.linspace(0, initial['data'].shape[-1]-1, 32).round().astype(int)
    states = initial['data'][:, :, grid]
    trajectory = [encoder.trajectory(states)]
    owners, tokens, effects = [], [], []
    original_meta = parse_metadata(initial)
    combined_p = np.r_[initial['params'], params]
    combined_h = np.r_[initial['y_init'], initials]
    combined_meta = original_meta+meta
    for start in range(0, len(combined_p), 256):
        stop = min(start+256, len(combined_p))
        # The first 2000 are observed anchors; later rows are unqueried inputs.
        lookup = np.r_[np.arange(2000), anchors][start:stop]
        proxy = states[lookup]
        if start >= 2000:
            trajectory.append(encoder.trajectory(proxy))
        elif stop > 2000:
            trajectory.append(encoder.trajectory(proxy[2000-start:]))
        ids, action, value = encoder.mechanisms(combined_p[start:stop], combined_h[start:stop],
                                               proxy, combined_meta[start:stop])
        owners.extend((ids+start).tolist()); tokens.extend(action); effects.append(value)
    trajectory = np.concatenate(trajectory)
    owners = np.asarray(owners); effects = np.concatenate(effects)
    vectors = conditional_vectors(trajectory, owners, effects)
    with log_stage('unqueried proposal proxy mechanism coverage'):
        ordering, diagnostics = facility_select(vectors, owners, tokens, 12000, np.arange(2000),
            seed=child_seed(cfg.seed, 'coverage_solver'), device=device)
    ordering = ordering-2000
    # Preserve a 12000-query failure-replacement reserve from the SAME bank.
    random = np.random.default_rng(child_seed(cfg.seed, 'gap_random_order')).permutation(len(candidates))[:12000]
    for method, order in zip(methods, (random, ordering)):
        if (work/'selections'/f'{method}.json').exists():
            continue
        directory = work/'generated'/method
        state_path = directory/'generation.json'
        previous = json.loads(state_path.read_text()) if state_path.exists() else dict(chunks=[], cursor=0, attempts=0, failed=0)
        store = GeneratedTrainingStore(source.path, roles['initial'], directory, 'mutation', cfg.seed,
            chunks=previous['chunks'], generation_metadata=dict(method=method,
                proxy='nearest initial input anchor state path; no unqueried solution used'))
        cursor, attempts, failed = previous['cursor'], previous['attempts'], previous['failed']
        context = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                initializer=worker_lifecycle, initargs=(os.getpid(),)) as executor:
            while store.count < 10000:
                if cursor == len(order):
                    raise RuntimeError('Explicit 12000-query cap reached before 8000 new valid trajectories')
                take = min(128, 10000-store.count, len(order)-cursor)
                ids = order[cursor:cursor+take]
                cursor += take
                with resource_slot('scoring', 'bounded canonical ODE query batch'):
                    results = list(bounded_ordered_map(executor, simulate,
                        [candidates[int(i)] for i in ids], stall_timeout=35.))
                successes = [r for r in results if r.status == 'success']
                attempts += len(results); failed += len(results)-len(successes)
                with (directory/'ode_attempts.jsonl').open('a') as stream:
                    for i, result in zip(ids, results):
                        stream.write(json.dumps(dict(candidate=int(i), signature=signatures[int(i)],
                            status=result.status, diagnostics=result.diagnostics), default=lambda v: v.item(), allow_nan=False)+'\n')
                    stream.flush(); os.fsync(stream.fileno())
                store.append(successes)
                atomic_json(state_path, dict(chunks=store.chunks, cursor=cursor, attempts=attempts, failed=failed))
                log_event('PROGRESS', 'canonical generated data', method=method, seed=cfg.seed,
                    accepted=store.count, attempts=attempts, failed=failed, target=10000)
        with resource_slot('data', 'publish immutable generated source'):
            published = store.publish(0)
        info = dict(diagnostics if method == methods[1] else dict(objective='uniform order of identical legal proposal bank'),
            initial_labels_accessed=2000, unqueried_proposal_inputs=len(candidates),
            new_ODE_attempts=attempts, new_ODE_successes=8000, new_ODE_failures=failed,
            unique_training_inputs=10000, proxy='nearest initial input anchor trajectory, not candidate true trajectory',
            queried_solution_labels_used_to_rank_proposals=False, original_pool_labels_used_to_rank_proposals=2000,
            complete_feedback_and_audit_contexts_excluded=True)
        write_selection(work, method, TrajectoryStore(published), np.arange(10000), info)
