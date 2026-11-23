"""Prepared cohorts and model-independent coverage experiments.

One final 10k cold start per arm, approximately 47k updates. This is a research
screen, explicitly NOT the historical five-round benchmark protocol. Source
trajectories are referenced, and all complete feedback backgrounds are excluded
from every training arm. No test/real truth participates in preparation.
"""
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from budding_yeast_v2.data.dataset_archive import TrajectoryStore, atomic_json, file_sha256
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.data.systems.budding.specification import SPEC
from budding_yeast_v2.utils.evaluation.checkpoints import Predictor
from budding_yeast_v2.utils.experiments.logging import log_stage, log_event
from budding_yeast_v2.utils.experiments.resources import resource_slot
from budding_yeast_v2.utils.sampling.contracts import PoolInputs, QueriedTraining
from budding_yeast_v2.utils.sampling.layout import experiment_directory
from .protocol import BenchmarkConfig, child_seed
from .rounds import fit_round
from .runner import source_inventory
from .diagnostics import context_key, reserve_training_roles, evaluate_reference_cohort
from ..ours.conditional_coverage.features import MechanismFeatures, parse_metadata, intervention_groups
from ..ours.conditional_coverage.selection import conditional_vectors, facility_select, transfer_select

METHODS = ('random_coverage_control_v1', 'trajectory_coverage_v1', 'context_coverage_v1',
           'ours_mechanism_coverage_v1', 'ours_composition_transfer_v1',
           'random_gap_generate_control_v1', 'ours_mechanism_gap_generate_v1')
FIELDS = ('params', 'y_init', 'data', 'candidate_metadata_json', 'overrides_json')


def definition(root):
    value = json.loads((root/'protocol.json').read_text())
    for name, digest in value['artifacts'].items():
        if file_sha256(root/name) != digest:
            raise ValueError(f'Shared artifact changed: {name}')
    source = TrajectoryStore(value['pools']['mutation']['path'])
    if source.digest != value['pools']['mutation']['sha256']:
        raise ValueError('Frozen source changed')
    return value, source


def study_config(protocol):
    base = BenchmarkConfig(**protocol['config']).validate()
    return replace(base, budgets=(10000,), epochs=(595,), warmup=(30,)).validate()


def subset_inputs(arrays, rows, metadata=None):
    return PoolInputs(np.asarray(rows), arrays['params'], arrays['y_init'],
                      parse_metadata(arrays) if metadata is None else metadata)


def read(root, source, rows, fields=FIELDS):
    with resource_slot('data', 'bounded study source read'):
        return source.read(rows, list(fields))


def reserve(root, source, seed):
    with np.load(root/'mutation_split.npz') as split:
        candidates, initial = split['candidates'].copy(), split['initial'].copy()
        validation, test = split['validation'].copy(), split['test'].copy()
    arrays = read(root, source, candidates, ('params', 'candidate_metadata_json'))
    mdt = SPEC.param_names.index('mdt')
    keys = [context_key(json.loads(str(m)), p[mdt]) for m, p in zip(arrays['candidate_metadata_json'], arrays['params'])]
    roles = reserve_training_roles(candidates, keys, initial, shortlist_size=1, probe_size=512,
                                  seed=child_seed(seed, 'conditional_coverage_reservations'))
    reserved = set(roles['feedback_contexts']) | set(roles['audit_contexts'])
    initial_set = set(initial)
    allowed = np.array([r for r, key in zip(candidates, keys) if key not in reserved and r not in initial_set])
    rng = np.random.default_rng(child_seed(seed, 'conditional_random_probe'))
    random_probe = rng.choice(allowed, 512, replace=False)
    allowed = allowed[~np.isin(allowed, random_probe)]
    return dict(initial=initial.tolist(), eligible=allowed.tolist(), feedback=roles['feedback'].tolist(),
        feedback_contexts=roles['feedback_contexts'], audit=roles['audit'].tolist(),
        audit_contexts=roles['audit_contexts'], random_probe=random_probe.tolist(),
        validation=validation.tolist(), test=test.tolist())


def feature_bank(arrays, encoder, device):
    trajectory, owners, tokens, effects = [], [], [], []
    meta = parse_metadata(arrays)
    for start in range(0, len(meta), 256):
        stop = min(start+256, len(meta))
        truth = arrays['data'][start:stop]
        grid = np.linspace(0, truth.shape[-1]-1, 32).round().astype(int)
        states = truth[:, :, grid]
        trajectory.append(encoder.trajectory(states))
        ids, action, feature = encoder.mechanisms(arrays['params'][start:stop],
            arrays['y_init'][start:stop], states, meta[start:stop])
        owners.extend((ids+start).tolist()); tokens.extend(action); effects.append(feature)
        if start % 4096 == 0 or stop == len(meta):
            log_event('PROGRESS', 'mechanism feature construction', completed=stop, total=len(meta))
    trajectory = np.concatenate(trajectory)
    owners = np.asarray(owners)
    effects = np.concatenate(effects)
    return dict(trajectory=trajectory, owners=owners, tokens=np.asarray(tokens), effects=effects)


def write_selection(work, method, source, rows, diagnostics):
    directory = work/'selections'; directory.mkdir(exist_ok=True)
    path = directory/f'{method}.json'
    if path.exists():
        raise FileExistsError(path)
    source.snapshot(path, np.asarray(rows), provenance=dict(workflow='conditional_dynamics_screen_v1',
        method=method, diagnostics=diagnostics, real_labels_used=False, test_labels_used=False))
    return path


def prepare(root, work, *, device='cpu', workers=8):
    protocol, source = definition(root)
    cfg = study_config(protocol)
    started = time.monotonic(); sources = source_inventory()
    config_path = work/'study.json'
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved['sources'] != sources or saved['protocol_sha256'] != file_sha256(root/'protocol.json'):
            raise ValueError('Prepared study source/config identity changed')
        if (work/'pool_prepared.json').exists():
            done = json.loads((work/'pool_prepared.json').read_text())
            for name, digest in done['artifacts'].items():
                if file_sha256(work/name) != digest:
                    raise ValueError('Prepared selection changed')
            return
        roles = saved['roles']
    else:
        roles = reserve(root, source, cfg.seed)
        atomic_json(config_path, dict(workflow='conditional_dynamics_screen_v1', training=asdict(cfg),
            protocol=str(root), protocol_sha256=file_sha256(root/'protocol.json'), sources=sources,
            roles=roles, methods=METHODS, model_seed=child_seed(cfg.seed, 'conditional_coverage_final_model'),
            information='supervised pool selection uses all eligible true trajectories; generation only uses initial truths',
            final_training='cold_start_10000_rows_595_epochs_47005_updates; not five-round benchmark',
            normalization='fit each final selected training cohort; unchanged shared log1p/Zscore implementation',
            independent_audit='complete background holdout, never scored by selectors',
            real_labels_used_for_selection=False, test_labels_used_for_selection=False))
    rows = np.r_[roles['initial'], roles['eligible']]
    arrays = read(root, source, rows)
    common = NormalizationStats.fit(arrays['params'][:2000], arrays['data'][:2000])
    encoder = MechanismFeatures(arrays['data'][:2000], seed=child_seed(cfg.seed, 'mechanism_features'), device=device)
    feature_path = work/'features.npz'
    if feature_path.exists():
        with np.load(feature_path) as value: features = {k: value[k] for k in value.files}
    else:
        with resource_slot('scoring', 'conditional dynamics features'):
            features = feature_bank(arrays, encoder, device)
        np.savez_compressed(feature_path, **features)
    initial = np.arange(2000)
    selection_seed = child_seed(cfg.seed, 'coverage_solver')
    rng = np.random.default_rng(child_seed(cfg.seed, 'coverage_random_control'))
    selections = {}
    for method in METHODS[:4]:
        if (work/'selections'/f'{method}.json').exists():
            continue
        with log_stage('full-budget subset construction', method=method, seed=cfg.seed):
            if method == METHODS[0]:
                chosen = rng.choice(np.arange(2000, len(rows)), 8000, replace=False)
                info = dict(objective='uniform_without_replacement')
            elif method == METHODS[1]:
                chosen, info = facility_select(features['trajectory'], np.arange(len(rows)),
                    ['trajectory']*len(rows), 8000, initial, seed=selection_seed, device=device, conditional=False)
            else:
                x = (conditional_vectors(features['trajectory'], features['owners'], features['effects'])
                     if method == METHODS[3] else features['trajectory'][features['owners']])
                chosen, info = facility_select(x, features['owners'], features['tokens'], 8000,
                    initial, seed=selection_seed, device=device)
            info.update(trajectory_labels_accessed=len(rows), unique_training_inputs=10000,
                new_ODE_attempts=0, feedback_labels_accessed=0)
            selections[method] = write_selection(work, method, source, rows[np.r_[initial, chosen]], info)
            atomic_json(work/'progress.json', dict(stage='selection', method=method, completed=list(selections)))
    del features
    gc.collect()
    prepare_transfer(root, work, source, cfg, roles, rows, arrays, common, device)
    del arrays
    gc.collect()
    if source_inventory() != sources:
        raise RuntimeError('Experiment sources changed during preparation')
    artifacts = {str(p.relative_to(work)): file_sha256(p) for p in (work/'selections').glob('*.json')}
    if not all(f'selections/{method}.json' in artifacts for method in METHODS[:5]):
        raise RuntimeError('Not all pool methods have prepared full-budget selections')
    atomic_json(work/'pool_prepared.json', dict(status='complete', artifacts=artifacts,
        seconds=time.monotonic()-started, sources_unchanged=True))


def generate(root, work, *, device='cpu', workers=8):
    from .coverage_generation import prepare_generation
    protocol, source = definition(root)
    cfg = study_config(protocol)
    settings = json.loads((work/'study.json').read_text())
    if settings['sources'] != source_inventory():
        raise ValueError('Prepared source identity changed')
    initial = read(root, source, settings['roles']['initial'])
    encoder = MechanismFeatures(initial['data'], seed=child_seed(cfg.seed, 'mechanism_features'), device=device)
    prepare_generation(root, work, source, cfg, settings['roles'], initial, encoder, device=device, workers=workers)
    artifacts = {str(p.relative_to(work)): file_sha256(p) for p in (work/'selections').glob('*.json')}
    if len(artifacts) != len(METHODS) or source_inventory() != settings['sources']:
        raise RuntimeError('Incomplete generation or source identity changed')
    atomic_json(work/'prepared.json', dict(status='complete', artifacts=artifacts))


def prepare_transfer(root, work, source, cfg, roles, rows, arrays, common, device):
    method = 'ours_composition_transfer_v1'
    if (work/'selections'/f'{method}.json').exists():
        return
    from ..baseline.regression import gradient_bank
    from .diagnostics import evaluate_probe
    initial_rows = rows[:2000]
    manifest = work/'initial_training.json'
    source.snapshot(manifest, initial_rows)
    initial = subset_inputs({k: v[:2000] for k, v in arrays.items()}, initial_rows)
    validation = read(root, source, roles['validation'], ('params', 'y_init', 'data'))
    pilot_cfg = replace(cfg, budgets=(2000,), epochs=(100,), warmup=(5,)).validate()
    with log_stage('common initial pilot for compositional selection', seed=cfg.seed):
        model, norm, history = fit_round(pilot_cfg, 0, QueriedTraining(initial, arrays['data'][:2000]),
            [(subset_inputs(validation, roles['validation'], ()), validation['data'])], work/'pilot',
            child_seed(cfg.seed, 'conditional_coverage_pilot'), device=device, dataset_path=manifest,
            state_dir=work/'_state/pilot', identity=dict(workflow='conditional_composition_pilot'))
    predictor = Predictor([model], norm, arrays['data'].shape[-1], device, 64,
                          sketch_seed=child_seed(cfg.seed, 'composition_gradient_sketch'))
    candidates = subset_inputs({k: v[2000:] for k, v in arrays.items()}, rows[2000:])
    feedback = read(root, source, roles['feedback'])
    probe = subset_inputs(feedback, roles['feedback'])
    forbidden_contexts = set(roles['feedback_contexts']) | set(roles['audit_contexts'])
    mdt = SPEC.param_names.index('mdt')
    if any(context_key(m, p[mdt]) in forbidden_contexts for m, p in zip(candidates.metadata, candidates.params)):
        raise ValueError('Transfer episode context leakage')
    index = {int(r): i for i, r in enumerate(rows)}
    with log_stage('cross-background gradient bank', candidates=len(candidates.ids)):
        gradient = gradient_bank(predictor, candidates, lambda ids: dict(data=arrays['data'][[index[int(r)] for r in ids]]), dimension=128)
        feedback_index = {int(r): i for i, r in enumerate(probe.ids)}
        feedback_gradient = gradient_bank(predictor, probe,
            lambda ids: dict(data=feedback['data'][[feedback_index[int(r)] for r in ids]]), dimension=128)
        feedback_metrics = evaluate_probe(predictor, probe, feedback['data'], per_sample=True)
        actions = lambda metadata: [{token for token, _ in intervention_groups(m)} for m in metadata]
        chosen, info = transfer_select(gradient, actions(candidates.metadata), feedback_gradient,
            actions(probe.metadata), feedback_metrics['per_sample']['normalized_relative'], 8000,
            seed=child_seed(cfg.seed, 'composition_transfer_selection'), device=device)
    info.update(trajectory_labels_accessed=len(rows), feedback_labels_accessed=len(probe.ids),
        pilot_optimizer_steps=history[-1]['optimizer_steps'], unique_training_inputs=10000,
        complete_feedback_contexts_excluded=True, new_ODE_attempts=0)
    write_selection(work, method, source, np.r_[initial_rows, candidates.ids[chosen]], info)
    del predictor, model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def train(root, work, results, method, *, device='cpu', plots=True, wandb_run=None):
    if method not in METHODS:
        raise ValueError('Unknown study arm')
    protocol, original = definition(root)
    cfg = study_config(protocol)
    settings = json.loads((work/'study.json').read_text())
    ready_path = work/('prepared.json' if method in METHODS[5:] else 'pool_prepared.json')
    ready = json.loads(ready_path.read_text())
    if settings['sources'] != source_inventory():
        raise ValueError('Prepared study sources changed')
    selection = work/'selections'/f'{method}.json'
    if file_sha256(selection) != ready['artifacts'][str(selection.relative_to(work))]:
        raise ValueError('Prepared selection changed')
    snapshot = json.loads(selection.read_text())
    source = TrajectoryStore(snapshot['source'])
    rows = np.asarray(snapshot['rows'], dtype=np.int64)
    if source.digest != snapshot['source_sha256'] or len(rows) != 10000:
        raise ValueError('Need the immutable prepared 10k selection')
    directory = experiment_directory(results, cfg, 'mutation', method)
    started = time.monotonic(); sources = source_inventory()
    config = dict(workflow='conditional_dynamics_screen_v1', training=asdict(cfg),
        shared_study=str(work), shared_study_sha256=file_sha256(work/'study.json'),
        selection_sha256=file_sha256(selection), method=method, sources=sources,
        source_protocol=str(root), real_evaluated_only_after_best_locked=True,
        model_seed=settings['model_seed'])
    if (directory/'config.json').exists() and json.loads((directory/'config.json').read_text()) != config:
        raise ValueError('Result configuration changed')
    atomic_json(directory/'config.json', config)
    if (directory/'completion.json').exists():
        return directory
    (directory/'data').mkdir(exist_ok=True)
    training_path = directory/'data/dataset.json'
    source.snapshot(training_path, rows, provenance=dict(shared_selection=str(selection),
        unique_training_inputs=10000, source_trajectories_copied=False))
    arrays = read(root, source, rows, ('params', 'y_init', 'data'))
    validation = read(root, original, settings['roles']['validation'], ('params', 'y_init', 'data'))
    common_arrays = read(root, original, settings['roles']['initial'], ('params', 'data'))
    common = NormalizationStats.fit(common_arrays['params'], common_arrays['data'])
    del common_arrays
    with log_stage('cold-start full-budget training', method=method, seed=cfg.seed, epochs=595):
        model, norm, history = fit_round(cfg, 0, QueriedTraining(subset_inputs(arrays, rows, ()), arrays['data']),
            [(subset_inputs(validation, settings['roles']['validation'], ()), validation['data'])],
            directory/'training', settings['model_seed'], device=device, dataset_path=training_path,
            state_dir=directory/'_state/primary', identity=config, plot_context=dict(training_rows=10000),
            wandb_run=wandb_run)
    if history[-1]['optimizer_steps'] != 47005:
        raise RuntimeError('Final optimizer budget changed')
    predictor = Predictor([model], norm, arrays['data'].shape[-1], device, cfg.prediction_batch_size)
    del arrays, validation
    gc.collect()
    metrics = {}
    for name, role in [('random_inputs', 'random_probe'), ('unseen_contexts', 'audit'), ('synth_test', 'test')]:
        cohort = read(root, original, settings['roles'][role], ('params', 'y_init', 'data'))
        with log_stage('locked-best independent evaluation', cohort=name):
            metrics[name] = evaluate_reference_cohort(predictor,
                subset_inputs(cohort, settings['roles'][role], ()), cohort['data'], common,
                directory/'eval'/name, variable_names=SPEC.variable_names,
                common_reference='common_initial_2000_train_stats', sample_ids=settings['roles'][role])
        del cohort
    real = protocol['real_reference']
    if file_sha256(real['path']) != real['sha256']:
        raise ValueError('Frozen real reference changed')
    with np.load(real['path'], allow_pickle=False) as archive:
        cohort = {k: archive[k] for k in ('params', 'y_init', 'data', 'mutant_names')}
    with log_stage('locked-best real mutant evaluation', samples=len(cohort['data'])):
        metrics['real_mutants'] = evaluate_reference_cohort(predictor,
            subset_inputs(cohort, np.arange(len(cohort['data'])), ()), cohort['data'], common,
            directory/'eval', variable_names=SPEC.variable_names, sample_names=cohort['mutant_names'],
            real_plots=plots, common_reference='common_initial_2000_train_stats')
    from budding_yeast_v2.utils.evaluation.reporting import save_global_metrics
    save_global_metrics(directory/'eval', dict(real_mutants=metrics['real_mutants']['physical'],
        mutation_test=metrics['synth_test']['physical']), pool_name='mutation', round_index=0, training_rows=10000)
    if source_inventory() != sources:
        raise RuntimeError('Sources changed during experiment')
    report = dict(status='complete', method=method, seed=cfg.seed, run=str(directory),
        metrics=metrics, training_updates=history[-1]['optimizer_steps'], seconds=time.monotonic()-started,
        selection_cost=snapshot.get('provenance', {}).get('diagnostics', {}),
        real_labels_used_for_selection=False, sources_unchanged=True)
    atomic_json(directory/'completion.json', report)
    if wandb_run is not None:
        wandb_run.log({f'final/{cohort}/{key}': value
            for cohort, result in metrics.items() for key, value in result['physical']['all_variables'].items()})
    return directory
