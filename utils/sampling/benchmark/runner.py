"""Configured cold-start rounds over a complete pool, with auditable truth revelation."""
from contextlib import contextmanager
from dataclasses import asdict
import csv
import fcntl
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np
import torch

from ..contracts import PoolInputs, QueriedTraining, SamplingContext
from ..query import LabelOracle
from ..registry import method_spec, select
from budding_yeast_v2.utils.evaluation.checkpoints import Predictor
from .protocol import BenchmarkConfig, child_seed
from budding_yeast_v2.data.dataset_archive import TrajectoryStore, atomic_json, fingerprint
from budding_yeast_v2.data.dataset_archive import file_sha256
from budding_yeast_v2.utils.experiments.logging import log_event, log_stage
from budding_yeast_v2.utils.experiments.resources import resource_slot, exclusive_gpu_slot
from .rounds import fit_round
from budding_yeast_v2.utils.evaluation.reporting import save_real_evaluation, export_sample_metrics, save_global_metrics
from budding_yeast_v2.utils.experiments.state import update_section, read_section
from budding_yeast_v2.utils.evaluation.metrics import summarize


def source_inventory():
    root = Path(__file__).resolve().parents[2]
    files = [*root.glob('sampling/**/*.py'), *root.glob('training/*.py'),
             *root.glob('physics/*.py'), *root.glob('experiments/*.py'),
             *root.glob('evaluation/*.py'),
             root.parent / 'train.py', root.parent / 'eval.py',
             root.parent / 'data/yeast_datamodule.py',
             root.parent / 'data/dataset_utils.py',root.parent / 'data/preprocessing_contract.py',
             *root.parent.glob('models/**/*.py'),
             root.parent / 'data/normalization.py',
             root.parent / 'data/dataset_archive.py',
             *[root.parent / 'data/systems/budding' / name for name in
               ('ode.py', 'states.py', 'parameters.py', 'real_mutants.py', 'pattern_groups.py', 'catalog.py')]]
    return {str(path): file_sha256(path) for path in sorted(files)}


def input_view(inputs, rows, metadata=None):
    rows = np.asarray(rows, dtype=np.int64)
    return PoolInputs(rows, inputs['params'][rows], inputs['y_init'][rows],
                      tuple(metadata[i] for i in rows) if metadata is not None else ())


@contextmanager
def experiment_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.run.lock').open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process owns {directory}') from None
        yield


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def training_costs(directory):
    """Count primary, auxiliary and native pilot training separately, including resumes."""
    totals = {role: dict(models=0, optimizer_steps=0, seconds=0.)
              for role in ('primary', 'auxiliary', 'pilot')}
    for path in directory.rglob('training.json'):
        row = json.loads(path.read_text())
        role = 'pilot' if 'pilot' in path.relative_to(directory).parts else (
            'auxiliary' if int(row.get('member', 0)) else 'primary')
        totals[role]['models'] += 1
        totals[role]['optimizer_steps'] += int(row['optimizer_steps'])
        totals[role]['seconds'] += float(row['seconds'])
    return totals


def run_experiment(protocol_path, pool_name, method, output, *, device='cpu', real_reference=None,
                   plots=False, wandb_enabled=False, diagnostic_only=False):
    protocol_path = Path(protocol_path).resolve()
    root = protocol_path if protocol_path.is_dir() else protocol_path.parent
    protocol = json.loads((root / 'protocol.json').read_text())
    stored_config = dict(protocol['config'])
    stored_config.setdefault('validation_scope', 'both')  # Historical explicit protocols.
    config = BenchmarkConfig(**stored_config).validate()
    spec = method_spec(method)
    if spec.mode=='dynamic':
        from ..dynamic.runner import run_dynamic
        return run_dynamic(root,pool_name,method,output,device=device,plots=plots,
                           wandb_enabled=wandb_enabled,diagnostic_only=diagnostic_only)
    if spec.status != 'implemented':
        raise NotImplementedError(f'{method}: {spec.note}')
    if spec.mutation_only and pool_name != 'mutation':
        raise ValueError(f'{method} requires mutation metadata; not supported on {pool_name}')
    if real_reference is None and not diagnostic_only:
        raise ValueError('Formal benchmark requires an explicit frozen real reference')
    directory = Path(output) / protocol['protocol_id'] / pool_name / method / f'seed_{config.seed}'
    with experiment_lock(directory), resource_slot('experiment', 'experiment admission'), exclusive_gpu_slot(device):
        return _run(root, protocol, config, pool_name, spec, directory, device,
                    real_reference, plots, wandb_enabled, diagnostic_only)


def _run(root, protocol, config, pool_name, spec, directory, device, real_reference, plots, wandb_enabled, diagnostic_only):
    for relative, digest in protocol['artifacts'].items():
        if file_sha256(root / relative) != digest:
            raise ValueError(f'Frozen protocol artifact changed: {relative}')
    inputs_file = root / f'{pool_name}_inputs.npz'
    split_file = root / f'{pool_name}_split.npz'
    for path in (inputs_file, split_file):
        if file_sha256(path) != protocol['artifacts'][path.name]:
            raise ValueError('Shared input/split artifact changed')
    store = TrajectoryStore(protocol['pools'][pool_name]['path'])
    if store.digest != protocol['pools'][pool_name]['sha256']:
        raise ValueError('Frozen complete pool changed')
    with np.load(inputs_file, allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    with np.load(split_file, allow_pickle=False) as archive:
        splits = {key: archive[key] for key in archive.files}
    identity = dict(protocol_fingerprint=protocol['fingerprint'], pool=pool_name,
        method=asdict(spec), device_type=str(device).split(':')[0], diagnostic_only=diagnostic_only,
        real_reference=None if real_reference is None else dict(path=str(Path(real_reference).resolve()),
                                                               sha256=file_sha256(real_reference)))
    identity.update(output_schema='sampling_report_v5_physical_sc_smooth_eps001',training=asdict(config),sources=source_inventory())
    path=directory/'config.json'
    if path.exists() and json.loads(path.read_text())!=identity:
        raise ValueError(f'Experiment source/config changed: {path}; use a new result directory')
    atomic_json(path,identity)
    state_path=directory/'_state/state.json'
    if read_section(state_path,'completion') is not None:
        log_event('SKIP','completed offline sampling run',path=directory)
        return directory
    oracle=LabelOracle(store,splits['candidates'],state_path,spec.track,state_key='oracle')
    test_metadata={}
    validation, tests = [], []
    with resource_slot('data', 'shared holdout loading'), log_stage('shared holdout loading', pool=pool_name):
        for name, source in sorted(protocol['pools'].items()):
            other = store if name == pool_name else TrajectoryStore(source['path'])
            if other.digest != source['sha256']:
                raise ValueError('Shared evaluation pool changed')
            with np.load(root / f'{name}_inputs.npz', allow_pickle=False) as a:
                other_inputs = {key:a[key] for key in a.files}
            split_path = root / f'{name}_split.npz'
            if file_sha256(split_path) != protocol['artifacts'][split_path.name]:
                raise ValueError('Shared validation/test split changed')
            with np.load(split_path, allow_pickle=False) as a:
                roles = ('validation', 'test') if config.validation_scope == 'both' or name == pool_name else ('test',)
                role_ids = [(role, a[role].copy()) for role in roles]
            if any(not len(ids) for _, ids in role_ids):
                raise ValueError('Benchmark requires nonempty validation/test sets')
            # Random holdouts touch most source blocks. Decode each block once,
            # then retain non-copying views for the separate evaluation roles.
            holdout = other.read(np.concatenate([ids for _, ids in role_ids]))['data']
            offset = 0
            for role, ids in role_ids:
                target = validation if role == 'validation' else tests
                target.append((name, input_view(other_inputs, ids), holdout[offset:offset + len(ids)]))
                if role=='test' and name==pool_name:
                    keys=[key for key in ('labels','levels','requested_levels','module_counts','mutation_descriptions')
                          if key in other.fields]
                    test_metadata[name]=dict(names=other_inputs.get('mutant_names',np.arange(other.count))[ids],
                        candidate_ids=other_inputs.get('candidate_ids',np.arange(other.count))[ids],
                        metadata=other.read(ids,keys) if keys else {})
                offset += len(ids)
    reference = None
    if real_reference is not None:
        with np.load(real_reference, allow_pickle=False) as archive:
            reference = {key: archive[key] for key in ('params', 'y_init', 'data', 'mutant_names')}
    monitoring={'synth':(view,truth) for name,view,truth in tests if name==pool_name}
    if reference is not None:
        monitoring['real']=(PoolInputs(np.arange(len(reference['data'])),reference['params'],reference['y_init']),reference['data'])
    metadata = None
    if spec.family == 'ours':
        keys = ('candidate_metadata_json', 'overrides_json') if 'overrides_json' in store.fields else ('candidate_metadata_json',)
        raw = store.read(np.arange(store.count), keys)
        metadata = [json.loads(str(row)) for row in raw['candidate_metadata_json']]
        if 'overrides_json' in raw:
            for row, override in zip(metadata, raw['overrides_json']):
                # Final effective input overrides, including ICs after rescue;
                # proposal/original_overrides may describe an earlier input.
                row['_effective_overrides'] = json.loads(str(override))
        metadata = tuple(metadata)
    selected = splits['initial'].tolist() if spec.mode == 'incremental' else []
    current_weights = None
    records, histories, plot_contexts = [], [], {}
    wandb_run = None
    if wandb_enabled:
        os.environ['WANDB_MODE'] = 'offline'
        import wandb
        (directory/'_state').mkdir(exist_ok=True)
        wandb_run = wandb.init(project='budding_sampling_benchmark', group=protocol['protocol_id'],
            name=f'{pool_name}/{spec.name}/seed{config.seed}', dir=str(directory/'_state'), mode='offline',
            config=dict(identity=identity, training=asdict(config)))
    began = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    log_event('START', 'offline sampling invocation', method=spec.name, pool=pool_name, path=directory)
    try:
        rounds = range(len(config.budgets)) if spec.mode == 'incremental' else (len(config.budgets) - 1,)
        if spec.mode in ('one_shot', 'weighted_one_shot'):
            path = directory / '_state/one_shot.json'
            if path.exists():
                saved = json.loads(path.read_text())
                selected = saved['selected_ids']
                current_weights = saved.get('weights')
                if not set(selected) <= set(splits['candidates']):
                    raise ValueError('Saved one-shot selection includes holdout/foreign rows')
            else:
                empty = PoolInputs(np.array([], dtype=int), inputs['params'][:0], inputs['y_init'][:0])
                pilot = QueriedTraining(empty, np.empty((0, 39, 500)))
                pilot_predictor = None
                if spec.mode == 'weighted_one_shot':
                    pilot_ids = splits['initial']
                    pilot = QueriedTraining(input_view(inputs, pilot_ids), oracle.reveal(pilot_ids)['data'])
                    pilot_path = directory/'_state/pilot/dataset.json'
                    store.snapshot(pilot_path,pilot_ids,provenance=dict(role='pilot',seed=config.seed))
                    pilot_model, pilot_norm, _ = fit_round(config, 0, pilot,
                        [(view, truth) for _, view, truth in validation], directory / '_state/pilot/primary',
                        child_seed(config.seed, 'model', 0, 0), device=device,
                        identity=dict(purpose='sampling_pilot', rows_sha256=fingerprint(pilot_ids.tolist()),
                                      protocol=protocol['fingerprint']), wandb_run=wandb_run,
                        dataset_path=pilot_path)
                    pilot_predictor = Predictor([pilot_model], pilot_norm, pilot.trajectories.shape[-1],
                                                device, config.prediction_batch_size)
                context = SamplingContext(input_view(inputs, splits['candidates']),
                    pilot, pilot_predictor, child_seed(config.seed, 'selection', 0), device,
                    config.sketch_dim, oracle.scoring_reader if spec.track == 'B' else None)
                result = select(spec.name, context, config.budgets[-1])
                selected = result.ids.tolist()
                current_weights = result.weights
                atomic_json(path, _json_safe(dict(selected_ids=selected, weights=current_weights,
                                                  diagnostics=result.diagnostics)))
                if pilot_predictor is not None:
                    pilot_model.cpu()
                del context, pilot, pilot_predictor
        for round_index in rounds:
            budget = config.budgets[round_index]
            expected_size = 1 < len(selected) <= budget if spec.mode == 'weighted_one_shot' else len(selected) == budget
            if not expected_size or len(set(selected)) != len(selected):
                raise RuntimeError('Cumulative dataset violates the declared unique-input budget')
            round_dir = directory / 'rounds' / f'round_{round_index:02d}'
            update_section(state_path,'progress', dict(phase='training', round=round_index, training_rows=len(selected)))
            with log_stage('offline round', round=round_index, method=spec.name, samples=len(selected)):
                data = oracle.reveal(selected)['data']
                training = QueriedTraining(input_view(inputs, selected, metadata), data)
                rows_fingerprint = fingerprint(selected)
                weights_fingerprint = fingerprint(None if current_weights is None else np.asarray(current_weights).tolist())
                store.snapshot(round_dir / 'dataset.json', selected,
                    weights=current_weights,
                    provenance=dict(protocol=protocol['fingerprint'], role='training', round=round_index, seed=config.seed))
                plot_contexts[round_index]=dict(model=config.model,loss=config.loss,pool=pool_name,
                    method=spec.name,training_rows=len(selected),seed=config.seed,warmup=config.warmup[round_index])
                seed = child_seed(config.seed, 'model', round_index, 0)
                with log_stage('primary cold-start training', round=round_index, seed=seed):
                    model, normalization, history = fit_round(config, round_index, training,
                        [(view, truth) for _, view, truth in validation], round_dir, seed,
                        device=device, identity=dict(rows_sha256=rows_fingerprint, weights_sha256=weights_fingerprint,
                                                     protocol=protocol['fingerprint']),
                        wandb_run=wandb_run, weights=current_weights,dataset_path=round_dir/'dataset.json',
                        state_dir=directory/'_state'/f'round_{round_index:02d}'/'primary',
                        plot_context=plot_contexts[round_index],monitoring=monitoring)
                histories.append((round_index, history))
                members = [model]
                predictor = Predictor([model], normalization, data.shape[-1], device,
                    config.prediction_batch_size, child_seed(config.seed, 'feature_sketch'))
                metrics = {}
                update_section(state_path,'progress',dict(phase='evaluation',round=round_index,training_rows=len(selected)))
                with log_stage('independent evaluation', round=round_index):
                    for name, view, truth in tests:
                        predictions=predictor.physical(view)
                        metrics[name+'_test']=summarize(predictions,truth)
                        if name==pool_name:
                            meta=test_metadata[name]
                            export_sample_metrics(predictions,truth,meta['names'],
                                round_dir/'eval/per_synth_test_metrics.csv',source=name,pool_rows=view.ids,
                                candidate_ids=meta['candidate_ids'],metadata=meta['metadata'])
                    if reference is not None:
                        view = PoolInputs(np.arange(len(reference['data'])), reference['params'], reference['y_init'])
                        predictions = predictor.physical(view)
                        metrics['real_mutants'] = save_real_evaluation(predictions, reference['data'],
                            reference['mutant_names'],round_dir/'eval',plots=plots,write_summary=False,
                            plot_workers=int(os.environ.get('SAMPLING_PLOT_WORKERS','2')),
                            context={k:v for k,v in plot_contexts[round_index].items() if k not in ('warmup','loss')})
                    save_global_metrics(round_dir/'eval',metrics,pool_name=pool_name,
                        round_index=round_index,training_rows=len(selected))
                records.append(dict(round=round_index, training_rows=len(selected), metrics=metrics))
                from .reports import save_progress_reports
                save_progress_reports(directory,records,histories,plot_contexts,round_dir)
                if wandb_run is not None:
                    wandb_run.log({f'evaluation/{domain}/macro_rel_l2': values['macro_trajectory_rel_l2']
                                   for domain, values in metrics.items()} |
                                  {f'evaluation/{domain}/sample_channel_rel_l2_physical':values['sample_channel_rel_l2']
                                   for domain,values in metrics.items()} |
                                  {'round': round_index, 'training_rows': len(selected)})
                if round_index == len(config.budgets) - 1:
                    continue
                update_section(state_path,'progress',dict(phase='selection',round=round_index,training_rows=len(selected)))
                remaining = np.setdiff1d(splits['candidates'], selected)
                after=directory/'_state'/f'round_{round_index:02d}'/'selection'
                result_path=round_dir/'selection.json'
                addition = config.budgets[round_index + 1] - budget
                if result_path.exists():
                    value = json.loads(result_path.read_text())
                    if value['previous_rows_sha256'] != rows_fingerprint:
                        raise ValueError('Saved acquisition does not follow this training dataset')
                    new_ids = value['selected_ids']
                    if len(new_ids) != addition or len(set(new_ids)) != addition or not set(new_ids) <= set(remaining):
                        raise ValueError('Saved acquisition violates the input/unique budget')
                else:
                    if spec.ensemble:
                        for member in range(1, config.ensemble_members):
                            member_seed = child_seed(config.seed, 'model', round_index, member)
                            with log_stage('independent ensemble training', member=member, round=round_index):
                                auxiliary, _, _ = fit_round(config, round_index, training,
                                    [(view, truth) for _, view, truth in validation],
                                    directory/'_state'/f'round_{round_index:02d}'/f'ensemble_member_{member}', member_seed, device=device, member=member,
                                    identity=dict(rows_sha256=rows_fingerprint, protocol=protocol['fingerprint']),
                                    wandb_run=wandb_run,dataset_path=round_dir/'dataset.json')
                            members.append(auxiliary)
                    predictor.members = members
                    context = SamplingContext(input_view(inputs, remaining, metadata), training, predictor,
                        child_seed(config.seed, 'selection', round_index), device, config.sketch_dim,
                        oracle.scoring_reader if spec.track == 'B' else None, config.prediction_batch_size)
                    scoring_queued_at=time.monotonic()
                    from ..ours.mechanistic import METHOD_NAMES as mechanism_methods
                    scoring_resource = 'cpu_scoring' if spec.name in mechanism_methods else 'scoring'
                    with resource_slot(scoring_resource, 'candidate scoring'), log_stage(
                                   'selection without unqueried truth' if spec.track == 'A' else 'labelled data selection',
                                   method=spec.name, pool=len(remaining), addition=addition):
                        selected_at = time.monotonic()
                        result = select(spec.name, context, addition)
                    new_ids = result.ids.tolist()
                    atomic_json(result_path, _json_safe(dict(selected_ids=new_ids,
                        previous_rows_sha256=rows_fingerprint, diagnostics=result.diagnostics,
                        selection_seconds=time.monotonic() - selected_at,
                        scoring_wait_seconds=selected_at-scoring_queued_at,
                        seed=context.seed, track=spec.track,source_pool=protocol['pools'][pool_name])))
                    artifacts = getattr(result, 'artifacts', {})
                    if artifacts:
                        after.mkdir(parents=True,exist_ok=True)
                        import joblib
                        joblib.dump(artifacts['classifier'], after / 'pattern_classifier.joblib')
                        np.savez_compressed(after / 'selected_predictions.npz', **artifacts['predictions'])
                # Truth is revealed strictly after the complete selection decision.
                oracle.reveal(new_ids)
                selected.extend(new_ids)
                if wandb_run is not None:
                    wandb_run.log({'acquisition/round': round_index, 'acquisition/added': len(new_ids),
                                   'acquisition/cumulative_training_rows': len(selected),
                                   'acquisition/unique_labels_accessed': len(oracle.revealed)})
                for member in members:
                    member.cpu()
                members.clear()
                if 'context' in locals():
                    del context
                del data, training, model, predictor
                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()
        update_section(state_path,'completion', dict(status='complete',
            protocol_fingerprint=protocol['fingerprint'], method=spec.name, track=spec.track,
            unique_training_inputs=len(selected), unique_labels_accessed=len(oracle.revealed),
            pilot_training_inputs=len(splits['initial']) if spec.mode=='weighted_one_shot' else 0,
            historical_training_inputs=len(set(selected)|set(map(int,splits['initial'])))
                if spec.mode=='weighted_one_shot' else len(selected),
            training_draw_slots=config.budgets[-1] if spec.mode == 'weighted_one_shot' else len(selected),
            new_ode_calls=0, invocation_seconds=time.monotonic() - began,
            invocation_started_at=started_at, completed_at=datetime.now(timezone.utc).isoformat(),
            training_costs=training_costs(directory),
            selection_seconds=sum(json.loads(path.read_text()).get('selection_seconds', 0.)
                                  for path in directory.glob('rounds/round_*/selection.json')),
            shared_holdout_rows={role: {name: len(view.ids) for name, view, _ in values}
                                 for role, values in (('validation', validation), ('test', tests))},
            diagnostic_only=diagnostic_only, real_evaluation=reference is not None))
        update_section(state_path,'progress',dict(phase='complete',round=records[-1]['round']))
        log_event('COMPLETE', 'offline sampling experiment', path=directory)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return directory
