"""Stage-gated input-order controls, with synthetic-only method selection.

Screen three controls on seed42, lock ONE quota rule using synthetic feedback,
confirm only that rule vs random on 123/1024, and open the independent audit
once. Real/test are decoded only if the preregistered synthetic audit passes.
This is a falsification study, not an already validated research contribution.
"""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from budding_yeast_v2.data.dataset_archive import TrajectoryStore, atomic_json, file_sha256
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.data.systems.budding.specification import SPEC
from budding_yeast_v2.utils.evaluation.checkpoints import Predictor, load_sampling_checkpoint
from budding_yeast_v2.utils.evaluation.reporting import save_global_metrics
from budding_yeast_v2.utils.experiments.logging import log_event, log_stage
from budding_yeast_v2.utils.sampling.layout import RESULTS, experiment_directory
from budding_yeast_v2.utils.sampling.contracts import QueriedTraining
from budding_yeast_v2.utils.sampling.ours.intervention_calibration import (
    METHODS, reserve_roles, select_rows, balanced_metrics, decision)
from .coverage_study import definition, study_config, read, subset_inputs
from .diagnostics import context_key, evaluate_reference_cohort
from .protocol import child_seed
from .rounds import fit_round
from .runner import source_inventory


def work_path(root):
    return Path(root)/'studies/intervention_order_calibration_v1'


def prepare(root):
    protocol, source = definition(root)
    cfg = study_config(protocol)
    work = work_path(root); work.mkdir(parents=True, exist_ok=True)
    if (work/'prepared.json').exists():
        saved = json.loads((work/'study.json').read_text())
        if saved['sources'] != source_inventory():
            raise ValueError('Preparation source identity changed')
        return work
    with np.load(root/'mutation_split.npz') as split:
        candidates, initial, validation, test = [split[key].copy()
            for key in ('candidates', 'initial', 'validation', 'test')]
    meta = read(root, source, np.r_[candidates, validation],
                ('params', 'candidate_metadata_json', 'module_counts', 'levels'))
    n = len(candidates)
    levels = meta['module_counts'][:n].astype(int)
    if not np.array_equal(meta['module_counts'], meta['levels']):
        raise ValueError('Stored actual intervention levels disagree')
    mdt = SPEC.param_names.index('mdt')
    metadata = [json.loads(str(v)) for v in meta['candidate_metadata_json']]
    for row, count in zip(metadata, meta['module_counts']):
        modules = {gene for gene, _ in row.get('pairs', row.get('genes', ())) if gene != 'MDT'}
        if len(modules) != count:
            raise ValueError('Actual module count must match canonical mutation metadata')
    keys = [context_key(m, p[mdt]) for m, p in zip(metadata, meta['params'])]
    roles = reserve_roles(candidates, levels, keys[:n], initial,
                          seed=child_seed(cfg.seed, 'order_calibration_roles'))
    reserved = set(roles['feedback_contexts']) | set(roles['audit_contexts'])
    roles['checkpoint_validation'] = [int(r) for r, key in zip(validation, keys[n:]) if key not in reserved]
    if len(roles['checkpoint_validation']) < 4000:
        raise ValueError('Too few independent source validation inputs remain')
    # Formal test IDs are recorded but their true trajectories are not read.
    roles['test'] = test.tolist()
    by_row = dict(zip(map(int, candidates), keys[:n]))
    level_by_row = dict(zip(map(int, candidates), map(int, levels)))
    for role in ('feedback', 'audit'):
        roles[role+'_levels'] = [level_by_row[r] for r in roles[role]]
        roles[role+'_row_contexts'] = [by_row[r] for r in roles[role]]
    allowed = np.asarray(roles['training'])
    allowed_levels = np.array([level_by_row[int(r)] for r in allowed])
    (work/'selections').mkdir(exist_ok=True)
    selections = {}
    for method in METHODS:
        rows, details = select_rows(allowed, allowed_levels, initial, 10000, method,
                                    seed=child_seed(cfg.seed, 'order_calibration_selection'))
        if set(rows) & set(roles['feedback']+roles['audit']):
            raise ValueError('Research holdout overlap')
        path = work/'selections'/f'{method}.json'
        source.snapshot(path, rows, provenance=dict(workflow='order_calibration_diagnostic_v1', diagnostics=details))
        selections[method] = dict(path=str(path), sha256=file_sha256(path), details=details)
    settings = dict(workflow='order_calibration_diagnostic_v1', training=asdict(cfg), seed=cfg.seed,
        protocol=str(root), protocol_sha256=file_sha256(root/'protocol.json'), pool_sha256=source.digest,
        sources=source_inventory(), roles=roles, selections=selections,
        model_seed=child_seed(cfg.seed, 'order_calibration_model'),
        primary='equal_L_mean_Relative_in_common_initial_2000_normalized_coordinates',
        secondary='equal_L_physical_MAE_and_physical_Relative; no metric cap or epsilon changes',
        checkpoint_selection='source_synthetic_normalized_validation_best; research contexts also excluded from this loader',
        max_final_updates_per_model=47005, parameter_distribution_and_ODE_changed=False,
        gate=dict(screen=dict(seed=42, min_gain=.05, physical_slack=.05),
            confirm=dict(seeds=[42, 123, 1024], min_gain=.05, physical_slack=0., minimum_positive_seeds=2),
            audit=dict(min_gain=.05, physical_slack=0., minimum_positive_seeds=2, max_each_L_ratio=1.10)),
        real_or_test_labels_used_for_method_choice=False,
        research_status='fixed quota controls are not a claimed novel sampling method')
    atomic_json(work/'study.json', settings)
    atomic_json(work/'prepared.json', dict(status='complete', study_sha256=file_sha256(work/'study.json')))
    return work


def settings_for(root):
    protocol, source = definition(root)
    work = work_path(root)
    settings = json.loads((work/'study.json').read_text())
    if (settings['sources'] != source_inventory() or settings['protocol_sha256'] != file_sha256(root/'protocol.json')
            or settings['pool_sha256'] != source.digest):
        raise ValueError('Frozen order-calibration identity changed')
    return protocol, source, work, settings


def run_path(root, method):
    protocol = json.loads((root/'protocol.json').read_text())
    return experiment_directory(RESULTS, study_config(protocol), 'mutation', method)


def train_feedback(root, method, *, device='cpu', wandb_run=None):
    if method not in METHODS:
        raise ValueError('Unknown diagnostic control')
    protocol, source, work, settings = settings_for(root)
    cfg = study_config(protocol)
    directory = run_path(root, method); started = time.monotonic()
    descriptor = settings['selections'][method]
    snapshot = Path(descriptor['path'])
    if file_sha256(snapshot) != descriptor['sha256']:
        raise ValueError('Input-only selection changed')
    rows = json.loads(snapshot.read_text())['rows']
    config = dict(workflow=settings['workflow'], training=asdict(cfg), method=method,
        study=str(work/'study.json'), study_sha256=file_sha256(work/'study.json'),
        selection_sha256=file_sha256(snapshot), sources=settings['sources'],
        checkpoint_selection=settings['checkpoint_selection'], model_seed=settings['model_seed'])
    if (directory/'config.json').exists() and json.loads((directory/'config.json').read_text()) != config:
        raise ValueError('Existing training configuration differs')
    atomic_json(directory/'config.json', config)
    if (directory/'completion.json').exists():
        return directory
    arrays = read(root, source, rows, ('params', 'y_init', 'data'))
    val_ids = settings['roles']['checkpoint_validation']
    validation = read(root, source, val_ids, ('params', 'y_init', 'data'))
    (directory/'data').mkdir(exist_ok=True)
    data_path = directory/'data/dataset.json'
    source.snapshot(data_path, rows, provenance=dict(shared_selection=str(snapshot), source_trajectories_copied=False))
    with log_stage('order-calibration cold training', method=method, seed=cfg.seed, epochs=595):
        model, norm, history = fit_round(cfg, 0,
            QueriedTraining(subset_inputs(arrays, rows, ()), arrays['data']),
            [(subset_inputs(validation, val_ids, ()), validation['data'])], directory/'training',
            settings['model_seed'], device=device, dataset_path=data_path, state_dir=directory/'_state/primary',
            identity=config, wandb_run=wandb_run, plot_context=dict(training_rows=10000))
    if history[-1]['optimizer_steps'] != 47005:
        raise RuntimeError('Update budget changed')
    common = NormalizationStats.fit(arrays['params'][:2000], arrays['data'][:2000])
    predictor = Predictor([model], norm, arrays['data'].shape[-1], device, cfg.prediction_batch_size)
    del arrays, validation
    gc.collect()
    risk = evaluate_role(source, settings, predictor, common, 'feedback', directory/'eval/feedback')
    if source_inventory() != settings['sources']:
        raise RuntimeError('Sources changed during training')
    atomic_json(directory/'completion.json', dict(status='complete', workflow=settings['workflow'],
        method=method, seed=cfg.seed, run=str(directory), training_updates=47005,
        seconds=time.monotonic()-started, feedback=risk, selection=descriptor['details'],
        checkpoint_sha256=file_sha256(directory/'training/model.ckpt'), sources_unchanged=True,
        real_labels_read=False, test_labels_read=False, audit_labels_read=False))
    return directory


def evaluate_role(source, settings, predictor, common, role, destination):
    ids = settings['roles'][role]
    cohort = read(None, source, ids, ('params', 'y_init', 'data'))
    value = evaluate_reference_cohort(predictor, subset_inputs(cohort, ids, ()), cohort['data'], common,
        destination, variable_names=SPEC.variable_names, common_reference='common_initial_2000_train_stats',
        sample_ids=ids, return_per_sample=True)
    result = balanced_metrics(value['per_sample'], settings['roles'][role+'_levels'],
                              settings['roles'][role+'_row_contexts'])
    atomic_json(Path(destination)/'balanced_risk.json', result)
    return result


def results_for(shared, seed, method):
    root = shared/f'seed_{seed}'
    directory = run_path(root, method)
    result = json.loads((directory/'completion.json').read_text())
    if (result['status'] != 'complete' or not result['sources_unchanged']
            or result['training_updates'] != 47005 or result['real_labels_read'] or result['test_labels_read']):
        raise ValueError('Need completed, clean synthetic-feedback diagnostic')
    if file_sha256(directory/'training/model.ckpt') != result['checkpoint_sha256']:
        raise ValueError('Locked checkpoint changed')
    return result


def choose(shared, decision_path):
    baseline = results_for(shared, 42, METHODS[0])
    alternatives = {method: results_for(shared, 42, method) for method in METHODS[1:]}
    gates = {m: decision([r['feedback']], [baseline['feedback']], min_gain=.05, slack=.05)
             for m, r in alternatives.items()}
    admissible = [m for m in alternatives if gates[m]['passed']]
    method = min(admissible or list(alternatives),
                 key=lambda m: (alternatives[m]['feedback']['balanced']['normalized_relative'], m))
    gate = gates[method]
    record = dict(status='locked', winner=method, promising=gate['passed'], gate=gate,
        selection_uses='seed42 synthetic feedback only; no audit, formal test or real metrics',
        alternatives={m: dict(metrics=r['feedback']['balanced'], gate=gates[m]) for m, r in alternatives.items()},
        random=baseline['feedback']['balanced'], checkpoint_sha256={m: r['checkpoint_sha256']
            for m, r in dict(alternatives, **{METHODS[0]: baseline}).items()})
    if decision_path.exists() and json.loads(decision_path.read_text()) != record:
        raise ValueError('Method choice is immutable; cannot reselect after opening audit')
    atomic_json(decision_path, record)
    return record


def conclude(shared, decision_path, output, *, device='cpu'):
    output.mkdir(parents=True, exist_ok=True)
    selected = json.loads(decision_path.read_text())
    report = dict(study='intervention_order_calibration_v1', selected=selected, diagnostic_only=True,
        fixed_quota_not_a_claimed_innovation=True, real_or_test_used_for_selection=False)
    if not selected['promising']:
        report.update(status='screen_negative', independent_audit_opened=False,
            message='No quota rule passed seed42 synthetic feedback; stop without confirmation/real testing')
        atomic_json(output/'report.json', report)
        return report
    winner = selected['winner']
    candidates = [results_for(shared, s, winner) for s in (42, 123, 1024)]
    controls = [results_for(shared, s, METHODS[0]) for s in (42, 123, 1024)]
    confirmation = decision([r['feedback'] for r in candidates], [r['feedback'] for r in controls], slack=0.)
    report['confirmation'] = confirmation
    if not confirmation['passed']:
        report.update(status='confirmation_negative', independent_audit_opened=False,
            message='Preselected quota failed confirmation; audit/test/real remain unopened')
        atomic_json(output/'report.json', report)
        return report
    # Lock all six checkpoints before opening ANY independent audit trajectory.
    atomic_json(output/'audit_lock.json', dict(winner=winner,
        checkpoints=[dict(method=r['method'], seed=r['seed'], sha256=r['checkpoint_sha256']) for r in candidates+controls]))
    risks = {winner: [], METHODS[0]: []}
    for seed in (42, 123, 1024):
        root = shared/f'seed_{seed}'
        protocol, source, work, settings = settings_for(root)
        common_arrays = read(root, source, settings['roles']['initial'], ('params', 'data'))
        common = NormalizationStats.fit(common_arrays['params'], common_arrays['data'])
        for method in (winner, METHODS[0]):
            directory = run_path(root, method)
            model, norm, _ = load_sampling_checkpoint(directory/'training/model.ckpt', device)
            predictor = Predictor([model], norm, 500, device, 128)
            risks[method].append(evaluate_role(source, settings, predictor, common, 'audit', directory/'eval/audit'))
            del predictor, model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    report.update(independent_audit_opened=True, audit_risks=risks,
        audit=decision(risks[winner], risks[METHODS[0]], slack=0.))
    report['status'] = 'synthetic_audit_positive' if report['audit']['passed'] else 'synthetic_audit_negative'
    atomic_json(output/'report.json', report)
    return report


def terminal_evaluate(shared, report, output, *, device='cpu'):
    """Post-lock terminal evaluation; these scores never change the quota rule."""
    if report['status'] != 'synthetic_audit_positive' or not report['audit']['passed']:
        raise ValueError('Real/test evaluation requires the predeclared independent synthetic gate')
    winner = report['selected']['winner']
    terminal = {}
    for seed in (42, 123, 1024):
        root = shared/f'seed_{seed}'
        protocol, source, work, settings = settings_for(root)
        initial = read(root, source, settings['roles']['initial'], ('params', 'data'))
        common = NormalizationStats.fit(initial['params'], initial['data'])
        test_ids = settings['roles']['test']
        test = read(root, source, test_ids, ('params', 'y_init', 'data'))
        ref = protocol['real_reference']
        if file_sha256(ref['path']) != ref['sha256']:
            raise ValueError('Frozen real reference changed')
        with np.load(ref['path'], allow_pickle=False) as archive:
            real = {k: archive[k] for k in ('params', 'y_init', 'data', 'mutant_names')}
        for method in (winner, METHODS[0]):
            result = results_for(shared, seed, method)
            directory = run_path(root, method)
            model, norm, _ = load_sampling_checkpoint(directory/'training/model.ckpt', device)
            predictor = Predictor([model], norm, real['data'].shape[-1], device, 128)
            metrics = {}
            metrics['synth_test'] = evaluate_reference_cohort(predictor,
                subset_inputs(test, test_ids, ()), test['data'], common, directory/'eval/synth_test',
                variable_names=SPEC.variable_names, sample_ids=test_ids,
                common_reference='common_initial_2000_train_stats')
            metrics['real_mutants'] = evaluate_reference_cohort(predictor,
                subset_inputs(real, np.arange(len(real['data'])), ()), real['data'], common, directory/'eval',
                variable_names=SPEC.variable_names, sample_names=real['mutant_names'], real_plots=True,
                common_reference='common_initial_2000_train_stats')
            save_global_metrics(directory/'eval', dict(real_mutants=metrics['real_mutants']['physical'],
                mutation_test=metrics['synth_test']['physical']), pool_name='mutation', round_index=0, training_rows=10000)
            value = dict(metrics=metrics, selected_without_real_or_test=True,
                checkpoint_sha256=result['checkpoint_sha256'], quota_unchanged_after_real=True)
            terminal[f'{method}/seed_{seed}'] = value
            atomic_json(directory/'terminal_evaluation.json', value)
            del model, predictor
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    report['terminal_results'] = terminal
    report['next_research_step'] = ('Fixed quota controls are not a novel method. If terminal evidence is consistent, '
        'test an adaptive cross-order learning-gain allocator using synthetic feedback only; do not tune quotas on real scores.')
    atomic_json(output/'report.json', report)


def write_report(shared, output):
    """Readable result ledger including negative stops, not just success tables."""
    import csv
    report = json.loads((output/'report.json').read_text())
    names = dict(zip(METHODS, ('池比例随机', '容量约束阶数均衡', '低阶优先')))
    lines = ['# 干预阶数分布校准检验', '', f"状态：{report['status']}", '',
        '固定quota是诊断对照，不声称新算法。最终每模型10000条、595epoch/47005步，'
        '相同FNO/训练损失/归一化流程及初始2000条；真实/正式测试指标不参与quota选择。', '',
        '| seed | 方法 | 最终L1/L2/L3 | 反馈均衡Relative（共同归一化） | 反馈均衡物理MAE | 反馈均衡物理Relative |',
        '|---|---|---|---:|---:|---:|']
    rows = []
    for seed in (42, 123, 1024):
        root = shared/f'seed_{seed}'
        for method in METHODS:
            directory = run_path(root, method)
            path = directory/'completion.json'
            if not path.exists():
                continue
            value = results_for(shared, seed, method)
            risk = value['feedback']['balanced']
            rows.append(dict(seed=seed, method=method, run=str(directory),
                final_counts=str(value['selection']['final_counts']), **risk))
            counts='/'.join(map(str,value['selection']['final_counts']))
            lines.append(f"| {seed} | {names[method]} | {counts} | {risk['normalized_relative']:.6f} | "
                         f"{risk['physical_mae']:.6f} | {risk['physical_relative']:.6f} |")
    if rows:
        with (output/'feedback_metrics.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines += ['', '## 预先锁定的决定', '',
        '```json', json.dumps({k:v for k,v in report.items() if k in ('selected','confirmation','audit','message')},
                             ensure_ascii=False,indent=2), '```', '',
        '筛选失败不开放独立审计/real/正式测试；确认失败也不重新选择另一配额。审计只在确认通过后开放一次。',
        '通过审计仍不等于证明论文创新或击败强baseline。后续必须与强baseline同协议比较并检验跨系统迁移。']
    if 'terminal_results' in report:
        lines += ['', '## 锁定后的Real结果', '', '| 方法/seed | MAE | MSE | Relative | Corr |', '|---|---:|---:|---:|---:|']
        for key, value in report['terminal_results'].items():
            metric=value['metrics']['real_mutants']['physical']['all_variables']
            lines.append(f"| {key} | {metric['MAE']:.6f} | {metric['MSE']:.6f} | {metric['Relative L2']:.6f} | {metric['Correlation']:.6f} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')
