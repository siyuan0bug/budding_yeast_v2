"""Check executed AL budgets, checkpoint steps, data references and final outputs.

Reads manifests, small NPZ metadata and memory-mapped CPU checkpoints. Does not
materialize trajectories or train a model. Reports incomplete runs explicitly.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from budding_yeast_v2.data.dataset_archive import open_dataset
from budding_yeast_v2.utils.active_learning.registry import strategy_spec


def read(path):
    return json.loads(path.read_text())


def stage_seconds(log,name):
    return sum(map(float,re.findall(r'\[END\] '+re.escape(name)+r' \|[^\r\n]*elapsed_s=([\d.]+)',log)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    manifest=read(args.manifest);results=[];reference_splits=None;starts=[];ends=[]
    for task in manifest['tasks']:
        run=Path(task['experiment_dir']);log=(run/'train.log').read_text(errors='replace')
        config=read(run/'config.json')['args'];issues=[];checks=[]
        expected=dict(model='fno',al_strategy=task['strategy'],width=192,n_blocks=4,modes=24,batch_size=512,lr=.001,min_lr=1e-5,
            al_round_epochs=100,al_final_epochs=200,al_round_warmup=5,al_final_warmup=10,
            num_al_rounds=4,al_round_seed_policy='fixed',seed=42,norm_type='log1p',num_workers=0)
        for key,value in expected.items():
            if config.get(key)!=value:issues.append(f'{key}={config.get(key)!r}, expected {value!r}')
        failures=[line for line in log.splitlines() if any(s in line for s in (
            'Traceback (most recent call last)','ValueError:','RuntimeError:','[FAIL]','[RETRY]',
            'CUDA out of memory','No ODE worker completed'))]
        exits=re.findall(r'END GPU=.*?exit=(\d+)',log);exit_code=int(exits[-1]) if exits else None
        if failures:issues.extend(failures)
        if exit_code!=0:issues.append(f'Process exit={exit_code}')
        duration=re.findall(r'\[END\] experiment \| elapsed_s=([\d.]+)',log)
        start=re.search(r'^\[([^\]]+)\] START GPU=',log,re.M)
        end=re.findall(r'^\[([^\]]+)\] END GPU=',log,re.M)
        if start:starts.append(datetime.fromisoformat(start[1]))
        if end:ends.append(datetime.fromisoformat(end[-1]))
        rounds=read(run/'summary/rounds_summary.json') if (run/'summary/rounds_summary.json').exists() else []
        candidate_ids=[];initial_ids=set();total_epochs=0;auxiliary_epochs=0;training_id_hashes={}
        for record in rounds:
            r=record['round'];folder=run/'rounds'/f'round_{r:02d}'
            size=[10000,20000,40000,70000][r];epochs=200 if r==3 else 100
            if (record['train_samples'],record['epochs'])!=(size,epochs):issues.append(f'round {r} budget mismatch')
            with (folder/'training_history.csv').open() as f:history=list(csv.DictReader(f))
            observed=[int(row['epochs']) for row in history];total_epochs+=len(history)
            if observed!=list(range(1,epochs+1)):issues.append(f'round {r} epoch sequence incomplete')
            if not all(math.isfinite(float(row['train_loss'])) for row in history):
                issues.append(f'round {r} nonfinite stored training loss')
            checkpoint=torch.load(folder/'model.ckpt',map_location='cpu',weights_only=False,mmap=True)
            # The production DataLoader has drop_last=True and no accumulation.
            expected_steps=(size//config['batch_size'])*epochs
            if checkpoint['global_step']!=expected_steps:issues.append(f'round {r} optimizer steps mismatch')
            contract=checkpoint['preprocessing_contract'];del checkpoint
            if contract['fit_scope']!='train_synthetic' or contract['norm_type']!='log1p':
                issues.append(f'round {r} preprocessing scope/type mismatch')
            if len(contract['split']['train'])!=size:issues.append(f'round {r} preprocessing fit count mismatch')
            for key in ('mean_y','std_y','p_mean','p_std'):
                if not torch.isfinite(contract[key]).all():issues.append(f'round {r} nonfinite {key}')
            if not (contract['std_y']>0).all() or not (contract['p_std']>0).all():
                issues.append(f'round {r} nonpositive normalization scale')
            with open_dataset(folder/'dataset.json') as archive:
                split={k:archive[k] for k in ('split_train','split_val','split_test','split_real')}
                lengths={k:len(v) for k,v in split.items()}
                if list(lengths.values())!=[size,1250,1250,126]:issues.append(f'round {r} dataset counts mismatch')
                heldout={k:archive['mutant_names'][split[k]].tolist() for k in ('split_val','split_test','split_real')}
                if reference_splits is None:reference_splits=heldout
                if heldout!=reference_splits:issues.append(f'round {r} fixed holdout identity/order changed')
                flat=np.concatenate(list(split.values()))
                if len(np.unique(flat))!=len(flat):issues.append(f'round {r} overlapping splits')
                counts=archive['module_counts'][split['split_train']]
                if not np.isin(counts,[1,2,3]).all():issues.append(f'round {r} module counts outside 1-3')
                ids=archive['candidate_ids']
                if r==0:initial_ids=set(map(int,ids[ids>=0]))
                training_ids=ids[split['split_train']]
                if len(np.unique(training_ids))!=size:issues.append(f'round {r} repeated training candidate IDs')
                training_id_hashes[r]=hashlib.sha256(training_ids.astype('<i8').tobytes()).hexdigest()
                if np.dtype(archive.manifest['fields']['data']['dtype'])!=np.float32:
                    issues.append(f'round {r} trajectory dtype mismatch')
            if r<3 and strategy_spec(task['strategy']).requires_ensemble:
                auxiliary=folder/'ensemble'/f'round_{r}_ensemble_member_1.ckpt'
                if not auxiliary.exists():issues.append(f'round {r} missing auxiliary checkpoint')
                else:
                    checkpoint=torch.load(auxiliary,map_location='cpu',weights_only=False,mmap=True)
                    if checkpoint['global_step']!=expected_steps:issues.append(f'round {r} auxiliary steps mismatch')
                    auxiliary_epochs+=epochs
                    del checkpoint
            metrics=read(folder/'eval/metrics_denorm.json')
            if metrics['real_mutants']['count']!=126 or metrics['synth_test']['count']!=1250:
                issues.append(f'round {r} evaluation count mismatch')
            if len(metrics['per_mutant'])!=126:issues.append(f'round {r} per-mutant rows mismatch')
            for key in ['MAE','MSE','Relative L2','Correlation']:
                if not math.isfinite(metrics['real_mutants']['all_variables'][key]):issues.append(f'round {r} nonfinite metric')
            for metric,suffix in [('MAE','mae'),('MSE','mse'),('Relative L2','rel_l2'),('Correlation','corr')]:
                if record['real_mutants_'+suffix]!=metrics['real_mutants']['all_variables'][metric]:
                    issues.append(f'round {r} summary metric mismatch: {metric}')
            plots=list((folder/'eval').rglob('*.png'))
            if len(plots)!=126:issues.append(f'round {r} real plots={len(plots)}')
            if not (folder/'loss_curves.png').is_file():issues.append(f'round {r} missing loss plot')
            checks.append(dict(round=r,train_samples=size,epochs=len(history),optimizer_steps=expected_steps,
                fit_scope=contract.get('fit_scope'),real_count=126,plots=len(plots)))
        for r,expected_count in enumerate([10000,20000,30000]):
            path=run/'acquisitions'/f'after_round_{r:02d}'/'report.json'
            if not path.exists():
                issues.append(f'acquisition {r} missing report')
                continue
            report=read(path);ids=np.asarray(report['candidate_ids'],dtype=np.int64)
            if report['accepted']!=expected_count or len(ids)!=expected_count:issues.append(f'acquisition {r} short budget')
            candidate_ids.extend(map(int,ids))
            if task['strategy'].startswith('ours_pattern_') or task['strategy']=='ours_sensitivity':
                directory=path.parent;predictions=directory/'mechanism_predictions.npz'
                if not predictions.exists():issues.append(f'acquisition {r} missing pre-query predictions')
                else:
                    with np.load(predictions,allow_pickle=False) as saved:
                        attempted=saved['candidate_ids']
                        if len(np.unique(attempted))!=len(attempted):issues.append(f'acquisition {r} repeated prediction IDs')
                        if len(np.setdiff1d(ids,attempted)):issues.append(f'acquisition {r} accepted ID lacks pre-query record')
                        if task['strategy'].startswith('ours_pattern_'):
                            probability=saved['pattern_probability']
                            if (probability.shape!=(len(attempted),6) or not np.isfinite(probability).all()
                                or (probability<0).any() or not np.allclose(probability.sum(1),1.,atol=1e-5)):
                                issues.append(f'acquisition {r} invalid Pattern probabilities')
                            if not np.isfinite(saved['physical_descriptor']).all():issues.append(f'acquisition {r} nonfinite descriptors')
                        elif not np.isfinite(saved['score']).all():issues.append(f'acquisition {r} nonfinite sensitivity scores')
                if task['strategy'].startswith('ours_pattern_'):
                    metadata=directory/'pattern_classifier.json'
                    if not metadata.exists() or not (directory/'pattern_classifier.joblib').exists():
                        issues.append(f'acquisition {r} missing classifier')
                    else:
                        classifier=read(metadata)
                        if classifier['training_rows']!=[10000,20000,40000][r]:issues.append(f'acquisition {r} classifier train size mismatch')
                        if classifier['training_ids_sha256']!=training_id_hashes.get(r):issues.append(f'acquisition {r} classifier train identities mismatch')
        if len(set(candidate_ids))!=len(candidate_ids):issues.append('Repeated acquired candidate IDs')
        if set(candidate_ids)&initial_ids:issues.append('Acquisition overlaps initial training/holdout candidates')
        if len(rounds)==4 and len(candidate_ids)!=60000:issues.append('Total acquired budget differs from 60000')
        finished=exit_code==0 and len(rounds)==4 and not issues
        if exit_code==0 and len(rounds)!=4:issues.append('Successful exit with fewer than four rounds')
        final=rounds[-1] if len(rounds)==4 else {}
        result=dict(strategy=task['strategy'],finished=finished,exit_code=exit_code,rounds=len(rounds),
            primary_epochs=total_epochs,auxiliary_epochs=auxiliary_epochs,
            acquired=len(candidate_ids),elapsed_seconds=float(duration[-1]) if duration else None,
            wall_seconds=(datetime.fromisoformat(end[-1])-datetime.fromisoformat(start[1])).total_seconds() if start and end else None,
            training_seconds=stage_seconds(log,'main model training'),ensemble_seconds=stage_seconds(log,'acquisition ensemble training'),
            selection_seconds=stage_seconds(log,'candidate scoring and selection'),ode_seconds=stage_seconds(log,'ODE process pool'),
            round_checks=checks,issues=issues,run=str(run),
            **{k:final.get(k) for k in ('real_mutants_mae','real_mutants_mse','real_mutants_rel_l2','real_mutants_corr')})
        results.append(result)
        print(task['strategy'],f'rounds={len(rounds)} epochs={total_epochs} acquired={len(candidate_ids)}',
              'OK' if finished else 'ISSUES: '+str(issues[-2:]),flush=True)
    report=dict(checked=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),tasks=results,
        batch_wall_seconds=(max(ends)-min(starts)).total_seconds(),
        completed=sum(r['finished'] for r in results),incomplete=sum(not r['finished'] for r in results))
    (args.output/'integrity.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    with (args.output/'completion.csv').open('w') as f:
        fields=[k for k in results[0] if k not in ('round_checks','issues')]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in results)


if __name__=='__main__':main()
