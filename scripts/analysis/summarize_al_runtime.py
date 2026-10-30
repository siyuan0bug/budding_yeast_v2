"""Summarize archived AL wall-clock windows without starting or changing runs."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

TZ=ZoneInfo('Asia/Shanghai')


def log_time(line):
    return datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=TZ)


def hms(seconds):
    if seconds is None:return ''
    n=round(seconds);return f'{n//3600:02d}:{n//60%60:02d}:{n%60:02d}'


def summarize(root):
    rows=[]
    supervisor=root/'overnight_2026-10-23/status.json'
    interruption=None
    if supervisor.exists() and json.loads(supervisor.read_text()).get('state')=='interrupted':
        interruption=datetime.fromtimestamp(supervisor.stat().st_mtime,TZ)
    completed_batch=set()
    for path in (root/'batch_logs').rglob('*.tsv'):
        for line in path.read_text().splitlines()[1:]:
            columns=line.replace('\\t','\t').split('\t')
            if len(columns)>=5 and columns[3]=='0':completed_batch.add(str(Path(columns[4]).resolve()))
    for config in sorted(root.rglob('config.json')):
        run=config.parent;metadata=json.loads(config.read_text());args=metadata.get('args',metadata)
        summary=run/'summary/rounds_summary.json'
        if not summary.exists():continue
        rounds=json.loads(summary.read_text());log=run/'train.log';content=log.read_text(errors='replace')
        configured=datetime.strptime(metadata['timestamp'],'%Y-%m-%d %H:%M:%S').replace(tzinfo=TZ)
        try:birth=int(subprocess.check_output(['stat','-c','%W','--',str(log)],text=True).strip())
        except (OSError,ValueError,subprocess.SubprocessError):birth=0
        start=datetime.fromtimestamp(birth,TZ) if birth>0 else configured
        time_source='train.log filesystem birth (shell launch redirection)' if birth>0 else 'config timestamp (after dataset setup)'
        successful=[];debug_sources=[]
        for debug in (run/'wandb').glob('offline-run-*/logs/debug.log'):
            successful.extend(log_time(line) for line in debug.read_text(errors='replace').splitlines() if '] got exitcode: 0' in line)
            debug_sources.append(str(debug))
        end=max(successful) if successful else None
        planned=int(args.get('num_al_rounds',4));complete=(len(rounds)==planned and end is not None and str(run.resolve()) in completed_batch)
        round_markers=re.findall(r'AL 轮 (\d+)/(\d+)',content);epochs=re.findall(r'\[Epoch\s+(\d+)/(\d+)\]',content)
        row=dict(model=args['model'],strategy=args['al_strategy'].removeprefix('al4pde_'),seed=args['seed'],completed=complete,
            completed_rounds=len(rounds),planned_rounds=planned,last_training_round=int(round_markers[-1][0])-1 if round_markers else None,
            last_logged_epoch=int(epochs[-1][0]) if epochs else None,last_logged_epoch_target=int(epochs[-1][1]) if epochs else None,
            start_time=start.isoformat(),successful_finish_time=end.isoformat() if end else None,
            elapsed_seconds=(end-start).total_seconds() if complete else None,
            recorded_training_seconds=(end-configured).total_seconds() if complete else None,
            interrupted_window_seconds=(interruption-start).total_seconds() if interruption and not complete else None,
            interruption_time=interruption.isoformat() if interruption and not complete else None,
            interruption_time_source='supervisor interrupted status file mtime; approximate' if interruption and not complete else None,
            start_time_source=time_source,elapsed_hms=hms((end-start).total_seconds()) if complete else None,
            interrupted_window_hms=hms((interruption-start).total_seconds()) if interruption and not complete else None,
            batch_exit_zero=str(run.resolve()) in completed_batch,experiment_dir=str(run),config=str(config),debug_sources=debug_sources,
            metric_summary=str(summary),train_log=str(log))
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/experiments/al_budding'))
    parser.add_argument('--output',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/analysis/al_pool_and_runtime_2026-10-28'))
    args=parser.parse_args();rows=summarize(args.experiments);args.output.mkdir(parents=True,exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.output/'experiment_runtimes.csv',index=False)
    (args.output/'experiment_runtimes.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(pd.DataFrame(rows)[['model','strategy','completed','completed_rounds','elapsed_hms','interrupted_window_hms','last_training_round','last_logged_epoch','last_logged_epoch_target']].to_string(index=False))


if __name__=='__main__':main()
