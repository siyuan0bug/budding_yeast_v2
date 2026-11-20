#!/usr/bin/env python3
"""Summarize completed batch wall time, task durations and observed RAM usage."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from budding_yeast_v2.data.dataset_archive import atomic_json


def duration(seconds):
    seconds = int(round(seconds))
    return f'{seconds//3600}:{seconds%3600//60:02d}:{seconds%60:02d}'


def summarize(batch):
    tasks = json.loads((batch/'tasks.json').read_text())
    events = []
    for line in (batch/'launcher.log').read_text().splitlines():
        match = re.match(r'^\[([^]]+)\] \[(START|END|FAIL)\] sampling batch task', line)
        if match:
            events.append((datetime.fromisoformat(match[1]),match[2]))
    starts = [stamp for stamp,event in events if event == 'START']
    ends = [stamp for stamp,event in events if event in ('END','FAIL')]
    if not starts or not ends:
        raise ValueError('No complete timestamped batch task events')
    began, ended = min(starts), max(ends)
    total = sum(row['seconds'] for row in tasks)
    peak = active = count = 0
    available = float('inf')
    with (batch/'resource_samples.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            peak = max(peak,row['pss_gib'])
            active = max(active,len(row['jobs']))
            available = min(available,row['available_gib'])
            count += 1
    wall = (ended-began).total_seconds()
    return dict(start=began.isoformat(),end=ended.isoformat(),wall_seconds=wall,
        wall_hms=duration(wall),task_count=len(tasks),failures=sum(row['exit_code']!=0 for row in tasks),
        aggregate_task_seconds=total,aggregate_task_hms=duration(total),
        mean_task_hms=duration(total/len(tasks)),mean_simultaneous_tasks=total/wall,
        peak_simultaneous_tasks=active,peak_owned_pss_gib=peak,
        minimum_available_gib=available,resource_samples=count,tasks=tasks,
        limits='Task times include loading, training, scoring, evaluation, saving and waits inside each job; '
               'batch wall time excludes earlier pool/protocol preparation. PSS excludes kernel page cache.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = summarize(args.batch)
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/'timing.json',report)
    table = {}
    for row in report['tasks']:
        table.setdefault(row['method'],{})[row['pool']] = duration(row['seconds'])
    lines = ['# Sampling batch time and memory', '',
        f"Wall time: {report['wall_hms']}; aggregate task time: {report['aggregate_task_hms']}.",
        f"Completed {report['task_count']}, failures {report['failures']}; peak jobs {report['peak_simultaneous_tasks']}.",
        f"Peak owned PSS {report['peak_owned_pss_gib']:.2f} GiB; minimum available RAM {report['minimum_available_gib']:.2f} GiB.",
        '', '| Method | Mutation | LHS |', '|---|---|---|']
    lines += [f"| {name} | {row.get('mutation','—')} | {row.get('lhs','—')} |" for name,row in sorted(table.items())]
    lines += ['',report['limits'],'']
    (args.output/'report.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in report.items() if k!='tasks'},indent=2))


if __name__ == '__main__':
    main()
