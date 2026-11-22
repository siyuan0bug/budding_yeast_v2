"""Validate multi-seed task/protocol expansion before any child admission."""
import json
from dataclasses import asdict
from pathlib import Path

from budding_yeast_v2.data.dataset_archive import file_sha256
from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig


def expand_seeds(rows,primary_root,seeds):
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Unique experiment seeds required')
    primary_root=Path(primary_root);first=json.loads((primary_root/'protocol.json').read_text())
    if seeds[0]!=first['config']['seed']:raise ValueError('First seed must match primary protocol')
    def training(v):return {k:x for k,x in asdict(BenchmarkConfig(**v['config']).validate()).items() if k!='seed'}
    def pools(v):return {k:x['sha256'] for k,x in v['pools'].items()}
    output=[];receipts={}
    for seed in seeds:
        root=primary_root if seed==seeds[0] else primary_root.parent/f'seed_{seed}'
        p=root/'protocol.json';value=json.loads(p.read_text())
        if value['config']['seed']!=seed or training(value)!=training(first):raise ValueError('Multi-seed training protocol mismatch')
        if pools(value)!=pools(first) or value.get('real_reference')!=first.get('real_reference'):
            raise ValueError('Multi-seed pool/reference identity mismatch')
        receipts[str(root.resolve())]=dict(sha256=file_sha256(p),fingerprint=value['fingerprint'])
        for row in rows:
            task=dict(row,seed=seed,protocol=str(root.resolve()))
            if seed!=seeds[0]:task['task_id']+=f'_seed_{seed}'
            output.append(task)
    return output,receipts


def verify_protocols(receipts):
    for root,record in receipts.items():
        if file_sha256(Path(root)/'protocol.json')!=record['sha256']:
            raise ValueError('Frozen multi-seed protocol changed')


def validate_adoption_tasks(rows,previous,*,extend=False):
    expected={r['task_id']:r for r in rows};old={r['task_id']:r for r in previous}
    if len(expected)!=len(rows) or len(old)!=len(previous):raise ValueError('Duplicate task identity')
    if not (set(old)<=set(expected) if extend else set(old)==set(expected)):
        raise ValueError('Live adoption task set changed')
    for name,row in old.items():
        if any(row.get(k)!=expected[name].get(k) for k in ('kind','method','pool','output','seed')):
            raise ValueError('Live adoption task definition/output/seed changed')
