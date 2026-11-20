#!/usr/bin/env python3
"""Sequential verified cross-device relocation; no unverified source deletion."""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def relocate(source,target,log):
    files=list(source.rglob('*')) if source.is_dir() else [source]
    records=[];links=[];total=0
    target.parent.mkdir(parents=True,exist_ok=True)
    for p in files:
        relative=p.relative_to(source) if source.is_dir() else Path('.')
        q=target/relative if source.is_dir() else target
        if p.is_symlink():links.append((p,q,p.resolve()));continue
        if p.is_dir():q.mkdir(parents=True,exist_ok=True);continue
        q.parent.mkdir(parents=True,exist_ok=True)
        checksum=digest(p)
        if q.exists():
            if digest(q)!=checksum:raise ValueError(f'Collision: {q}')
        else:
            tmp=q.with_name('.'+q.name+'.copying');shutil.copy2(p,tmp)
            if digest(tmp)!=checksum:raise ValueError(f'Copy verification failed: {p}')
            tmp.replace(q)
        total+=p.stat().st_size;records.append(dict(relative=str(relative),sha256=checksum,bytes=p.stat().st_size))
        if len(records)%200==0:print(json.dumps(dict(source=str(source),files=len(records),bytes=total)),flush=True)
    for p,q,dest in links:
        try:dest=target/dest.relative_to(source.resolve())
        except ValueError:pass
        q.parent.mkdir(parents=True,exist_ok=True)
        if q.is_symlink():q.unlink()
        q.symlink_to(dest,target_is_directory=dest.is_dir())
    log.parent.mkdir(parents=True,exist_ok=True)
    log.write_text(json.dumps(dict(source=str(source),target=str(target),bytes=total,files=records,symlinks=len(links)),indent=2))
    if source.is_dir():shutil.rmtree(source)
    else:source.unlink()
    print(json.dumps(dict(complete=str(source),target=str(target),files=len(records),bytes=total)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--result-root',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
    repo=a.repository.resolve();dest=a.result_root.resolve()
    names=['archive','eval_result','train_result','tune_results','model_visualizations','swanlog','analysis']
    plan=[(repo/n,dest/'archive/repository_outputs_20261121'/n) for n in names]
    plan += [(repo/'scripts/analysis/outputs',dest/'budding_sampling/summary/historical_analysis_outputs')]
    plan += [(repo/n,dest/'archive/repository_outputs_20261121'/n) for n in ['codex.txt','mutant_cross_model_comparison.csv']]
    plan=[(s,t) for s,t in plan if s.exists() and not s.is_symlink()]
    if not a.apply:print(json.dumps([(str(s),str(t)) for s,t in plan],indent=2));return
    logs=dest/'budding_sampling/batches/repository_governance_20261121'
    for s,t in plan:relocate(s,t,logs/(s.name.replace('.','_')+'_migration.json'))


if __name__=='__main__':main()
