#!/usr/bin/env python3
"""Move frozen completed runs on one filesystem, keeping legacy paths as aliases."""
import argparse,csv,json,os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from budding_yeast_v2.utils.sampling.layout import experiment_directory,RESULTS


def inventory(directory):
    return {str(p.relative_to(directory)):(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)
            for p in directory.rglob('*') if p.is_file() and not p.is_symlink()}


def move(source,destination):
    source=Path(source);destination=Path(destination)
    if source.is_symlink() and source.resolve()==destination.resolve():return dict(files=len(inventory(destination)),reused=True)
    if destination.exists():raise FileExistsError(destination)
    if not source.is_dir():raise NotADirectoryError(source)
    before=inventory(source)
    links={str(p.relative_to(source)):p.resolve() for p in source.rglob('*') if p.is_symlink()}
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source.stat().st_dev!=destination.parent.stat().st_dev:raise ValueError('Refusing cross-device copy')
    source.rename(destination)
    source.symlink_to(destination,target_is_directory=True)
    # Preserve both external log links and internal links after changing depth.
    for relative,target in links.items():
        p=destination/relative
        try:target=destination/target.relative_to(source)
        except ValueError:pass
        p.unlink();p.symlink_to(target,target_is_directory=target.is_dir())
    assert inventory(destination)==before,'File bytes/mtime/inode inventory changed'
    return dict(files=len(before),reused=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,default=RESULTS);p.add_argument('--apply',action='store_true')
    args=p.parse_args();records=json.loads(args.inventory.read_text())['experiments'];plans=[]
    for r in records:
        c=json.loads(Path(r['config']).read_text())['training'];kind='generate' if 'generation' in r['workflow'] else 'run'
        new=experiment_directory(args.output,c,r['pool'],r['method'],kind,legacy=r['method']=='hggs_projected')
        if r['method']=='ogas_async_projected' and r['version']=='ogas_pre_alignment':
            new=Path(str(new).replace('/ogas/','/ogas_legacy/'))
        plans.append(dict(source=r['directory'],destination=str(new),pool=r['pool'],method=r['method'],workflow=r['workflow']))
    assert len({r['destination'] for r in plans})==len(plans),'Destination collision'
    if not args.apply:print(json.dumps(plans,ensure_ascii=False,indent=2));return
    log=args.output/'batches/layout_migration_20261120';log.mkdir(parents=True,exist_ok=True)
    (log/'plan.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2))
    # Preserve all original scientific metadata, including completion artifact hashes.
    for r in plans:
        r.update(move(r['source'],r['destination']))
        (log/'progress.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2))
    shared=args.output/'shared/pool100k/seed_42'
    shared_result=move(args.protocol,shared)
    # Queue logs are administration data, kept separate from algorithm directories.
    experiments=args.output.parent
    for family in ('budding_sampling_extensions','budding_ours_queue'):
        source=experiments/family
        if source.is_dir() and not source.is_symlink():move(source,args.output/'batches'/family)
    source=experiments/'budding_sampling'/json.loads((shared/'protocol.json').read_text())['protocol_id']/'batch'
    if source.is_dir() and not source.is_symlink():move(source,args.output/'batches/core_seed42')
    summary=args.output/'summary';summary.mkdir(exist_ok=True)
    for name in ('sampling_baselines_ours_20261119','selected_baselines_three_ours_20261119'):
        source=experiments.parent/'analysis'/name
        if source.is_dir() and not source.is_symlink():move(source,summary/name)
    with (summary/'experiments.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,list(plans[0]));w.writeheader();w.writerows(plans)
    result=dict(experiments=len(plans),files_checked=sum(r['files'] for r in plans),
                shared=shared_result,old_paths_are_aliases=True,trajectory_copies=0,scientific_files_rewritten=0)
    (log/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
