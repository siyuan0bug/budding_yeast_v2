"""Read-only storage inventory and loss-tail audit for manifest-selected AL runs.

Writes small CSV/JSON reports only. Does not remove, link, compress or rewrite
datasets/checkpoints. ZIP inspection reads headers and central directories only.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import re
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def category(path):
    parts=path.parts
    if 'wandb' in parts:return 'wandb'
    if 'shared' in parts and 'datasets' in parts and path.suffix=='.npz':return 'shared_initial'
    if path.name in ('dataset.json','initial_dataset.json') or path.name.startswith('dataset.metadata.'):
        return 'dataset_manifests'
    if path.name=='initial_dataset.npz':return 'initial_copy'
    if path.name=='dataset.npz' and 'rounds' in parts:return 'cumulative_datasets'
    if path.name=='added_samples.npz':return 'added_datasets'
    if path.suffix=='.ckpt':return 'ensemble_checkpoints' if 'ensemble' in parts else 'main_checkpoints'
    if path.suffix in ('.png','.pdf','.svg'):return 'plots'
    if path.suffix=='.npz':return 'other_arrays'
    return 'logs_metadata_other'


def inventory(root):
    files=[];configs=set();directories=0
    for folder,dirs,names in os.walk(root,followlinks=False):
        directories+=1
        for name in names:
            path=Path(folder)/name
            if path.is_symlink():continue
            try:stat=path.stat()
            except FileNotFoundError:continue
            if not path.is_file():continue
            if name=='config.json':configs.add(path.parent)
            files.append((path,stat.st_size,stat.st_blocks*512,stat.st_dev,stat.st_ino))
    rows=defaultdict(lambda:dict(logical_bytes=0,allocated_bytes=0,files=0))
    categories=defaultdict(lambda:dict(logical_bytes=0,allocated_bytes=0,files=0))
    run_inodes=defaultdict(set)
    for path,size,allocated,dev,ino in files:
        owner=next((parent for parent in path.parents if parent in configs),None)
        name=str(owner) if owner else '__outside_config_runs__'
        kind=category(path.relative_to(root))
        row=rows[name];row['logical_bytes']+=size;row['files']+=1
        group=categories[(name,kind)];group['logical_bytes']+=size;group['files']+=1
        if (dev,ino) not in run_inodes[name]:
            row['allocated_bytes']+=allocated;group['allocated_bytes']+=allocated
            run_inodes[name].add((dev,ino))
    # The per-run view charges a shared inode to each referencing run. The total
    # filesystem view counts each inode once across the entire tree.
    unique={}
    for _,_,allocated,dev,ino in files:unique[(dev,ino)]=allocated
    return rows,categories,dict(files=len(files),directories=directories,
        logical_bytes=sum(x[1] for x in files),unique_allocated_file_bytes=sum(unique.values()),
        note='Allocated file blocks; directory blocks are excluded. Symlinks not followed.'),files


def array_header(archive,key):
    with archive.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):return np.lib.format.read_array_header_1_0(stream)
        if version==(2,0):return np.lib.format.read_array_header_2_0(stream)
        raise ValueError(f'Unsupported header {version}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    run_rows,categories,overall,files=inventory(a.experiments)
    tasks=json.loads(a.manifest.read_text())['tasks']
    current={t['experiment_dir']:t for t in tasks}
    all_rows=[]
    for name,values in run_rows.items():
        row=dict(run=name,**values)
        if name!='__outside_config_runs__':
            try:
                conf=json.loads((Path(name)/'config.json').read_text());args=conf.get('args',conf)
                row.update(model=args.get('model'),strategy=args.get('al_strategy','none'),seed=args.get('seed'))
            except (ValueError,OSError):pass
        row['current_batch']=name in current
        all_rows.append(row)
    frame=pd.DataFrame(all_rows).sort_values('allocated_bytes',ascending=False)
    frame.to_csv(a.output/'all_experiment_sizes.csv',index=False)
    groups=pd.DataFrame([dict(run=name,category=kind,**values) for (name,kind),values in categories.items()])
    groups.to_csv(a.output/'all_experiment_categories.csv',index=False)
    selected=frame[frame.current_batch].copy()
    selected['GiB']=selected.allocated_bytes/2**30
    selected.to_csv(a.output/'current_batch_sizes.csv',index=False)
    groups[groups.run.isin(current)].to_csv(a.output/'current_batch_categories.csv',index=False)

    convergence=[];arrays=[];save_rows=[]
    for task in tasks:
        run=Path(task['experiment_dir']);name=task['strategy']
        for folder in sorted(run.glob('rounds/round_*')):
            report=folder/'convergence.json'
            if report.exists():
                obj=json.loads(report.read_text());end=obj.get('final') or {}
                h=pd.read_csv(folder/'training_history.csv')
                row=dict(strategy=name,round=int(folder.name.rsplit('_',1)[-1]),epochs=len(h),
                    status=obj['status'],consecutive_checks=obj['consecutive_checks'],
                    train_trend_pct=end.get('train_trend_pct'),val_trend_pct=end.get('val_trend_pct'),
                    best_val_epoch=int(h.loc[h.val_synth.idxmin(),'epochs']),
                    final_val_gap_pct=100*(h.val_synth.iloc[-1]/h.val_synth.min()-1))
                for window in (10,20):
                    for key in ('train_loss','val_synth'):
                        values=h[key].tail(window+1).to_numpy()
                        row[f'{key}_net_drop_last{window}_pct']=100*(1-values[-1]/values[0])
                convergence.append(row)
        for path in [run/'data/initial_dataset.npz',*sorted(run.glob('rounds/round_*/dataset.npz')),
                     *sorted(run.glob('acquisitions/after_round_*/added_samples.npz'))]:
            if not path.exists():continue
            with zipfile.ZipFile(path) as z:
                info=z.getinfo('data.npy');shape,_,dtype=array_header(z,'data')
                arrays.append(dict(strategy=name,path=str(path),rows=shape[0],dtype=str(dtype),
                    file_bytes=path.stat().st_size,data_compressed_bytes=info.compress_size,
                    data_raw_bytes=info.file_size,category=category(path.relative_to(run))))
        # Two mutually exclusive lossless layout estimates. They assume a loader
        # supporting dataset views; deleting current files would break replay.
        present=[x for x in arrays if x['strategy']==name]
        snapshot=sum(x['file_bytes'] for x in present if x['category']=='cumulative_datasets')
        added=sum(x['file_bytes'] for x in present if x['category']=='added_datasets')
        initial=sum(x['file_bytes'] for x in present if x['category']=='initial_copy')
        snapshots=[x for x in present if x['category']=='cumulative_datasets']
        if snapshots:
            latest=max(snapshots,key=lambda x:x['rows'])
            save_rows.append(dict(strategy=name,current_redundant_dataset_bytes=snapshot+added+initial,
                estimated_shards_local_bytes=added,
                estimated_shards_saved_bytes=snapshot+initial,
                estimated_final_view_local_bytes=latest['file_bytes'],
                estimated_final_view_saved_bytes=snapshot+added+initial-latest['file_bytes'],
                latest_rows=latest['rows'],
                note='Proposal only; assumes shared initial and exact round-view loader. No deletion authorized.'))
    pd.DataFrame(convergence).to_csv(a.output/'convergence_all_rounds.csv',index=False)
    pd.DataFrame(arrays).to_csv(a.output/'dataset_archive_sizes.csv',index=False)
    pd.DataFrame(save_rows).to_csv(a.output/'lossless_layout_estimates.csv',index=False)
    folders=defaultdict(int)
    for path,_,allocated,_,_ in files:
        relative=path.relative_to(a.experiments)
        folders[relative.parts[0]]+=allocated
    fs=os.statvfs(a.experiments)
    report=dict(checked=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),experiments=overall,
        current_batch_allocated_file_bytes=int(selected.allocated_bytes.sum()),
        current_batch_categories=groups[groups.run.isin(current)].groupby('category').allocated_bytes.sum().to_dict(),
        top_level_allocated_file_bytes=dict(folders),
        filesystem=dict(total_bytes=fs.f_blocks*fs.f_frsize,available_to_user_bytes=fs.f_bavail*fs.f_frsize,
            free_including_reserved_bytes=fs.f_bfree*fs.f_frsize),
        constraints=['Read-only audit; no data contents changed by this tool',
            'Evaluation supports adjacent dataset.json or legacy dataset.npz; hashes/splits must match checkpoints',
            'Filesystem capacity is not administrator-assigned personal allowance'])
    (a.output/'storage_snapshot.json').write_text(json.dumps(report,indent=2))
    print(selected[['strategy','GiB']].to_string(index=False))
    print('Current total GiB',selected.allocated_bytes.sum()/2**30)
    print(groups[groups.run.isin(current)].groupby('category').allocated_bytes.sum().div(2**30).to_string())
    print('Additional estimated saving GiB',sum(row['estimated_shards_saved_bytes'] for row in save_rows)/2**30)
    print(a.output)


if __name__=='__main__':main()
