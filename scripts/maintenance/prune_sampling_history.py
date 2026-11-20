#!/usr/bin/env python3
"""Prune obsolete weights/figures after rebasing live JSON and symlink paths."""
import argparse,hashlib,json,os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_name('.'+path.name+'.rebase');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(temp,path)


def repair_checksums(root,plan,audit):
    """Propagate audited JSON locator changes without changing array/model digests."""
    changed=[]
    for _ in range(8):
        count=0
        for row in plan:
            d=Path(row['destination'])
            for f in d.rglob('*.json'):
                if f.is_symlink():continue
                a=json.loads(f.read_text());before=json.dumps(a,sort_keys=True)
                def repair(x):
                    if isinstance(x,list):
                        for v in x:repair(v)
                    elif isinstance(x,dict):
                        for v in list(x.values()):repair(v)
                        source=x.get('source')
                        if isinstance(source,str) and 'source_sha256' in x:
                            target=(f.parent/source).resolve()
                            if target.suffix=='.json':x['source_sha256']=sha(target)
                        source=x.get('path')
                        if isinstance(source,str) and 'sha256' in x:
                            target=(f.parent/source).resolve()
                            if target.suffix=='.json' and target!=f.resolve():x['sha256']=sha(target)
                        for relative in x.get('artifacts',{}):
                            target=(d/relative).resolve()
                            if target.suffix=='.json' and target!=f.resolve():x['artifacts'][relative]=sha(target)
                repair(a)
                if json.dumps(a,sort_keys=True)!=before:
                    backup=audit/'before'/f.relative_to(root)
                    if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(f.read_bytes())
                    atomic(f,a);changed.append(str(f));count+=1
        if not count:return sorted(set(changed))
    raise RuntimeError('JSON reference hash propagation did not converge')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--apply',action='store_true');args=p.parse_args();root=args.root
    audit=root/'batches/history_prune_20261121';plan=json.loads((root/'batches/layout_migration_20261120/plan.json').read_text())
    replacements={}
    for row in plan:
        for prefix in (row['source'],row['source'].replace('/home/users/zsy/mydata','/data/zsy')):
            replacements[prefix]=row['destination']
    def replace(x):
        if isinstance(x,dict):return {k:replace(v) for k,v in x.items()}
        if isinstance(x,list):return [replace(v) for v in x]
        if isinstance(x,str):
            for old,new in replacements.items():
                if x==old or x.startswith(old+'/'):return new+x[len(old):]
        return x
    changes=[];links=[];metrics={}
    for row in plan:
        d=Path(row['destination']);metrics[str(d)]=(d/'eval/global_metrics_denorm.json').read_bytes()
        for f in d.rglob('*.json'):
            if f.is_symlink():continue
            a=json.loads(f.read_text());b=replace(a)
            if a!=b:changes.append((f,b))
        for f in d.rglob('*'):
            if f.is_symlink():
                target=f.resolve(strict=True)
                if str(target)!=os.readlink(f):links.append((f,target))
    archives=root.parent/'archive/sampling_legacy_paths'
    names=('seed42_02eebd57592f','seed42_bb035a147b38','seed42_a866aef5d988')
    removals=[f for name in names for f in (archives/name).rglob('*')
              if f.is_file() and not f.is_symlink() and f.suffix.lower() in ('.ckpt','.png','.pt','.pth')]
    report=dict(metadata_files_rebased=len(changes),links_rebased=len(links),
                obsolete_files_deleted=len(removals),bytes_freed=sum(f.stat().st_size for f in removals),
                preserved='configs, metrics JSON/CSV, histories, logs, scientific data and current weights')
    if not args.apply:print(json.dumps(report));return
    audit.mkdir(parents=True,exist_ok=True)
    # Keep exact before bytes for every path-only metadata change; no model/data edits.
    for f,b in changes:
        backup=audit/'before'/f.relative_to(root);backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(f.read_bytes());atomic(f,b)
    for f,target in links:f.unlink();f.symlink_to(target,target_is_directory=target.is_dir())
    completion_changes=repair_checksums(root,plan,audit)
    (audit/'removed_files.json').write_text(json.dumps([str(f) for f in removals],indent=2))
    for f in removals:f.unlink()
    # Drop unused compatibility descendants in the archived latest protocol.
    for name in names:
        original=root/name
        assert original.is_symlink() and original.resolve()==(archives/name).resolve()
        original.unlink()
    for row in plan:
        d=Path(row['destination']);assert (d/'eval/global_metrics_denorm.json').read_bytes()==metrics[str(d)]
        for f in d.rglob('*'):
            if f.is_symlink():assert f.exists(),f
        for f in d.glob('rounds/round_*/dataset.json'):
            a=json.loads(f.read_text())
            if isinstance(a.get('source'),str):assert Path(a['source']).exists(),f
    report.update(completion_records_refreshed=len(completion_changes),current_metrics_unchanged=True,old_root_links_removed=list(names))
    atomic(audit/'verification.json',report);print(json.dumps(report))


if __name__=='__main__':main()
