#!/usr/bin/env python3
"""Check repository boundaries without importing models or reading trajectories."""
import argparse,ast,json,re
from pathlib import Path


ROOT_ENTRIES={'.git','.gitignore','.pytest_cache','__pycache__','AGENTS.md','README.md',
 'train.py','eval.py','tune.py','run_pipeline.sh','run_tune.sh','run_sampling_budding.sh',
 'run_sampling_fission.sh','data','models','utils','scripts','tests','scratch','docs','experiments'}


def audit(root):
    issues=[]
    for p in root.iterdir():
        if p.name not in ROOT_ENTRIES:issues.append(dict(rule='root_allowlist',path=str(p.relative_to(root))))
    for p in (root/'scripts').rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        if p.suffix not in ('.py','.sh','.md'):
            issues.append(dict(rule='generated_product_in_scripts',path=str(p.relative_to(root))))
    active=[root/'train.py',root/'eval.py',root/'tune.py']
    for folder in ('utils','data','models','scripts'):
        active.extend(p for p in (root/folder).rglob('*.py') if 'archive' not in p.relative_to(root).parts and 'outputs' not in p.parts)
    for p in active:
        tree=ast.parse(p.read_text(),filename=str(p))
        for n in ast.walk(tree):
            names=([n.module or ''] if isinstance(n,ast.ImportFrom) else [x.name for x in n.names] if isinstance(n,ast.Import) else [])
            for name in names:
                if name.startswith(('budding_yeast_v2.utils.archive','utils.archive')) or 'archive' in name.split('.'):
                    issues.append(dict(rule='retired_runtime_dependency',path=str(p.relative_to(root)),line=n.lineno,module=name))
                if p.parts[len(root.parts)] in ('utils','models','data') and name.startswith(('scripts.','budding_yeast_v2.scripts.')):
                    issues.append(dict(rule='reverse_script_dependency',path=str(p.relative_to(root)),line=n.lineno,module=name))
        if p.name=='_compat.py':
            for assignment in tree.body:
                if not isinstance(assignment,ast.Assign) or not any(isinstance(t,ast.Name) and t.id=='ALIASES' for t in assignment.targets):continue
                for n in assignment.value.values:
                    if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('budding_yeast_v2.utils.archive'):
                        issues.append(dict(rule='retired_compat_target',path=str(p.relative_to(root)),target=n.value))
    # Only active navigation must be link-clean; history can describe retired assets.
    for p in [root/'README.md',root/'AGENTS.md',root/'utils/README.md',*(root/'docs').glob('*.md')]:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if link.startswith(('http:','https:','mailto:','#')):continue
            path=link.split('#')[0]
            if not path:continue
            if path.startswith('file://'):path=path[7:]
            if not (p.parent/path).exists():issues.append(dict(rule='broken_current_doc_link',path=str(p.relative_to(root)),target=link))
    return issues


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path)
    a=p.parse_args();issues=audit(a.root.resolve());result=dict(ok=not issues,issues=issues)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text)
    raise SystemExit(bool(issues))


if __name__=='__main__':main()
