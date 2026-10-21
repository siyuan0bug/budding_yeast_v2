#!/usr/bin/env python3
"""Count real benchmark entries by named model-module multiplicity, without simulation.

CLN1/2, CLB1/2 and CLB5/6 are lumped model units. Carbon-source
backgrounds are not additional genetic modules. Same-parameter real identities
and distinct experimental conditions remain separate evaluation entries.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
ALIASES = {'CLN1': 'CLN2', 'CLB1': 'CLB2', 'CLB6': 'CLB5',
           'TAB6-1': 'NET1', 'APC': 'APC_Core'}
BUDDING_TOKENS = re.compile(r'CLN[123]|CLB[1256]|SIC1|CDH1|CDC6|SWI5|CDC20|PDS1|ESP1|TEM1|CDC15|NET1|CDC14|MAD2|BUB2|BCK2|TAB6-1|APC')
FISSION_TOKENS = re.compile(r'WEE1|MIK1|PYP3|CDC25|RUM1|STE9|CDC13')
FISSION_NAMES = {'WEE1':'Wee1','MIK1':'Mik1','PYP3':'Pyp3','CDC25':'Cdc25',
                 'RUM1':'Rum1','STE9':'Ste9','CDC13':'Cdc13'}


def literal_assignment(path, name):
    tree = ast.parse(path.read_text())
    return ast.literal_eval(next(node.value for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id==name for target in node.targets)))


def count_rows(system):
    path = ROOT/f'data/systems/{system}/real_mutants.py'
    table = literal_assignment(path, 'REAL_MUTANTS')
    catalog_path = ROOT/f'data/systems/{system}/catalog.py'
    catalog = literal_assignment(catalog_path, 'BINDING_GROUPS' if system=='budding' else 'SAMPLING_MODULES')
    legal = {group['name'] for group in catalog.values()} if system=='budding' else set(catalog)
    result=[]
    for identity, definition in table.items():
        name, overrides = definition if system=='budding' else (identity, definition[0])
        tokens = sorted(set((BUDDING_TOKENS if system=='budding' else FISSION_TOKENS).findall(name.upper())))
        modules = sorted({ALIASES.get(token,token) if system=='budding' else FISSION_NAMES[token] for token in tokens})
        assert set(modules) <= legal, (identity,modules)
        is_wt = str(identity) in ('1','2.1','2.2') if system=='budding' else identity=='WT'
        assert bool(modules) != is_wt, (identity,name)
        notes=[]
        if 'APC' in tokens:notes.append('APC counted as APC_Core, not each downstream parameter target')
        if 'TAB6-1' in tokens:notes.append('TAB6-1 mapped to NET1')
        if system=='budding' and str(identity)=='104':notes.append('Delayed SIC1 induction counted; is_mutant_104 acts in ODE at t>=180')
        if system=='fission' and identity=='cdc13dbD':notes.append('Named Cdc13 mutant; V2/V2c footprint shared with Ste9 KO')
        result.append(dict(system=system,identity=str(identity),name=name,is_wt=is_wt,
            module_count=len(modules),modules=';'.join(modules),
            name_token_count=len(tokens),name_tokens=';'.join(tokens),
            override_key_count=len(overrides),notes='; '.join(notes)))
    counts=Counter(row['module_count'] for row in result)
    mutant_total=sum(not row['is_wt'] for row in result)
    summary=dict(total=len(result),wt_entries=counts[0],mutant_entries=mutant_total,
        counts={str(key):counts[key] for key in range(max(counts)+1)},
        percent_all={str(key):100*counts[key]/len(result) for key in range(max(counts)+1)},
        percent_non_wt={str(key):100*counts[key]/mutant_total for key in range(1,max(counts)+1)},
        source=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result,summary


def recipe_rows(system):
    tree=ast.parse((ROOT/f'data/systems/{system}/recipes.py').read_text())
    table=next(node.value for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='RECIPES' for target in node.targets))
    rows=[]
    for key,value in zip(table.keys,table.values):
        name=ast.literal_eval(key)
        sampler=ast.literal_eval(value.args[1])
        ratios=(literal_assignment(ROOT/'data/systems/sampling_policy.py',value.args[2].id)
                if isinstance(value.args[2],ast.Name) else ast.literal_eval(value.args[2]))
        rows.append(dict(recipe=name,sampler=sampler,level_percent=[100*v for v in ratios]))
    return rows


def run_audit(output):
    output.mkdir(parents=True,exist_ok=True)
    all_rows=[];systems={}
    for system in ('budding','fission'):
        rows,summary=count_rows(system)
        all_rows.extend(rows);systems[system]=summary
    assert systems['budding']['total']==126 and systems['fission']['total']==20
    assert systems['budding']['wt_entries']==3 and systems['fission']['wt_entries']==1
    report=dict(counting_unit='named model modules; benchmark entries not deduplicated',
        excludes=['carbon-source background','multiple modes on the same module','parameter-key multiplicity'],
        budding_aliases=ALIASES,systems=systems,
        recipe_profiles=recipe_rows('budding')+recipe_rows('fission'),
        limits=['Name/biological-module counting; not a unique inverse of parameter touch sets.',
                'APC_Core, CDC20 and CDH1 have overlapping footprints; do not sum key matches.',
                'Nominal level quotas are not guaranteed final module counts after conflict rescue/filtering.',
                'MDT/Growth environmental sampling modules have no extra gene count here.',
                'Benchmarks define an evaluation distribution, not population prevalence or optimal training allocation.'])
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (output/'per_real_mutant_modules.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    lines=['# Real mutant 模块数统计','',
        '按模型中的命名模块计数，同模块多种突变只计一次；不按参数键数量计数。WT/不同碳源和同参数身份保留为独立评测行。','',
        '| 模块数 | Budding 数量/126 | Budding 占比 | Fission 数量/20 | Fission 占比 |',
        '|---|---:|---:|---:|---:|']
    for key in range(5):
        values=[]
        for system in ('budding','fission'):
            summary=systems[system];values.extend([summary['counts'].get(str(key),0),summary['percent_all'].get(str(key),0)])
        lines.append(f'| {key} | {values[0]} | {values[1]:.2f}% | {values[2]} | {values[3]:.2f}% |')
    lines+=['','排除 WT 后：']
    for system in ('budding','fission'):
        lines.append(f"- {system}（n={systems[system]['mutant_entries']}）："+', '.join(f"{key} 模块 {value:.2f}%" for key,value in systems[system]['percent_non_wt'].items()))
    lines+=['','## 当前 nominal level 配额','',
            '| recipe | sampler | L1–Ln 百分比 |','|---|---|---|']
    for row in report['recipe_profiles']:
        lines.append(f"| {row['recipe']} | {row['sampler']} | "+'/'.join(f'{value:g}' for value in row['level_percent'])+' |')
    lines+=['','budding_random / budding_pool：起始选择 k=level，冲突处理可能减少最终模块数。',
        'MASS 分层：L1=1、L2=2、L3=3–5、L4=4–6，不能将四个 level 当作精确的 1/2/3/4 模块。',
        'Fission：起始选择 1/2/3 个采样模块，其中 Growth 是环境/生长模块。',
        '上述定义是结构统计；并不据此断言经验配额最优，也未修改任何采样配额。','']
    (output/'report.md').write_text('\n'.join(lines))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    report=run_audit(args.output_dir)
    print(json.dumps(report['systems'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
