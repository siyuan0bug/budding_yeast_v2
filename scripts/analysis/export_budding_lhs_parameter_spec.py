"""Export all configured Budding LHS marginals as readable per-parameter tables.

This exports and checks a specification; it does not sample a joint LHS design,
integrate trajectories, infer mutation labels or write a production pool.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem


def number(x):
    return f'{float(x):.9g}'


def component_text(part):
    family=part['family']
    if family=='point':return f'固定 {number(part["value"])}'
    if family=='uniform':return f'U({number(part["lower"])}, {number(part["upper"])})'
    if family=='log_uniform':return f'LogU({number(part["lower"])}, {number(part["upper"])})'
    if family=='scaled_beta':
        return f'{number(part["parameter_scale"])} × Beta({number(part["a"])}, {number(part["b"])})'
    if family=='clipped_normal_scaled':
        return (f'{number(part["parameter_scale"])} × clip(N({number(part["alpha_mean"])}, '
                f'{number(part["alpha_std"])}²), {number(part["alpha_clip"][0])}, {number(part["alpha_clip"][1])})')
    if family=='discrete':
        values=part['values'];weights=part['probabilities']
        # Keep every value and probability in the machine-readable CSV/JSON.
        text='{'+', '.join(number(x) for x in values)+'}'
        if part.get('source_mode') in ('MC','GAL-MC'):
            return f'离散 {text}，各值权重∝1/N（N为原拷贝数）'
        return text+'，概率 {'+', '.join(f'{p:.8%}' for p in weights)+'}'
    raise ValueError(f'Unknown marginal family: {family}')


def merge_points(parts):
    result=[];points={}
    for part in parts:
        if part['family']!='point':result.append(dict(part));continue
        value=part['value']
        if value not in points:
            points[value]=dict(part,weight=0.);result.append(points[value])
        points[value]['weight']+=part['weight']
    return result


def atom_probability(parts,value):
    probability=0.
    for part in parts:
        if part['family']=='point' and part['value']==value:probability+=part['weight']
        if part['family']=='discrete':
            probability+=part['weight']*sum(p for v,p in zip(part['values'],part['probabilities']) if v==value)
    return probability


def bounds(part):
    if part['family']=='point':return part['value'],part['value']
    if part['family']=='discrete':return min(part['values']),max(part['values'])
    return part['lower'],part['upper']


def export(config_dir,output):
    output.mkdir(parents=True,exist_ok=True)
    spec=json.loads((config_dir/'mixed_lhs_domain.json').read_text())
    inventory=json.loads((config_dir/'inventory.json').read_text())
    if inventory['range_fingerprint']!=RANGE_FINGERPRINT:
        raise ValueError('LHS marginal configuration predates current approved ranges')
    with (config_dir/'parameter_domain_inventory.csv').open() as f: domain={r['name']:r for r in csv.DictReader(f)}
    with (config_dir/'unaffected_model_parameters.csv').open() as f: unaffected={r['name'] for r in csv.DictReader(f)}
    backend=BuddingSystem();defaults=backend.defaults();marginals=spec['marginals']
    assert set(marginals)==set(backend.PARAM_KEYS)==set(domain)
    rows=[]
    for index,key in enumerate(backend.PARAM_KEYS):
        parts=marginals[key]
        assert all(np.isfinite(p['weight']) and p['weight']>=0 for p in parts)
        assert abs(sum(p['weight'] for p in parts)-1)<1e-10
        for part in parts:
            lo,hi=bounds(part);assert 0<=lo<=hi and np.isfinite([lo,hi]).all()
            if part['family']=='discrete':
                assert len(part['values'])==len(part['probabilities'])
                assert min(part['probabilities'])>=0 and abs(sum(part['probabilities'])-1)<1e-10
        if key in unaffected:
            assert len(parts)==1 and parts[0]['family']=='uniform'
            assert np.allclose(bounds(parts[0]),[.5*defaults[key],1.5*defaults[key]],rtol=0,atol=1e-12)
        text='；'.join(f'{p["weight"]:.8%}：{component_text(p)}' for p in merge_points(parts))
        rows.append(dict(index_zero_based=index,parameter=key,kind=domain[key]['kind'],wt=defaults[key],
            group='positive_50_percent_window' if key in unaffected else 'mixed_marginal',
            lower=min(bounds(p)[0] for p in parts),upper=max(bounds(p)[1] for p in parts),
            zero_atom_probability=atom_probability(parts,0.),wt_atom_probability=atom_probability(parts,defaults[key]),
            readable_distribution=text,components_json=json.dumps(parts,ensure_ascii=False)))
    with (output/'all_141_parameters.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'mixed_lhs_domain.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))
    report=[
        '# Budding混合边际LHS：141维完整参数表', '',
        '状态：边际配置已整理；受约束联合LHS采样器未实现。本工具仅导出规格，没有生成样本或仿真。', '',
        '输入顺序为当前PARAM_KEYS；编号从0起。135个模型参数＝45个mutation可变＋90个±50%窗口，另有2个事件控制、3个初值覆盖和1个环境参数。', '',
        'U是数值均匀，LogU是对数均匀；固定值为离散原子。百分比是分量权重，不是参数上下界或最终仿真成功池的频率。相同数值的固定原子已在展示中合并，CSV/JSON保留完整来源与概率。', '',
        '零点概率只计算精确离散原子；U(0,a)或Beta触及0不自动赋予正概率零点。包络列仅供阅读，不允许代替混合CDF。', '',
        '90参数的±50%只属于LHS，mutation样本中未修改参数仍保持WT。ki20p/ki20pp是保留的无直接动力学作用坐标，不应解释为有敏感性的参数。', '',
        '权重来自1159个合法模块集合等概率、集合内完整合法模式等概率的生成前提案。非GAL环境80/15/5，含GAL固定150；这用于推导mdt单参数边际，不代表LHS中仍有GAL标签。', '',
        '必须反演完整混合CDF，不能把重叠区间按来源顺序拼接。初值/事件共享控制需保持物理约束；不同速率一般不再强制由同一个gene/mode共同缩放。', '',
    ]
    for title,condition in [
        ('一、45个mutation可变模型参数',lambda r:r['kind']=='model_parameter' and r['parameter'] not in unaffected),
        ('二、90个未被mutation覆盖的模型参数',lambda r:r['parameter'] in unaffected),
        ('三、6个环境、事件及初值控制',lambda r:r['kind']!='model_parameter')]:
        report += ['## '+title,'']
        if title.startswith('二'):
            report+=['这些参数全部按U(下界,上界)分层；没有额外零点原子。', '',
                     '| 序号 | 参数 | WT | 下界 | 上界 |','|---:|---|---:|---:|---:|']
            for r in filter(condition,rows):
                report.append(f'| {r["index_zero_based"]} | {r["parameter"]} | {number(r["wt"])} | {number(r["lower"])} | {number(r["upper"])} |')
        else:
            report+=['| 序号 | 参数 | WT | 完整边际分布 |','|---:|---|---:|---|']
            for r in filter(condition,rows):
                report.append(f'| {r["index_zero_based"]} | {r["parameter"]} | {number(r["wt"])} | {r["readable_distribution"].replace("；","<br>")} |')
        report.append('')
    report += ['## 初值与联合采样边界','',
        '- init_CDH1T与init_CDH1须共用同一个关闭指示/概率坐标，避免活性量超过总量。',
        '- BUB2_active与init_BUB2须共用关闭指示，避免初始化后边际被自动修正。',
        '- Tem1T或Cdc15T=0时，通过canonical初始化同步关闭对应初始活性状态。',
        '- 剩余39维初始状态按canonical初始化处理，不是39个变量全部独立LHS。',
        '- 没有保留gene/mode生成依赖的其他参数，可能出现mutation规则未产生过的联合取值；这是独立参数设计域，不可回填虚构mutation标签。',
        '- 边际一致不等于联合分布一致；90个额外扰动维度也使LHS与mutation域不同。联合数值成功率/Pattern分布需生成后审计。',
        '- real排除、去重和失败补采会改变最终经验分布，不能将本表权重当已生成10万有效样本的实测比例。','',
        '来源：'+str(config_dir), '',
        '生成工具：scripts/analysis/export_budding_lhs_parameter_spec.py。完整双精度权重、离散值概率与来源见all_141_parameters.csv及mixed_lhs_domain.json。',
    ]
    (output/'all_141_parameters.md').write_text('\n'.join(report)+'\n')
    summary=dict(parameters=len(rows),kinds=dict(Counter(r['kind'] for r in rows)),
        unaffected_model_parameters=len(unaffected),affected_model_parameters=135-len(unaffected),
        components=sum(map(len,marginals.values())),weights_sum_to_one=True,
        status=spec['status'],range_fingerprint=RANGE_FINGERPRINT,
        config_sha256=hashlib.sha256((config_dir/'mixed_lhs_domain.json').read_bytes()).hexdigest(),
        ode_calls=0,lhs_samples_generated=0)
    assert summary['parameters']==141 and len(unaffected)==90
    (output/'export_validation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();export(args.config_dir,args.output)


if __name__=='__main__': main()
