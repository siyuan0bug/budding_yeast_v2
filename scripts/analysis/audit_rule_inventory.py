"""Read-only inventory of every original file and its modular rule counterparts.

Normalized AST matches establish source structure, not equivalence of a whole
driver or of changed global bindings. Unmatched rules remain explicitly listed.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'
B='systems/budding/'
F='systems/fission/'
S='sampling/'
TARGETS={
 'mutant_aware.py':[B+p for p in ('states.py','parameters.py','ode.py','phenotypes.py','simulation.py','real_mutants.py','catalog.py','topology.py','mutations/mutant_aware.py','distributions/mutant_aware.py')]+[S+'common.py',S+'mass.py'],
 'mass_dataset_generator.py':[B+p for p in ('states.py','parameters.py','ode.py','phenotypes.py','simulation.py','real_mutants.py','catalog.py','mutations/mass.py','distributions/mass.py')]+[S+'mass_motifs.py'],
 'generate_pure_random_dataset.py':[S+'random.py'],
 'mass_random_sampler.py':[S+'random.py'],
 'mass_random_mutant_dist.py':[S+'random.py'],
 'gibbs_boltzmann_sampler.py':[S+'gibbs.py'],
 'gibbs_sobol_sampler.py':[S+'gibbs.py',S+'sobol.py',B+'distributions/quantiles.py'],
 'generate_kdpp_gated_dataset.py':[S+'kdpp.py'],
 'generate_kdpp_curv_gated_dataset.py':[S+'kdpp.py'],
 'generate_motif_topology_dataset.py':[S+'motif.py',B+'topology.py'],
 'bams_dataset_generator.py':[S+'bams.py',B+'topology.py'],
 'mass_wt_blind_sampler.py':[S+'wt_blind.py'],
 'generate_1m_pure_random_pool.py':[S+'sobol.py',B+'distributions/quantiles.py'],
 'generate_distribution_ablation_twins.py':[S+'twins.py'],
 'mirror_pure_random_ko_interval.py':['generation/derived.py'],
 'generate_bams_ablation_datasets.py':[S+'targeted_ablation.py','generation/derived.py'],
 'fission_random_sampler.py':[F+p for p in ('constants.py','states.py','parameters.py','ode.py','simulation.py','phenotypes.py','real_mutants.py','catalog.py','distributions.py')]+[S+'fission_random.py'],
}
RULE_CONSTANTS={'BINDING_GROUPS','REAL_MUTANTS','MODULE_EDGES','TOUCH','MODE_CLASS','MODE_CLASS_QUOTA',
 'TARGET_MODES','PARALLEL_DISTAL_PAIRS','MASTER_SWITCH_CLUSTERS','LEVEL_QUOTAS','STRUCT_L2','STRUCT_L34',
 'EPS','TRAJ_MAX','N_T','TF_MIN','MASS_UNIT','MAX_EVENTS','MASS_LIMIT','VAR_NAMES'}


class InterfaceOnly(ast.NodeTransformer):
    """Remove docs/import moves and explicit backend injection only."""
    def visit_FunctionDef(self,node):
        node.body=[s for s in node.body if not (isinstance(s,ast.Assign) and len(s.targets)==1
                      and ast.unparse(s.targets[0])=='self.ma' and ast.unparse(s.value)=='ma')]
        node=self.generic_visit(node)
        node.body=[s for s in node.body if not (isinstance(s,ast.Expr) and isinstance(s.value,ast.Constant)
                         and isinstance(s.value.value,str)) and not isinstance(s,(ast.Import,ast.ImportFrom))]
        keep=[i for i,a in enumerate(node.args.kwonlyargs) if a.arg!='ma']
        node.args.kwonlyargs=[node.args.kwonlyargs[i] for i in keep]
        node.args.kw_defaults=[node.args.kw_defaults[i] for i in keep]
        return node
    def visit_ClassDef(self,node):
        node=self.generic_visit(node)
        node.body=[s for s in node.body if not (isinstance(s,ast.Expr) and isinstance(s.value,ast.Constant) and isinstance(s.value.value,str))]
        return node
    def visit_Attribute(self,node):
        node=self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='self' and node.attr=='ma':
            return ast.copy_location(ast.Name(id='ma',ctx=node.ctx),node)
        if isinstance(node.value,ast.Name) and node.value.id=='ma':
            return ast.copy_location(ast.Name(id=node.attr,ctx=node.ctx),node)
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        node.keywords=[k for k in node.keywords if k.arg!='ma']
        return node


def normalize(node,rng_interface=False):
    node=copy.deepcopy(node)
    if rng_interface:
        keep=[i for i,a in enumerate(node.args.kwonlyargs) if a.arg!='rng']
        node.args.kwonlyargs=[node.args.kwonlyargs[i] for i in keep]
        node.args.kw_defaults=[node.args.kw_defaults[i] for i in keep]
        node.body=[s for s in node.body if not (isinstance(s,ast.Assign) and len(s.targets)==1
                   and isinstance(s.targets[0],ast.Name) and s.targets[0].id=='rng')]
        class RNG(ast.NodeTransformer):
            def visit_Name(self,x):
                return ast.parse('np.random',mode='eval').body if x.id=='rng' and isinstance(x.ctx,ast.Load) else x
        node=RNG().visit(node)
    return ast.dump(InterfaceOnly().visit(node),include_attributes=False)


def definitions(path):
    tree=ast.parse(path.read_text())
    return tree,{n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def options(tree):
    rows=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            flags=[a.value for a in node.args if isinstance(a,ast.Constant) and isinstance(a.value,str)]
            default=next((ast.unparse(k.value) for k in node.keywords if k.arg=='default'),'<argparse implicit>')
            rows.append(dict(line=node.lineno,flags=flags,default_expression=default))
    return rows


def default_args(node):
    if not isinstance(node,ast.FunctionDef):return {}
    names=[a.arg for a in node.args.posonlyargs+node.args.args]
    result=dict(zip(names[-len(node.args.defaults):],map(ast.unparse,node.args.defaults))) if node.args.defaults else {}
    result.update({a.arg:ast.unparse(v) for a,v in zip(node.args.kwonlyargs,node.args.kw_defaults) if v is not None})
    return result


def inventory():
    manifest=json.loads((DATA/'generation/legacy_sources.json').read_text())
    new_paths=[DATA/'generate.py']+[p for directory in ('systems','sampling','generation','recipes') for p in (DATA/directory).rglob('*.py')]
    modules={str(p.relative_to(DATA)):definitions(p) for p in new_paths}
    rows=[]
    for relative,digest in manifest.items():
        path=ROOT/relative
        old_tree,old_defs=definitions(path)
        basename=path.name
        scope='migrated' if basename in TARGETS else ('excluded_obsolete' if 'LHS' in path.parts or basename=='real_mutants_only.py' else 'original_module_unchanged')
        if scope=='original_module_unchanged' and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            scope='consumer_migrated'
        candidates=TARGETS.get(basename,[])
        rules=[]
        for name,node in old_defs.items():
            if scope!='migrated':
                rules.append(dict(name=name,line=node.lineno,status=scope));continue
            available=[(p,modules[p][1][name]) for p in candidates if name in modules[p][1]]
            use_rng=name in ('sample_scaling_factor','apply_gene_mutation')
            matches=[(p,n) for p,n in available if normalize(node,use_rng)==normalize(n,use_rng)]
            if matches:
                p,new=matches[0]
                status='normalized_source_match'
            elif available:
                p,new=available[0]
                status='source_diff_requires_rule_explanation'
            else:
                p,new=None,None
                status='replaced_driver_or_wrapper_requires_rule_explanation'
            rules.append(dict(name=name,line=node.lineno,status=status,new_file=p,new_line=getattr(new,'lineno',None),
                              original_function_defaults=default_args(node),new_function_defaults=default_args(new)))
        constants=[]
        for node in old_tree.body:
            if not isinstance(node,ast.Assign):continue
            for target in node.targets:
                if not isinstance(target,ast.Name) or target.id not in RULE_CONSTANTS:continue
                matches=[]
                differing=[]
                for dest in candidates:
                    for other in modules[dest][0].body:
                        if isinstance(other,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target.id for t in other.targets):
                            (matches if ast.dump(node.value,include_attributes=False)==ast.dump(other.value,include_attributes=False) else differing).append(dict(file=dest,line=other.lineno))
                constants.append(dict(name=target.id,line=node.lineno,status='source_match' if matches else ('source_diff' if differing else 'binding_not_directly_mapped'),counterparts=matches or differing))
        exports=[]
        for node in ast.walk(old_tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='savez_compressed':
                exports.append(dict(line=node.lineno,keys=[k.arg for k in node.keywords]))
        rows.append(dict(original_file=relative,sha256_unchanged=hashlib.sha256(path.read_bytes()).hexdigest()==digest,
                         scope=scope,new_locations=candidates,definitions=rules,constants=constants,
                         original_cli=options(old_tree),original_export_calls=exports))
    counts={}
    for file in rows:
        for rule in file['definitions']:
            key=rule['status'];counts[key]=counts.get(key,0)+1
    mapped={p for paths in TARGETS.values() for p in paths}
    return dict(all_rules_identical=False,original_files=len(rows),new_python_files=len(new_paths),
                source_definition_counts=counts,
                warning='Normalized source equality is not whole-file rule equality; defaults, bindings, driver branches and output schema are separate rules.',
                new_cli=options(modules['generate.py'][0]),files=rows,
                new_glue_or_package_files=[str(p.relative_to(DATA)) for p in new_paths if str(p.relative_to(DATA)) not in mapped])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--markdown-output',help='Optional full file/function inventory')
    args=parser.parse_args()
    report=inventory()
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if args.markdown_output:
        lines=['# 新旧文件与函数规则清单','',
               '这是逐文件出处清单，不是“全部规则等价”的证明。默认值、全局绑定、生成主循环和导出规则另见 [逐文件结论](DATA_GENERATION_FILE_RULES_2026-10-20.md)。',
               '',f"原文件 {report['original_files']} 个，新 Python 文件 {report['new_python_files']} 个。",'',
               '## 原文件覆盖','',
               '| 原文件 | 范围 | 源码匹配定义数 | 需文字解释定义数 | 原文件 SHA256 |',
               '| --- | --- | ---: | ---: | --- |']
        for file in report['files']:
            matches=sum(r['status']=='normalized_source_match' for r in file['definitions'])
            changed=sum('requires_rule_explanation' in r['status'] for r in file['definitions'])
            lines.append(f"| `{file['original_file']}` | {file['scope']} | {matches} | {changed} | {'未变' if file['sha256_unchanged'] else '已变'} |")
        lines+=['','## 迁移定义逐项对应','',
                '仅忽略 docstring、移动的 import、显式 ma 后端注入及 MASS 可选 RNG 接口。source_match 表示该定义结构相同，不表示它依赖的全局对象、调用顺序或整文件默认行为已相同。类按顶层类计，方法包含在类结构比较中。','',
                '| 原定义 | 新定义 | 结构结论 |','| --- | --- | --- |']
        for file in report['files']:
            if file['scope']!='migrated':continue
            for rule in file['definitions']:
                old=f"{file['original_file']}:{rule['line']} `{rule['name']}`"
                new=f"data/{rule['new_file']}:{rule['new_line']}" if rule['new_file'] else '公共 driver / worker / IO，需文字解释'
                lines.append(f"| {old} | {new} | {rule['status']} |")
        lines+=['','## 每个新文件的来源','',
                '| 新文件 | 对应旧文件 |','| --- | --- |']
        paths=sorted([DATA/'generate.py']+[p for directory in ('systems','sampling','generation','recipes') for p in (DATA/directory).rglob('*.py')])
        for file in paths:
            relative=str(file.relative_to(DATA))
            sources=[name for name,targets in TARGETS.items() if relative in targets]
            origin='、'.join(f'`{s}`' for s in sources) if sources else '新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义'
            lines.append(f'| `data/{relative}` | {origin} |')
        lines+=['','## 读取方式','',
                '`migrated` 是迁移的 17 个生成文件；`excluded_obsolete` 是不迁移的 LHS/real_mutants_only；`consumer_migrated` 是已切换到新接口的辅助模块；`original_module_unchanged` 是原样保留的其他工具模块。',
                '',f"结构计数：`{json.dumps(report['source_definition_counts'],ensure_ascii=False)}`。",'',
                '完整 JSON 同时记录旧函数默认值、每个 argparse 选项的原默认表达式、规则常量及旧 NPZ 键集合。', '']
        md=Path(args.markdown_output);md.parent.mkdir(parents=True,exist_ok=True);md.write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','new_cli','new_glue_or_package_files')},ensure_ascii=False))


if __name__=='__main__':main()
