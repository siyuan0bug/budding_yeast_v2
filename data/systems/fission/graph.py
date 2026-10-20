"""Fission potential dependencies extracted from its own ODE and event source.

Set propagation follows scalar intermediates and control conditions, without
executing source or consulting mutants. Dynamic kS effects are represented
separately from actual concentration resets. Matrices use A[target, source].
"""
import ast
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np

from ...yeast_graph import (YeastGraph,EquationDependency,ODETermDependency,
                           EventDependency,GraphExtractionError,_flux_terms)
from .constants import VAR_NAMES


@dataclass(frozen=True)
class Value:
    sources: frozenset = frozenset()
    names: frozenset = frozenset()
    parameters: frozenset = frozenset()


def union(*values):
    return Value(*(frozenset().union(*(getattr(v,key) for v in values))
                   for key in ('sources','names','parameters')))


@dataclass(frozen=True)
class FissionGraph(YeastGraph):
    latent_events: tuple = ()

    def adjacency(self,view,*,self_loops='physical'):
        matrix=super().adjacency(view,self_loops=self_loops)
        for event_id,targets,control_sources in self.latent_events:
            event=self.events[event_id-1]
            sources=set()
            if view in ('event_trigger','event','hybrid'):sources.update(event.guard_sources)
            if view in ('reset_value','event','hybrid'):sources.update(control_sources)
            for target in targets:matrix[target,list(sources)]=True
        if self_loops!='physical':np.fill_diagonal(matrix,self_loops=='all')
        return matrix

    def summary(self):
        result=super().summary()
        result['system_id']='fission'
        result['latent_parameter_events']=[
            {'event':self.events[e-1].name,'parameter':'kS',
             'affected_derivatives':[self.variable_names[t] for t in targets]}
            for e,targets,_ in self.latent_events]
        return result


@lru_cache(maxsize=4)
def _build(source,path):
    tree=ast.parse(source,filename=path)
    functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    if not {'eqns','run_event_sim','ev_start','ev_sfinish','ev_division'}<=functions.keys():
        raise GraphExtractionError('Fission graph requires its ODE, event guards and run_event_sim source')
    indices={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Tuple) and isinstance(node.value,ast.Call) and ast.unparse(node.value)=='range(16)':
            indices.update({n.id:i for i,n in enumerate(node.targets[0].elts)})
    if len(indices)!=16:raise GraphExtractionError('Fission state indices must declare range(16)')

    def dependencies(node,env):
        if isinstance(node,ast.Name):
            return union(env.get(node.id,Value()),Value(names=frozenset([node.id])))
        if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name):
            if node.value.id in ('y','ye','y0') and isinstance(node.slice,ast.Name) and node.slice.id in indices:
                return Value(sources=frozenset([indices[node.slice.id]]))
            if node.value.id=='p' and isinstance(node.slice,ast.Constant):
                return Value(parameters=frozenset([node.slice.value]))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in functions:
            helper=functions[node.func.id]
            local=dict(env)
            local.update({a.arg:dependencies(v,env) for a,v in zip(helper.args.args,node.args)})
            result=Value()
            for statement in helper.body:
                if isinstance(statement,ast.Assign) and isinstance(statement.targets[0],ast.Name):
                    local[statement.targets[0].id]=dependencies(statement.value,local)
                elif isinstance(statement,ast.Return):result=dependencies(statement.value,local)
            return result
        return union(*(dependencies(child,env) for child in ast.iter_child_nodes(node)))

    eqns=functions['eqns'];env={}
    unpack=next(n for n in eqns.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple))
    names=tuple(n.id for n in unpack.targets[0].elts)
    if names!=tuple(VAR_NAMES):raise GraphExtractionError('Fission ODE state order differs from catalog')
    env.update({name:Value(sources=frozenset([i])) for i,name in enumerate(names)})
    derivative_nodes={}
    for node in eqns.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            env[node.targets[0].id]=dependencies(node.value,env)
            derivative_nodes[node.targets[0].id]=node
    ret=next(n for n in eqns.body if isinstance(n,ast.Return))
    derivatives=ret.value.args[0].elts
    if len(derivatives)!=16:raise GraphExtractionError('Fission RHS must return 16 derivatives')
    equations=[];terms=[];ks_targets=[]
    for i,derivative in enumerate(derivatives):
        node=derivative_nodes[derivative.id];value=env[derivative.id]
        equations.append(EquationDependency(i,tuple(sorted(value.sources)),ast.unparse(node.value),(node.lineno,)))
        if 'kS' in value.parameters:ks_targets.append(i)
        for term in _flux_terms(node.value):
            v=dependencies(term,env)
            terms.append(ODETermDependency(i,tuple(sorted(v.sources)),ast.unparse(term),
                         tuple(sorted(v.names)),tuple(sorted(v.parameters)),node.lineno))

    dispatch=next(n for n in ast.walk(functions['run_event_sim']) if isinstance(n,ast.If) and ast.unparse(n.test)=="name == 'DIV'")
    start=dispatch.orelse[0]
    branches={'ev_division':dispatch.body,'ev_start':start.body,'ev_sfinish':start.orelse}
    events=[];latent=[]
    for event_id,name in enumerate(('ev_start','ev_sfinish','ev_division'),1):
        function=functions[name]
        guard=next(n for n in function.body if isinstance(n,ast.Return))
        guard_value=dependencies(guard.value,{})
        resets={};parameter_controls=[]
        def visit(statements,local,control=Value()):
            for node in statements:
                if isinstance(node,ast.If):
                    conditional=union(control,dependencies(node.test,local))
                    visit(node.body,dict(local),conditional);visit(node.orelse,dict(local),conditional)
                elif isinstance(node,(ast.Assign,ast.AugAssign)):
                    targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                    value=union(dependencies(node.value,local),control)
                    for target in targets:
                        if isinstance(target,ast.Name):local[target.id]=value
                        elif isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name):
                            if target.value.id=='ye' and isinstance(target.slice,ast.Name):
                                index=indices[target.slice.id]
                                if isinstance(node,ast.AugAssign):value=union(value,Value(sources=frozenset([index])))
                                resets[index]=union(resets.get(index,Value()),value)
                            elif target.value.id=='p' and isinstance(target.slice,ast.Constant) and target.slice.value=='kS':
                                parameter_controls.append(value)
        visit(branches[name],{})
        attrs={}
        for node in tree.body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Attribute) and ast.unparse(node.targets[0].value)==name:
                attrs[node.targets[0].attr]=ast.literal_eval(node.value)
        records=tuple(EquationDependency(i,tuple(sorted(v.sources)),'event reset',()) for i,v in sorted(resets.items()))
        events.append(EventDependency(event_id,name,ast.unparse(guard.value),tuple(sorted(guard_value.sources)),
                       attrs['direction'],attrs['terminal'],(guard.lineno,),records))
        if parameter_controls:
            latent.append((event_id,tuple(ks_targets),tuple(sorted(union(*parameter_controls).sources))))
    fingerprint=sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    return FissionGraph(tuple(names),tuple(equations),tuple(events),path,fingerprint,(),terms=tuple(terms),latent_events=tuple(latent))


def build_fission_graph(source_path=None):
    root=Path(source_path) if source_path else Path(__file__).resolve().parent
    source=('\n\n'.join((root/name).read_text() for name in ('constants.py','ode.py','simulation.py'))
            if root.is_dir() else root.read_text())
    return _build(source,str(root.resolve()))
