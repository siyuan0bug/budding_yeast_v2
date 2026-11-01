"""Vectorize the CURRENT canonical NumPy RHS without copying its equations.

Only array layout, backend functions and scalar if-expressions are adapted.
Numerical constants, algebra and biological flag rules come from systems/budding.
The original generator, solver and existing PINN implementations are untouched.
"""
import ast
import hashlib
import inspect
import math
from types import SimpleNamespace
import torch
from budding_yeast_v2.data.systems.budding import ode
from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS


def _maximum(a,b):
    tensor=a if isinstance(a,torch.Tensor) else b
    return torch.maximum(torch.as_tensor(a,device=tensor.device,dtype=tensor.dtype),
                         torch.as_tensor(b,device=tensor.device,dtype=tensor.dtype))


def _where(condition,a,b):
    tensor=next(v for v in (a,b,condition) if isinstance(v,torch.Tensor))
    return torch.where(torch.as_tensor(condition,device=tensor.device,dtype=torch.bool),a,b)


class _Vectorize(ast.NodeTransformer):
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='y' and isinstance(node.targets[0],ast.Tuple):
            node.value=ast.Call(ast.Attribute(ast.Name('torch',ast.Load()),'unbind',ast.Load()),
                                [ast.Name('y',ast.Load())],[ast.keyword('dim',ast.Constant(-1))])
        return node

    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='np' and node.func.attr=='zeros':
            if ast.literal_eval(node.args[0])!=39:raise ValueError('Unsupported canonical RHS allocation')
            return ast.copy_location(ast.Call(ast.Attribute(ast.Name('torch',ast.Load()),'zeros_like',ast.Load()),
                                              [ast.Name('y',ast.Load())],[]),node)
        return node

    def visit_Subscript(self,node):
        node=self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='dydt':
            node.slice=ast.Tuple([ast.Constant(Ellipsis),node.slice],ast.Load())
        return node

    def visit_BoolOp(self,node):
        values=[self.visit(v) for v in node.values];op=ast.BitAnd if isinstance(node.op,ast.And) else ast.BitOr
        value=values[0]
        for other in values[1:]:value=ast.BinOp(value,op(),other)
        return ast.copy_location(value,node)

    def visit_IfExp(self,node):
        node=self.generic_visit(node)
        return ast.copy_location(ast.Call(ast.Name('_where',ast.Load()),[node.test,node.body,node.orelse],[]),node)

    def visit_If(self,node):
        node=self.generic_visit(node)
        if (len(node.body)!=1 or len(node.orelse)!=1 or not isinstance(node.body[0],ast.Assign)
            or not isinstance(node.orelse[0],ast.Assign) or len(node.body[0].targets)!=1
            or len(node.orelse[0].targets)!=1 or ast.dump(node.body[0].targets[0])!=ast.dump(node.orelse[0].targets[0])):
            raise ValueError('Unsupported canonical RHS conditional; review adapter before use')
        return ast.copy_location(ast.Assign(node.body[0].targets,
            ast.Call(ast.Name('_where',ast.Load()),[node.test,node.body[0].value,node.orelse[0].value],[])),node)


SOURCE=inspect.getsource(ode.GK)+'\n'+inspect.getsource(ode.eqns)
SOURCE_SHA256=hashlib.sha256(SOURCE.encode()).hexdigest()
namespace=dict(torch=torch,_where=_where,np=SimpleNamespace(
    sqrt=torch.sqrt,maximum=_maximum,log=lambda x:torch.log(x) if isinstance(x,torch.Tensor) else math.log(x)))
tree=ast.fix_missing_locations(_Vectorize().visit(ast.parse(SOURCE)))
exec(compile(tree,'<canonical_budding_torch_rhs>','exec'),namespace)


def rhs(y,p,t,flags=None):
    """y: B,[T,]39, p: B,141, t in physical minutes; autograd-compatible."""
    if y.shape[-1]!=39 or p.shape[-1]!=len(PARAM_KEYS):raise ValueError('Canonical RHS dimension mismatch')
    suffix=(1,)*(y.ndim-p.ndim)
    values={key:p[...,i].reshape(p.shape[:-1]+suffix) for i,key in enumerate(PARAM_KEYS)}
    for key in ('is_sbf_ca','is_mutant_104'):
        flag=torch.as_tensor((flags or {}).get(key,False),device=y.device,dtype=torch.bool)
        if flag.ndim and suffix:flag=flag.reshape(*flag.shape,*suffix)
        values[key]=flag
    return namespace['eqns'](torch.as_tensor(t,device=y.device,dtype=y.dtype),y,values)
