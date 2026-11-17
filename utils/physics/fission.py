"""Differentiable Fission RHS compiled from the canonical NumPy equation AST.

No second equation table is maintained. Exported mass is paper-relative;
latent kS is reconstructed from event history, beginning at kSmin.
"""
import ast
import inspect
from functools import lru_cache

import torch
import torch.nn.functional as F

from budding_yeast_v2.data.systems.fission import ode
from budding_yeast_v2.data.systems.fission.constants import MASS_UNIT
from budding_yeast_v2.data.systems.fission.parameters import EXPORT_KEYS


def _where(condition,yes,no):
    return torch.where(condition,torch.as_tensor(yes,device=condition.device),torch.as_tensor(no,device=condition.device))


def _maximum(a,b):
    ref=a if isinstance(a,torch.Tensor) else b
    return torch.maximum(torch.as_tensor(a,device=ref.device,dtype=ref.dtype),torch.as_tensor(b,device=ref.device,dtype=ref.dtype))


class TorchEquation(ast.NodeTransformer):
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if isinstance(node.targets[0],ast.Tuple) and isinstance(node.value,ast.Name) and node.value.id=='y':
            node.value=ast.parse('y.unbind(dim=-1)',mode='eval').body
        return node

    def visit_IfExp(self,node):
        return ast.copy_location(ast.Call(func=ast.Name(id='_where',ctx=ast.Load()),
            args=[self.visit(node.test),self.visit(node.body),self.visit(node.orelse)],keywords=[]),node)

    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='max':node.func.id='_maximum'
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='np':
            if node.func.attr=='array':
                node.func=ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='stack',ctx=ast.Load())
                node.keywords=[ast.keyword(arg='dim',value=ast.Constant(-1))]
            else:raise ValueError('Unexpected NumPy operation in Fission RHS')
        return node


@lru_cache(maxsize=1)
def _compiled_rhs():
    tree=TorchEquation().visit(ast.parse(inspect.getsource(ode.eqns)))
    ast.fix_missing_locations(tree)
    namespace={'torch':torch,'_where':_where,'_maximum':_maximum,
               '_Y_switch':lambda mpf:100.*torch.sigmoid(80.*(mpf-.2))}
    exec(compile(tree,ode.__file__+' [torch adapter]','exec'),namespace)
    return namespace['eqns']


def params_tensor_to_dict(vector):
    if vector.shape[-1]!=58:raise ValueError('Fission physics expects 58 ordered parameters')
    return {name:vector[...,i] for i,name in enumerate(EXPORT_KEYS)}


def phase_history(trajectory,parameters):
    """Reconstruct the solver's latent switch from a complete sampled target.

    Every exported run starts at kSmin, including cut strains whose initial
    RDNA may be below one. START is the only downward RDNA reset; reaching
    0.99999 finishes S phase. State-only RDNA<1 is insufficient for cut strains.
    """
    rdna=trajectory[:,1,:]
    start=torch.zeros_like(rdna,dtype=torch.bool)
    start[:,1:]=rdna[:,1:]<rdna[:,:-1]-1e-5
    finish=rdna>=.99999
    positions=torch.arange(rdna.shape[-1],device=rdna.device)[None,:].expand_as(rdna)
    latest_start=torch.where(start,positions,-1).cummax(-1).values
    latest_finish=torch.where(finish,positions,-1).cummax(-1).values
    high=latest_start>latest_finish
    return torch.where(high,parameters['kSmax'][:,None],parameters['kSmin'][:,None])


def eqns_torch(y,parameters,t=None,*,exported=True):
    if y.shape[-1]!=16:raise ValueError('Fission physics expects 16 ordered states')
    p=dict(parameters)
    if 'kS' not in p:
        raise ValueError('Fission RHS requires latent kS; reconstruct it from the complete target event history, not RDNA at one instant')
    # Batch scalar defaults are broadcast to the state batch shape.
    ref=y[...,0]
    p={key:torch.zeros_like(ref)+value for key,value in p.items()}
    if exported:
        scale=torch.ones(16,device=y.device,dtype=y.dtype);scale[0]=MASS_UNIT
        raw=y*scale
    else:raw=y
    result=_compiled_rhs()(t,raw,p)
    return result/scale if exported else result


def detect_event_mask(target,*,event_window=4,event_tol=.08,jump_ratio=.3,parameters=None):
    """True at continuous points; mask both sides of phase/division discontinuities.

    Uses target jumps, not predicted events. kS phase changes are included even
    if they do not create a large concentration jump. Dilation covers stencils.
    """
    batch,_,steps=target.shape
    jump=torch.zeros(batch,steps,device=target.device,dtype=torch.bool)
    difference=(target[:,:,1:]-target[:,:,:-1]).abs()
    scale=target.abs().amax(-1,keepdim=True).clamp_min(1e-6)
    adjacent=(difference/scale>jump_ratio).any(1)
    phase=target[:,1,:]<.99999
    adjacent |= phase[:,1:]!=phase[:,:-1]
    jump[:,1:] |= adjacent;jump[:,:-1] |= adjacent
    if parameters is not None:
        eps=parameters['eps']
        if isinstance(eps,torch.Tensor) and eps.ndim==1:eps=eps[:,None]
        ratio=(target[:,2,:]+eps*target[:,3,:])/(1.-target[:,15,:]).clamp_min(1e-6)
        jump |= (ratio-.65).abs()<event_tol
        jump |= (ratio-.5).abs()<event_tol
    if event_window>0:
        jump=F.max_pool1d(jump.float().unsqueeze(1),2*event_window+1,stride=1,padding=event_window).squeeze(1).bool()
    return ~jump


def apply_event_torch(y,parameters,event):
    """Functional canonical event resets in exported or raw units (mass halves).

    Return both state and parameters: START/SFINISH update latent kS. Numerical
    guards/micro-pushes match run_event_sim; callers select an event row mask.
    """
    states=list(y.clamp_min(0.).unbind(dim=-1));p=dict(parameters)
    states[1]=states[1].clamp_max(1.)
    ratio=(states[2]+p['eps']*states[3])/(1.-states[15]).clamp_min(1e-6)
    if event=='DIV':
        states[0]=states[0]/2.
        factor=torch.where(((1.-states[15])>1e-6)&(ratio>=.4995),torch.full_like(ratio,.9995),torch.ones_like(ratio))
        states[2]=states[2]*factor;states[3]=states[3]*factor
    elif event=='START':
        legal=states[1]>=1.
        states[1]=torch.where(legal,0.,states[1])
        old=p.get('kS',p['kSmin'])
        p['kS']=torch.where(legal,torch.as_tensor(p['kSmax'],device=y.device,dtype=y.dtype),torch.as_tensor(old,device=y.device,dtype=y.dtype))
        f=.6504/ratio.clamp_min(1e-12)
        f=torch.where((ratio<=.6502)&(f<=1.05),f,1.)
        states[2]=states[2]*f;states[3]=states[3]*f
    elif event=='SFINISH':
        states[1]=torch.ones_like(states[1]);p['kS']=p['kSmin']
    else:raise ValueError('Fission event must be START, SFINISH or DIV')
    return torch.stack(states,dim=-1),p
