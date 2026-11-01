"""Experimental marker readout ablation; not a full hybrid ODE solver.

ORI/BUD/SPN follow their known linear driven equations between events. Drivers
come from BioRefinerPlus, guards use physical concentrations, and every downward
crossing can reset a marker again. No real labels or trajectories enter forward.
Event selection is piecewise differentiable: gradients pass through drivers and
crossing times inside a selected interval, not through crossing existence.
"""
from __future__ import annotations

import torch
from torch import nn

from budding_yeast_v2.data.normalization import (
    NORM_TYPES, check_norm_type, normalize_y, denormalize_y, denormalize_p,
)
from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
from .bio_refiner_plus import BioRefinerPlus
from .input_contracts import extract_initial_state


def affine_scan(a, b, initial):
    """Inclusive parallel scan of y[k+1]=a[k]*y[k]+b[k], O(log T) depth."""
    if a.shape != b.shape or initial.shape != a.shape[:-1]:
        raise ValueError('Affine scan requires matching (...,T) and initial (...)')
    stride = 1
    while stride < a.shape[-1]:
        a, b = (torch.cat((a[..., :stride], a[..., stride:] * a[..., :-stride]), -1),
                torch.cat((b[..., :stride], b[..., stride:] + a[..., stride:] * b[..., :-stride]), -1))
        stride *= 2
    return a * initial.unsqueeze(-1) + b


def linear_forcing_step(u0, u1, rate, dt):
    """Exact y'=u(t)-rate*y for a linearly interpolated u on one interval."""
    z = rate * dt
    small = z.abs() < 1e-3
    safe = torch.where(small, torch.ones_like(z), z)
    phi1 = torch.where(small, 1-z/2+z*z/6-z**3/24, -torch.expm1(-z)/safe)
    phi2 = torch.where(small, .5-z/6+z*z/24-z**3/120, (1-phi1)/safe)
    return torch.exp(-z), dt * ((phi1-phi2)*u0 + phi2*u1)


def downward_crossings(signal, threshold):
    """Guard crossing and linear crossing fraction in each sample interval."""
    left, right = signal[..., :-1], signal[..., 1:]
    crossing = (left > threshold) & (right <= threshold)
    difference = left-right
    denominator = torch.where(crossing, difference, torch.ones_like(difference))
    fraction = torch.where(crossing, (left-threshold)/denominator, torch.ones_like(left))
    return crossing, fraction.clamp(0, 1)


def marker_closure(trajectory, parameters, initial, times, *, reset_override=None):
    """Return physical (ORI,BUD,SPN) on a physical, increasing time grid.

    reset_override is exclusively an oracle diagnosis hook: (mask,fraction), each
    (B,3,T-1). The registered model never supplies it. At most one crossing of
    each guard per query interval is resolved; subgrid events can be missed.
    """
    if trajectory.ndim != 3 or trajectory.shape[1] != 39:
        raise ValueError('Expected physical trajectory (B,39,T)')
    batch, _, steps = trajectory.shape
    if parameters.shape != (batch, len(PARAM_KEYS)) or initial.shape != (batch,39):
        raise ValueError('Expected parameters (B,141) and initial (B,39)')
    if times.shape != (batch, steps) or not torch.isfinite(times).all() or (times.diff() <= 0).any():
        raise ValueError('Expected finite strictly increasing physical times (B,T)')
    if not torch.isfinite(parameters).all() or (parameters < 0).any():
        raise ValueError('Marker closure requires finite nonnegative physical parameters')
    idx = {name:i for i,name in enumerate(PARAM_KEYS)}
    def p(name):
        return parameters[:,idx[name],None]
    mass, cln2, clb2, clb5 = (trajectory[:,i] for i in (0,1,2,3))
    cln3 = p('C0')*p('Dn3')*mass/(p('Jn3')+p('Dn3')*mass).clamp_min(1e-12)
    source = torch.stack((
        p('ksori')*(p('eorib5')*clb5+p('eorib2')*clb2),
        p('ksbud')*(p('ebudn2')*cln2+p('ebudn3')*cln3+p('ebudb5')*clb5),
        p('ksspn')*clb2/(p('Jspn')+clb2).clamp_min(1e-12)), 1)
    rate = torch.stack((p('kdori'),p('kdbud'),p('kdspn')),1)
    dt = times.diff().unsqueeze(1)
    if reset_override is None:
        exit_mask,exit_fraction = downward_crossings(clb2, .3)
        license_mask,license_fraction = downward_crossings(clb2+clb5, .2)
        reset = torch.stack((license_mask,exit_mask,exit_mask),1)
        fraction = torch.stack((license_fraction,exit_fraction,exit_fraction),1)
    else:
        reset,fraction = reset_override
        if reset.shape != (batch,3,max(steps-1,0)) or fraction.shape != reset.shape:
            raise ValueError('Invalid diagnostic reset schedule')
    u0,u1 = source[...,:-1],source[...,1:]
    a,b = linear_forcing_step(u0,u1,rate,dt)
    at_crossing = u0+fraction*(u1-u0)
    _,after_reset = linear_forcing_step(at_crossing,u1,rate,dt*(1-fraction))
    a = torch.where(reset,torch.zeros_like(a),a)
    b = torch.where(reset,after_reset,b)
    y0 = initial[:,[33,34,35]]
    return torch.cat((y0.unsqueeze(-1),affine_scan(a,b,y0)), -1)


class BioRefinerMarkerClosure(BioRefinerPlus):
    """BioRefinerPlus with a parameter-free, physical marker readout.

    Retains the global backbone. This ablation does not repair its false START
    predictions or enforce consistency of all 39 coupled equations.
    """
    def __init__(self, num_vars=39, param_dim=141, **kwargs):
        if num_vars != 39 or param_dim != len(PARAM_KEYS):
            raise ValueError('Marker closure currently supports Budding 39/141 only')
        super().__init__(num_vars=num_vars,param_dim=param_dim,**kwargs)
        for name,value in (
            ('mean_y',torch.zeros(1,39,1)), ('std_y',torch.ones(1,39,1)),
            ('s_c',torch.ones(1,39,1)), ('p_mean',torch.zeros(1,param_dim)),
            ('p_std',torch.ones(1,param_dim)), ('t_max',torch.tensor(210.)),
            ('norm_code',torch.tensor(-1,dtype=torch.long))):
            self.register_buffer(name,value)

    def set_stats(self,mean_y,std_y,p_mean,p_std,t_max=210.,*,norm_type='log1p',s_c=None):
        check_norm_type(norm_type)
        if norm_type=='asinh_scale' and s_c is None:
            raise ValueError('asinh_scale requires s_c')
        for name,value in (('mean_y',mean_y),('std_y',std_y),('p_mean',p_mean),('p_std',p_std)):
            buffer=getattr(self,name)
            value=torch.as_tensor(value,device=buffer.device,dtype=buffer.dtype).reshape_as(buffer)
            if not torch.isfinite(value).all():
                raise ValueError(f'Invalid {name}')
            buffer.copy_(value)
        if (self.std_y <= 0).any() or (self.p_std < 0).any():
            raise ValueError('Invalid normalization scales')
        if norm_type in ('log1p','linear') and (self.p_std <= 0).any():
            raise ValueError('Affine parameter scales must be positive')
        if not 0 < float(t_max) < float('inf'):
            raise ValueError('Invalid horizon')
        self.s_c.copy_(torch.ones_like(self.s_c) if s_c is None else torch.as_tensor(s_c).to(self.s_c).reshape_as(self.s_c))
        if not torch.isfinite(self.s_c).all() or (self.s_c <= 0).any():
            raise ValueError('Invalid state scales')
        self.t_max.fill_(float(t_max))
        self.norm_code.fill_(NORM_TYPES.index(norm_type))

    def forward(self,x,p,return_coarse=False):
        if int(self.norm_code) < 0:
            raise RuntimeError('Bind DataModule/checkpoint statistics using set_stats before forward')
        if x.ndim != 4 or x.shape[1:3] != (39,2):
            raise ValueError('Marker closure requires explicit (B,39,2,T) IC/time input')
        if not torch.allclose(x[:,:,1],x[:,:1,1].expand_as(x[:,:,1])):
            raise ValueError('All species must share the query grid')
        if not torch.allclose(x[:,0,1,0],torch.zeros_like(x[:,0,1,0])):
            raise ValueError('Query grid must start at the supplied IC time zero')
        base,coarse = super().forward(x,p,return_coarse=True)
        # The closure and long scans run in float32 under mixed precision.
        dtype = torch.float64 if base.dtype==torch.float64 else torch.float32
        with torch.autocast(device_type=base.device.type,enabled=False):
            norm_type=NORM_TYPES[int(self.norm_code)]
            physical=denormalize_y(base.to(dtype),self.mean_y,self.std_y,norm_type,self.s_c)
            ic=denormalize_y(extract_initial_state(x,39).squeeze(-1).to(dtype),self.mean_y,self.std_y,norm_type,self.s_c)
            params=denormalize_p(p.to(dtype),self.p_mean,self.p_std,norm_type).clamp_min(0)
            markers=marker_closure(physical,params,ic,x[:,0,1].to(dtype)*self.t_max)
            # Only marker channels change; no physical clipping of other outputs.
            marker_states=physical.clone()
            marker_states[:,33:36]=markers
            normalized=normalize_y(marker_states,self.mean_y,self.std_y,norm_type,self.s_c)
            out=torch.cat((base[:,:33],normalized[:,33:36].to(base.dtype),base[:,36:]),1)
        return (out,coarse) if return_coarse else out
