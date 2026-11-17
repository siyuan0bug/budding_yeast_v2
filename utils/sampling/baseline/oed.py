"""Independent Xu2026 Bayesian informativeness model (Eq.5/8, Algorithm 1).

The acquisition core is domain independent. The explicitly named temporal
adaptation below uses variance/spectral content, not an invented yeast energy.
No simulator, mutation rule, held-out label, or reference repository is accessed.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OEDConfig:
    projection_dim: int = 16
    noise_variance: float = 1.
    prior_variance: float = 1.
    exploration: float = 1.
    variance_weight: float = 1.
    spectral_weight: float = 1.

    def validate(self):
        values=(self.noise_variance,self.prior_variance,self.exploration,
                self.variance_weight,self.spectral_weight)
        if self.projection_dim<1 or not np.isfinite(values).all():raise ValueError('Invalid OED settings')
        if min(values[:2])<=0 or min(values[2:])<0 or sum(values[3:])<=0:
            raise ValueError('Invalid OED scales/weights')
        return self


def temporal_informativeness(trajectories, *, channel_scale=None, config=OEDConfig()):
    """ODE adaptation of solution variance and high-frequency energy fraction.

    Time is the functional axis; there is no spatial energy dissipation in this
    task. The caller freezes channel scales from the initial *training* cohort.
    True trajectories can only enter after an actual authorized solver query.
    """
    config.validate()
    y=np.asarray(trajectories,dtype=np.float64)
    if y.ndim!=3 or y.shape[-1]<4 or not np.isfinite(y).all():raise ValueError('Invalid OED observations')
    scale=np.ones(y.shape[1]) if channel_scale is None else np.asarray(channel_scale)
    if scale.shape!=(y.shape[1],) or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Invalid frozen informativeness scales')
    z=y/scale[None,:,None]
    centered=z-z.mean(-1,keepdims=True)
    variance=np.mean(centered**2,axis=(1,2))
    power=np.abs(np.fft.rfft(centered,axis=-1))**2
    split=max(1,power.shape[-1]//2)
    fraction=np.divide(power[:,:,split:].sum(-1),power.sum(-1),
                       out=np.zeros(power.shape[:2]),where=power.sum(-1)>1e-24).mean(1)
    value=config.variance_weight*np.log1p(variance)+config.spectral_weight*fraction
    if not np.isfinite(value).all():raise FloatingPointError('Nonfinite OED informativeness')
    return value


class BayesianOED:
    """Conjugate Gaussian posterior; true observations update it sequentially."""
    def __init__(self, initial_inputs, seed=42, config=OEDConfig()):
        self.config=config.validate()
        x=np.asarray(initial_inputs,dtype=np.float64)
        if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Need finite initial inputs')
        self.mean=x.mean(0);self.scale=x.std(0)
        self.scale=np.where(self.scale>1e-8,self.scale,1.)
        rng=np.random.default_rng(seed)
        self.projection=rng.standard_normal((x.shape[1],config.projection_dim))/np.sqrt(x.shape[1])
        dim=1+6*config.projection_dim
        self.covariance=np.eye(dim)*config.prior_variance
        self.coefficients=np.zeros(dim)
        self.observations=0

    def features(self, inputs):
        x=np.asarray(inputs,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Invalid OED feature inputs')
        z=((x-self.mean)/self.scale)@self.projection
        # Bounded polynomial/trigonometric dictionary: geometry only, never
        # clips/modifies the physical candidate passed to the ODE.
        z=z/np.sqrt(1+z*z)
        return np.column_stack((np.ones(len(x)),z,z*z,z*z*z,
                                np.sin(np.pi*z),np.cos(np.pi*z),np.exp(-z*z)))

    def observe(self, inputs, values):
        features=self.features(inputs);values=np.asarray(values,dtype=np.float64)
        if values.shape!=(len(features),) or not np.isfinite(values).all():raise ValueError('Invalid OED targets')
        for phi,value in zip(features,values):
            projected=self.covariance@phi
            denominator=self.config.noise_variance+phi@projected
            if denominator<=0 or not np.isfinite(denominator):raise FloatingPointError('Invalid OED posterior')
            self.coefficients+=projected*(value-phi@self.coefficients)/denominator
            self.covariance-=np.outer(projected,projected)/denominator
            self.covariance=(self.covariance+self.covariance.T)/2
            self.observations+=1

    def scores(self, inputs):
        phi=self.features(inputs)
        variance=np.einsum('ij,jk,ik->i',phi,self.covariance,phi)
        if (variance < -1e-7).any():raise FloatingPointError('OED covariance lost positive semidefiniteness')
        k=max(2,self.observations+1)
        coefficient=self.config.exploration*np.sqrt(np.log(k)/k)
        scores=phi@self.coefficients+coefficient*np.sqrt(np.maximum(variance,0.))
        if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite OED score')
        return scores

    def state_dict(self):
        return dict(mean=self.mean,scale=self.scale,projection=self.projection,
                    covariance=self.covariance,coefficients=self.coefficients,observations=self.observations)

    def load_state_dict(self,state):
        for key in ('mean','scale','projection','covariance','coefficients'):
            value=np.asarray(state[key],dtype=np.float64)
            if value.shape!=getattr(self,key).shape or not np.isfinite(value).all():raise ValueError('Changed OED state')
            setattr(self,key,value.copy())
        self.observations=int(state['observations'])
