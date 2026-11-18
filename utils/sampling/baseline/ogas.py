"""Independent OGAS2026 conditional DDPM and reference-prior debiasing core.

Implements the conditional noise objective, rolling feedback, EMA signals,
discriminator density ratio, high-signal triangular proposal and exploration
mixture. Legal biological input projection and the training schedule are explicit
workflow adaptations, not changes to the canonical biological definitions.
"""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from torch import nn


OGAS_IMPLEMENTATION_VERSION = 'author_cfg_and_reference_provenance_v2'


@dataclass(frozen=True)
class OGASConfig:
    width: int = 128
    diffusion_steps: int = 1000
    sampling_steps: int = 50
    updates: int = 50
    batch_size: int = 128
    capacity: int = 20000
    learning_rate: float = 1e-4
    exploration: float = .1
    ema_decay: float = .99
    maximum_signal: float = 5.
    minimum_weight: float = .1
    maximum_weight: float = 10.
    condition_dropout: float = .1
    guidance_scale: float = 5.
    paper_version: str = 'arxiv_v1'
    discriminator_learning_rate: float = 1e-4
    discriminator_weight_decay: float = .01
    discriminator_logit_penalty: float = .03

    def validate(self):
        if min(self.width,self.diffusion_steps,self.sampling_steps,self.updates,self.batch_size,self.capacity)<1:
            raise ValueError('Invalid OGAS dimensions')
        vals=(self.learning_rate,self.exploration,self.ema_decay,self.maximum_signal,
              self.minimum_weight,self.maximum_weight,self.condition_dropout,self.guidance_scale)
        if not np.isfinite(vals).all() or self.learning_rate<=0 or not 0<self.exploration<1:
            raise ValueError('Invalid OGAS settings')
        if (not 0<=self.ema_decay<1 or min(vals[3:6])<=0 or self.minimum_weight>self.maximum_weight
                or not 0<=self.condition_dropout<1 or self.guidance_scale<0):
            raise ValueError('Invalid OGAS signal/weight bounds')
        if (self.paper_version not in ('arxiv_v1','arxiv_v2')
                or not np.isfinite((self.discriminator_learning_rate,self.discriminator_weight_decay,
                                    self.discriminator_logit_penalty)).all()
                or self.discriminator_learning_rate<=0
                or min(self.discriminator_weight_decay,self.discriminator_logit_penalty)<0):
            raise ValueError('Invalid OGAS version/discriminator settings')
        return self


def density_ratio(reference_probability,minimum=.1,maximum=10.):
    """Balanced discriminator P(reference|x) => q_reference(x)/p_buffer(x)."""
    p=torch.as_tensor(reference_probability)
    if not torch.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid discriminator probability')
    return (p/(1-p).clamp_min(torch.finfo(p.dtype).eps)).clamp(minimum,maximum)


def density_weights_v2(reference_logits,*,buffer_reference_fraction,discriminator_reference_prior=.5):
    """v2 Appendix B.2: log-ratio clip, prior cap, then unit batch mean.

The discriminator class prior and the actual reference share of the feedback
batch are distinct. The former corrects posterior odds; the latter caps weights.
As in the author code, an all-generated batch uses a one-observation floor.
"""
    logits=torch.as_tensor(reference_logits)
    if (logits.ndim!=1 or not len(logits) or not torch.isfinite(logits).all()
            or not np.isfinite((buffer_reference_fraction,discriminator_reference_prior)).all()
            or not 0<=buffer_reference_fraction<=1 or not 0<discriminator_reference_prior<1):
        raise ValueError('Invalid v2 density logits/priors')
    logratio=logits+np.log((1-discriminator_reference_prior)/discriminator_reference_prior)
    pi_u=max(float(buffer_reference_fraction),1/len(logits))
    weights=logratio.clamp(-5.,5.).exp().clamp(max=1/pi_u)
    return weights/weights.mean()


def guided_noise(conditional,unconditional,scale):
    """Author DDPM convention: w=0 is unconditional, w=1 is conditional."""
    return unconditional+scale*(conditional-unconditional)


class ConditionalDenoiser(nn.Module):
    def __init__(self,dimension,width,*,time_scale=1000,residual_modulation=True):
        super().__init__()
        self.width=width
        self.time_scale=time_scale;self.residual_modulation=residual_modulation
        embedding=4*width
        self.input=nn.Linear(dimension,width)
        self.time=nn.Sequential(nn.Linear(width,embedding),nn.SiLU(),nn.Linear(embedding,embedding))
        self.condition=nn.Sequential(nn.Linear(1,embedding),nn.SiLU(),nn.Linear(embedding,embedding))
        self.null=nn.Parameter(torch.zeros(embedding))
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.modulations=nn.ModuleList([nn.Linear(embedding,2*width) for _ in range(2)])
        self.layers=nn.ModuleList([nn.Linear(width,width) for _ in range(2)])
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,dimension))

    def forward(self,x,t,signal,drop=None):
        frequencies=torch.exp(-np.log(10000)*torch.arange(max(1,self.width//2),device=x.device)
                              /max(1,self.width//2-1))
        phase=t[:,None]*self.time_scale*frequencies[None]
        time=torch.cat((phase.sin(),phase.cos()),1)
        time=nn.functional.pad(time,(0,max(0,self.width-time.shape[1])))[:,:self.width]
        condition=self.condition(signal[:,None])
        if drop is not None:condition=torch.where(drop[:,None],self.null[None],condition)
        embedding=self.time(time)+condition;h=self.input(x)
        for norm,modulation,layer in zip(self.norms,self.modulations,self.layers):
            gamma,beta=modulation(embedding).chunk(2,dim=1)
            multiplier=1+gamma if self.residual_modulation else gamma
            h=h+layer(nn.functional.silu(multiplier*norm(h)+beta))
        return self.output(h)


class ConditionalOGAS:
    def __init__(self,initial_inputs,seed=42,config=OGASConfig(),device='cpu'):
        self.config=config.validate();self.device=torch.device(device)
        x=np.asarray(initial_inputs,dtype=np.float32)
        if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Need finite initial OGAS inputs')
        self.mean=x.mean(0);self.scale=x.std(0);self.scale=np.where(self.scale>1e-8,self.scale,1.)
        self.dimension=x.shape[1]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.generator=ConditionalDenoiser(self.dimension,config.width,time_scale=config.diffusion_steps,
                                              residual_modulation=config.paper_version=='arxiv_v1').to(device)
            self.discriminator=nn.Sequential(nn.Linear(self.dimension,64),nn.ReLU(),nn.Linear(64,1)).to(device)
        self.noise_rng=torch.Generator(device=self.device).manual_seed(int(seed))
        self.rng=np.random.default_rng(seed)
        self.optimizer=torch.optim.Adam(self.generator.parameters(),lr=config.learning_rate)
        self.discriminator_optimizer=(torch.optim.AdamW(self.discriminator.parameters(),
            lr=config.discriminator_learning_rate,weight_decay=config.discriminator_weight_decay)
            if config.paper_version=='arxiv_v2'
            else torch.optim.Adam(self.discriminator.parameters(),lr=config.learning_rate))
        self.beta=torch.linspace(1e-4,.02,config.diffusion_steps,device=device)
        self.alpha=1-self.beta;self.cumulative=torch.cumprod(self.alpha,0)
        self.inputs=np.empty((0,self.dimension),np.float32);self.signals=np.empty(0,np.float32)
        self.is_reference=np.empty(0,dtype=bool)
        self.signal_mean=0.;self.signal_scale=1.;self.cycles=0

    def normalize(self,x):
        x=np.asarray(x,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=self.dimension or not np.isfinite(x).all():raise ValueError('Invalid OGAS inputs')
        return torch.as_tensor((x-self.mean)/self.scale,device=self.device)

    def observe(self,inputs,difficulty,*,is_reference=None):
        x=np.asarray(inputs,dtype=np.float32);s=np.asarray(difficulty,dtype=np.float32)
        self.normalize(x)
        if s.shape!=(len(x),) or not np.isfinite(s).all() or (s<0).any():raise ValueError('Invalid OGAS feedback')
        if is_reference is None:
            if self.config.paper_version=='arxiv_v2':raise ValueError('OGAS v2 feedback requires reference provenance')
            is_reference=np.ones(len(x),dtype=bool)
        reference=np.asarray(is_reference)
        if reference.shape!=(len(x),) or reference.dtype!=np.bool_:
            raise ValueError('OGAS feedback reference provenance must be aligned boolean flags')
        if not len(x):return
        decay=self.config.ema_decay
        self.signal_mean=decay*self.signal_mean+(1-decay)*float(s.mean())
        self.signal_scale=decay*self.signal_scale+(1-decay)*float(s.std())
        standardized=(s-self.signal_mean)/max(self.signal_scale,1e-6)
        self.inputs=np.concatenate((self.inputs,x))[-self.config.capacity:]
        self.signals=np.concatenate((self.signals,standardized))[-self.config.capacity:]
        self.is_reference=np.concatenate((self.is_reference,reference))[-self.config.capacity:]

    def train_cycle(self,reference_inputs):
        if not len(self.inputs):raise ValueError('OGAS needs revealed feedback')
        reference=self.normalize(reference_inputs)
        if not len(reference):raise ValueError('Need nonempty legal reference proposal bank')
        x=self.normalize(self.inputs);signals=torch.tensor(self.signals,device=self.device)
        diagnostics={}
        for _ in range(self.config.updates):
            ids=torch.randint(len(x),(min(self.config.batch_size,len(x)),),device=self.device,generator=self.noise_rng)
            refs=torch.randint(len(reference),(len(ids),),device=self.device,generator=self.noise_rng)
            values=x[ids];condition=signals[ids]
            logits=self.discriminator(torch.cat((values,reference[refs]))).squeeze(1)
            labels=torch.cat((torch.zeros(len(ids),device=self.device),torch.ones(len(ids),device=self.device)))
            dloss=nn.functional.binary_cross_entropy_with_logits(logits,labels)
            if self.config.paper_version=='arxiv_v2':
                dloss=dloss+self.config.discriminator_logit_penalty*logits.square().mean()
            self.discriminator_optimizer.zero_grad();dloss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(),1.,error_if_nonfinite=True)
            self.discriminator_optimizer.step()
            with torch.no_grad():
                reference_logits=self.discriminator(values).squeeze(1)
                reference_fraction=float(self.is_reference[ids.cpu().numpy()].mean())
                weights=(density_weights_v2(reference_logits,buffer_reference_fraction=reference_fraction)
                    if self.config.paper_version=='arxiv_v2'
                    else density_ratio(reference_logits.sigmoid(),self.config.minimum_weight,self.config.maximum_weight))
            t=torch.randint(self.config.diffusion_steps,(len(ids),),device=self.device,generator=self.noise_rng)
            noise=torch.randn(values.shape,device=self.device,generator=self.noise_rng)
            abar=self.cumulative[t,None]
            perturbed=abar.sqrt()*values+(1-abar).sqrt()*noise
            drop=torch.rand(len(ids),device=self.device,generator=self.noise_rng)<self.config.condition_dropout
            estimated=self.generator(perturbed,t.float()/self.config.diffusion_steps,condition,drop)
            loss=(weights*(estimated-noise).square().mean(1)).mean()
            if not torch.isfinite(loss) or not torch.isfinite(dloss):raise FloatingPointError('Nonfinite OGAS objective')
            self.optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(self.generator.parameters(),1.,error_if_nonfinite=True)
            self.optimizer.step()
            diagnostics=dict(diffusion_loss=float(loss.detach()),discriminator_loss=float(dloss.detach()),
                weight_min=float(weights.min()),weight_max=float(weights.max()),
                batch_reference_fraction=reference_fraction,
                weight_cap_before_normalization=(1/max(reference_fraction,1/len(ids))
                    if self.config.paper_version=='arxiv_v2' else self.config.maximum_weight),
                discriminator_reference_prior=.5)
        self.cycles+=1
        return dict(diagnostics,cycle=self.cycles,feedback_rows=len(self.inputs),updates=self.config.updates,
                    buffer_reference_fraction=float(self.is_reference.mean()),
                    implementation_version=OGAS_IMPLEMENTATION_VERSION)

    @torch.no_grad()
    def sample_targets(self,count,reference_inputs,*,exploration=True):
        """Raw proposed input features. The domain must project before ODE use."""
        if count<1 or not self.cycles:raise ValueError('Train OGAS before sampling')
        reference=np.asarray(reference_inputs,dtype=np.float32)
        self.normalize(reference)
        if not len(reference):raise ValueError('Missing reference inputs')
        condition=self.config.maximum_signal*torch.rand(count,device=self.device,generator=self.noise_rng).sqrt()
        x=torch.randn((count,self.dimension),device=self.device,generator=self.noise_rng)
        schedule=np.unique(np.linspace(0,self.config.diffusion_steps-1,
                                       min(self.config.sampling_steps,self.config.diffusion_steps),dtype=int))[::-1]
        for position,i in enumerate(schedule):
            t=torch.full((count,),i/self.config.diffusion_steps,device=self.device)
            conditional=self.generator(x,t,condition)
            unconditional=self.generator(x,t,condition,torch.ones(count,dtype=torch.bool,device=self.device))
            noise=guided_noise(conditional,unconditional,self.config.guidance_scale)
            previous=(self.cumulative[schedule[position+1]] if position+1<len(schedule)
                      else self.cumulative.new_tensor(1.))
            alpha=self.cumulative[i]/previous;beta=1-alpha
            x=(x-beta*noise/(1-self.cumulative[i]).sqrt())/alpha.sqrt()
            if position+1<len(schedule):
                variance=beta*(1-previous)/(1-self.cumulative[i])
                x+=variance.sqrt()*torch.randn(x.shape,device=self.device,generator=self.noise_rng)
        values=x.cpu().numpy()*self.scale+self.mean
        explore=(self.rng.random(count)<self.config.exploration if exploration else np.zeros(count,dtype=bool))
        values[explore]=reference[self.rng.integers(len(reference),size=int(explore.sum()))]
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite OGAS proposal')
        return values,dict(exploration_count=int(explore.sum()),reference_flags=explore.tolist(),
                           maximum_target_signal=self.config.maximum_signal,
                           guidance_convention='unconditional + w * (conditional - unconditional)')

    def state_dict(self):
        return dict(implementation_version=OGAS_IMPLEMENTATION_VERSION,
            config=asdict(self.config),mean=self.mean,scale=self.scale,
            generator=self.generator.state_dict(),discriminator=self.discriminator.state_dict(),
            optimizer=self.optimizer.state_dict(),discriminator_optimizer=self.discriminator_optimizer.state_dict(),
            inputs=self.inputs,signals=self.signals,is_reference=self.is_reference,
            signal_mean=self.signal_mean,signal_scale=self.signal_scale,
            cycles=self.cycles,rng=self.rng.bit_generator.state,noise_rng=self.noise_rng.get_state())

    def load_state_dict(self,state):
        if state.get('implementation_version')!=OGAS_IMPLEMENTATION_VERSION:
            raise ValueError('OGAS implementation changed; legacy state lacks corrected sampling/provenance semantics')
        if state['config']!=asdict(self.config):raise ValueError('OGAS configuration changed')
        flags=np.asarray(state['is_reference'])
        if flags.dtype!=np.bool_ or flags.shape!=(len(state['inputs']),):
            raise ValueError('Invalid saved OGAS reference provenance')
        for key in ('mean','scale','inputs','signals','is_reference'):
            setattr(self,key,np.asarray(state[key]).copy())
        for key in ('generator','discriminator','optimizer','discriminator_optimizer'):
            getattr(self,key).load_state_dict(state[key])
        self.signal_mean=float(state['signal_mean']);self.signal_scale=float(state['signal_scale'])
        self.cycles=int(state['cycles']);self.rng.bit_generator.state=state['rng']
        self.noise_rng.set_state(state['noise_rng'].cpu())
