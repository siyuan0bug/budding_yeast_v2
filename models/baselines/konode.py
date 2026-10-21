"""KoNODE augmented state/parameter/Koopman dynamics adapted to known ODE P.

Follows Bai & Ding's main.py: Koopman modes w evolve linearly, a hypernetwork
maps w to the weights theta(t) of the observed-state vector field. P is a
static additional input to that field. Initial w is deterministically encoded
from (y0,P), replacing random initialization for a deterministic surrogate.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint
from .common import joint_condition_grid


class KoNODE(nn.Module):
    def __init__(self, num_vars=39, param_dim=141, width=64, n_blocks=4,
                 koopman_dim=16, method='dopri5', rtol=1e-3, atol=1e-4):
        super().__init__()
        if koopman_dim % 2 or n_blocks < 2:
            raise ValueError('Koopman dimension must be even; vector field needs at least two layers')
        self.num_vars,self.param_dim,self.koopman_dim = num_vars,param_dim,koopman_dim
        self.method,self.rtol,self.atol = method,rtol,atol
        self.sizes = [num_vars+param_dim] + [width]*(n_blocks-1) + [num_vars]
        self.initial_modes = nn.Linear(num_vars+param_dim,koopman_dim)
        self.spectrum = nn.Parameter(torch.zeros(koopman_dim))
        self.encoder = nn.Sequential(nn.Linear(koopman_dim,50),nn.SiLU(),nn.Linear(50,n_blocks*50))
        self.weight_heads = nn.ModuleList()
        for i,(din,dout) in enumerate(zip(self.sizes[:-1],self.sizes[1:])):
            count = din*dout + dout
            if i == 0 or i == n_blocks - 1:
                head = nn.Sequential(nn.Linear(50,count),nn.SiLU())
            else:
                head = nn.Sequential(nn.Linear(50,50),nn.SiLU(),nn.Linear(50,count))
            self.weight_heads.append(head)
        # The source initializes the generated-weight networks with Kaiming;
        # the output stays unconstrained: no artificial negative spectrum clamp.

    def vector_field(self, y, w, params):
        chunks = self.encoder(w).split(50,dim=-1)
        value = torch.cat((y,params),dim=-1)
        for i,(head,chunk,din,dout) in enumerate(zip(self.weight_heads,chunks,self.sizes[:-1],self.sizes[1:])):
            theta = head(chunk)
            weight = theta[:,:din*dout].reshape(-1,din,dout)
            bias = theta[:,din*dout:]
            value = torch.bmm(value.unsqueeze(1),weight).squeeze(1) + bias
            if i < len(self.weight_heads)-1:
                value = F.relu(value)
        return value

    def mode_derivative(self, w):
        re,im = self.spectrum.chunk(2)
        pairs = w.reshape(w.shape[0],-1,2)
        rotation = torch.stack((pairs[...,1],-pairs[...,0]),dim=-1)
        return (re.view(1,-1,1)*pairs + im.view(1,-1,1)*rotation).flatten(1)

    def _solve(self,y0,params,time):
        w0 = self.initial_modes(torch.cat((y0,params),dim=-1)) * .01
        def rhs(t,state):
            y,w = state
            return self.vector_field(y,w,params),self.mode_derivative(w)
        y,_ = odeint(rhs,(y0,w0),time,method=self.method,rtol=self.rtol,atol=self.atol)
        return y.permute(1,2,0)

    def forward(self,x,params):
        joint_condition_grid(x,params,self.num_vars,self.param_dim,uniform=False)
        y0,time = x[:,:,0,0],x[:,0,1,:]
        if torch.equal(time,time[:1].expand_as(time)):
            return self._solve(y0,params,time[0])
        return torch.cat([self._solve(y0[i:i+1],params[i:i+1],time[i]) for i in range(len(x))])
