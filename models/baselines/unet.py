"""Plain 1D U-Net: double convolutions, pooling, transposed convolutions and skips.

The original U-Net encoder/decoder is reduced to the temporal axis, with same
padding and right padding/cropping for arbitrary trajectory lengths. Full y0,
P and query times are concatenated at the input. No added conditioning blocks.
"""
from torch import nn
from torch.nn import functional as F
import torch
from .common import joint_condition_grid


def double_conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv1d(in_channels,out_channels,3,padding=1),nn.ReLU(),
                         nn.Conv1d(out_channels,out_channels,3,padding=1),nn.ReLU())


class UNet(nn.Module):
    def __init__(self,num_vars=39,param_dim=141,width=64,n_blocks=4):
        super().__init__()
        if width < 1 or not 1 <= n_blocks <= 6:
            raise ValueError('UNet requires positive width and 1..6 encoder levels')
        self.num_vars,self.param_dim,self.n_blocks=num_vars,param_dim,n_blocks
        self.stem=double_conv(num_vars+param_dim+1,width)
        self.pool=nn.MaxPool1d(2)
        self.down=nn.ModuleList([double_conv(width*2**i,width*2**(i+1)) for i in range(n_blocks)])
        self.up=nn.ModuleList([nn.ConvTranspose1d(width*2**(i+1),width*2**i,2,stride=2)
                              for i in reversed(range(n_blocks))])
        self.decode=nn.ModuleList([double_conv(width*2**(i+1),width*2**i)
                                  for i in reversed(range(n_blocks))])
        self.head=nn.Conv1d(width,num_vars,1)

    def forward(self,x,params):
        h=joint_condition_grid(x,params,self.num_vars,self.param_dim)
        length=h.shape[-1]
        multiple=2**self.n_blocks
        h=F.pad(h,(0,(-length)%multiple))
        h=self.stem(h)
        skips=[]
        for block in self.down:
            skips.append(h)
            h=block(self.pool(h))
        for up,block,skip in zip(self.up,self.decode,reversed(skips)):
            h=block(torch.cat((up(h),skip),dim=1))
        return self.head(h)[...,:length]
