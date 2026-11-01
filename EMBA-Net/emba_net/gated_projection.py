"""Dimension-consistent implementation of Figure 4's right-hand Conv_GN.

Let C=2*out_channels. Proj(2C) splits a C/4 seed and a 7C/4
DS-convolution path. That path splits into C/4, C/2, C gates;
Proj(C/2), Proj(C), Proj(C/2) complete the three Mul stages.
The middle gate is C/2: the figure's repeated C/4 label cannot
multiply a C/2 feature elementwise. This interpretation is documented.
"""
import math
from torch import nn


class GatedProjectionGN(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        if c2%2:raise ValueError('Figure 4 output width must be even')
        self.C=2*c2
        self.gate_widths=(c2//2,c2,2*c2)
        context_channels=sum(self.gate_widths)
        self.proj_in=nn.Conv2d(c1,4*c2,1,bias=False)
        self.dsconv=nn.Sequential(
            nn.Conv2d(context_channels,context_channels,3,padding=1,groups=context_channels,bias=False),
            nn.BatchNorm2d(context_channels),nn.ReLU(),
            nn.Conv2d(context_channels,context_channels,1,bias=False),
            nn.BatchNorm2d(context_channels),nn.ReLU())
        self.proj_half=nn.Conv2d(c2//2,c2,1,bias=False)
        self.proj_full=nn.Conv2d(c2,2*c2,1,bias=False)
        self.proj_out=nn.Conv2d(2*c2,c2,1,bias=False)
        self.norm=nn.GroupNorm(math.gcd(16,c2),c2)
        self.act=nn.SiLU()

    def forward(self,x):
        seed,context=self.proj_in(x).split((self.gate_widths[0],sum(self.gate_widths)),dim=1)
        g0,g1,g2=self.dsconv(context).split(self.gate_widths,dim=1)
        z=seed*g0
        z=self.proj_half(z)*g1
        z=self.proj_full(z)*g2
        return self.act(self.norm(self.proj_out(z)))
