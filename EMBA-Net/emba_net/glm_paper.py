"""Figure 2: RepConv -> EMA -> split -> serial bottlenecks/taps -> RepConv.

The EMA bypass and each bottleneck output feed the final concatenation; the
pre-bottleneck seed is not an extra concatenation input in Figure 2.
"""
import math
import torch
from torch import nn
from .glm_pb import GLMContext,Bottleneck
from .repconv import RepConv


class GLM_P_B(nn.Module):
    def __init__(self,c1,c2,n=1,shortcut=False,g=1,e=0.5,rep_branches=1):
        super().__init__()
        self.c=int(round(c2*e))
        if self.c<1:raise ValueError("GLM hidden width must be positive")
        self.hidden_channels=self.c
        self.cv1=RepConv(c1,2*self.c,branches=rep_branches)
        if n<1:raise ValueError("Figure 2 requires at least one bottleneck")
        self.context=GLMContext(2*self.c,factor=math.gcd(8,2*self.c))
        self.m=nn.ModuleList(Bottleneck(self.c,self.c,shortcut,g,k=(3,3),e=1.0) for _ in range(n))
        self.cv2=RepConv((1+n)*self.c,c2,branches=rep_branches)

    def forward(self,x):
        bypass,local=self.context(self.cv1(x)).chunk(2,dim=1)
        taps=[bypass]
        for bottleneck in self.m:
            local=bottleneck(local)
            taps.append(local)
        return self.cv2(torch.cat(taps,dim=1))

    def forward_split(self,x):return self.forward(x)
