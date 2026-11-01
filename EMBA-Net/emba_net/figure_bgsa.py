"""Equations (1)-(5): aligned features, joint gates and simultaneous updates.

The common hidden width is half the output width, retaining external channels.
The same psi is used for the two ordered concatenations, as in equation (3).
"""
import torch
from torch import nn
from torch.nn import functional as F
from .common import Conv


class BGSA(nn.Module):
    def __init__(self,inc,input_dim=64):
        super().__init__()
        self.input_dim=input_dim;self.common_channels=input_dim//2
        self.fc1=nn.Conv2d(inc[1],self.common_channels,1,bias=False)
        self.fc2=nn.Conv2d(inc[0],self.common_channels,1,bias=False)
        self.psi=nn.Conv2d(2*self.common_channels,self.common_channels,1,bias=True)
        self.conv=Conv(2*self.common_channels,input_dim,3)

    def forward(self,x):
        source,target=x
        source=F.interpolate(source,size=target.shape[-2:],mode='bilinear',align_corners=False)
        shallow=self.fc1(target);deep=self.fc2(source)
        gs=self.psi(torch.cat((shallow,deep),dim=1)).sigmoid()
        gd=self.psi(torch.cat((deep,shallow),dim=1)).sigmoid()
        refined_shallow=shallow+gs*deep
        refined_deep=deep+gd*shallow
        return self.conv(torch.cat((refined_shallow,refined_deep),dim=1))
