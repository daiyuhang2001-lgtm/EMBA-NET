"""Figure 4: gated Conv_GN, GAP/Softmax, shared DEConv and DW/PW decoders.

Each level stays on its native grid. No P3 common latent or resize/residual
paths from the previous draft are used. GAP gives per-channel scale weights.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .lsd_head import LSDheadreplacement as OriginalHead,DEConv_GN,Scale,Conv_GN
from .common import DFL
from .gated_projection import GatedProjectionGN


class DepthwiseSeparable(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.dw=nn.Conv2d(c,c,3,padding=1,groups=c,bias=False)
        self.dw_bn=nn.BatchNorm2d(c)
        self.pw=nn.Conv2d(c,c,1,bias=False)
        self.pw_bn=nn.BatchNorm2d(c);self.act=nn.ReLU()
    def forward(self,x):
        return self.act(self.pw_bn(self.pw(self.act(self.dw_bn(self.dw(x))))))


class LSDheadreplacement(OriginalHead):
    def __init__(self,nc=80,hidc=256,ch=()):
        nn.Module.__init__(self)
        self.nc,self.nl,self.reg_max=nc,len(ch),16
        self.no=nc+4*self.reg_max;self.stride=torch.zeros(self.nl);self.hidc=hidc
        self.conv=nn.ModuleList(GatedProjectionGN(c,hidc) for c in ch)
        if hidc%16:raise ValueError("Shared DEConv GroupNorm requires hidc divisible by 16")
        self.share_conv=nn.Sequential(Conv_GN(hidc,hidc,1),DEConv_GN(hidc),Conv_GN(hidc,hidc,1))
        self.cls_decoder=DepthwiseSeparable(hidc);self.reg_decoder=DepthwiseSeparable(hidc)
        self.cv2=nn.Conv2d(hidc,4*self.reg_max,1);self.cv3=nn.Conv2d(hidc,nc,1)
        self.scale=nn.ModuleList(Scale(1.0) for _ in ch);self.dfl=DFL(self.reg_max)

    def forward(self,x):
        projected=[conv(f) for conv,f in zip(self.conv,x)]
        weights=torch.stack([F.adaptive_avg_pool2d(f,1) for f in projected],dim=1).softmax(dim=1)
        raw=[]
        for i,f in enumerate(projected):
            decoded=self.share_conv(f*weights[:,i])
            reg=self.scale[i](self.cv2(self.reg_decoder(decoded)))
            cls=self.cv3(self.cls_decoder(decoded));raw.append(torch.cat((reg,cls),dim=1))
        if self.training:return raw
        shape=raw[0].shape;x_cat=torch.cat([f.view(shape[0],self.no,-1) for f in raw],dim=2)
        if self.dynamic or self.shape!=shape:
            from .geometry import make_anchors
            self.anchors,self.strides=(a.transpose(0,1) for a in make_anchors(raw,self.stride,0.5));self.shape=shape
        box,cls=x_cat.split((4*self.reg_max,self.nc),dim=1)
        y=torch.cat((self.decode_bboxes(box),cls.sigmoid()),dim=1)
        return y if self.export else (y,raw)
