"""RepVGG-style train-time branches, exactly fused to one deploy-time 3x3.

An established operator, not a novelty claim. All branches participate in training.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RepConv(nn.Module):
    def __init__(self,c1,c2,stride=1,groups=1,branches=1,deploy=False):
        super().__init__()
        if branches<1:raise ValueError("branches must be positive")
        self.c1,self.c2,self.stride,self.groups=c1,c2,stride,groups
        self.branches=branches;self.act=nn.ReLU()
        if deploy:self.reparam=nn.Conv2d(c1,c2,3,stride,1,groups=groups,bias=True)
        else:
            self.dense=nn.ModuleList(self._branch(3,1) for _ in range(branches))
            self.pointwise=self._branch(1,0)
            self.identity=nn.BatchNorm2d(c1) if c1==c2 and stride==1 else None

    def _branch(self,k,p):
        return nn.Sequential(nn.Conv2d(self.c1,self.c2,k,self.stride,p,groups=self.groups,bias=False),nn.BatchNorm2d(self.c2))

    def forward(self,x):
        if hasattr(self,"reparam"):return self.act(self.reparam(x))
        y=self.pointwise(x)
        for branch in self.dense:y=y+branch(x)
        if self.identity is not None:y=y+self.identity(x)
        return self.act(y)

    @staticmethod
    def _fold(branch):
        conv,bn=branch;scale=bn.weight/torch.sqrt(bn.running_var+bn.eps)
        bias=conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        return conv.weight*scale[:,None,None,None],bn.bias+(bias-bn.running_mean)*scale

    def equivalent_kernel_bias(self):
        if hasattr(self,"reparam"):return self.reparam.weight,self.reparam.bias
        kernel,bias=self._fold(self.pointwise);kernel=F.pad(kernel,[1,1,1,1])
        for branch in self.dense:
            w,b=self._fold(branch);kernel=kernel+w;bias=bias+b
        if self.identity is not None:
            bn=self.identity;identity=torch.zeros_like(kernel)
            index=torch.arange(self.c2,device=kernel.device)
            identity[index,index%(self.c1//self.groups),1,1]=1
            scale=bn.weight/torch.sqrt(bn.running_var+bn.eps)
            kernel=kernel+identity*scale[:,None,None,None]
            bias=bias+bn.bias-bn.running_mean*scale
        return kernel,bias

    def switch_to_deploy(self):
        if hasattr(self,"reparam"):return self
        if self.training:raise RuntimeError("Call eval() before RepConv conversion")
        kernel,bias=self.equivalent_kernel_bias()
        conv=nn.Conv2d(self.c1,self.c2,3,self.stride,1,groups=self.groups,bias=True).to(device=kernel.device,dtype=kernel.dtype)
        with torch.no_grad():conv.weight.copy_(kernel);conv.bias.copy_(bias)
        self.reparam=conv
        del self.dense,self.pointwise,self.identity
        return self
