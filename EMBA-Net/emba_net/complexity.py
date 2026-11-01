"""Reproducible convolution/GEMM operation count, including functional DEConv.

One MAC is two FLOPs. Normalization, activation, pooling, interpolation, elementwise
ops, decoding and NMS are excluded and are explicitly not claimed as counted.
"""
import copy
import torch
from torch import nn
from .deconv import DEConv
from .glm_pb import GLMContext
from .repconv import RepConv


def to_deploy(model,inplace=False):
    model=model if inplace else copy.deepcopy(model)
    model.eval()
    for module in list(model.modules()):
        if isinstance(module,RepConv):module.switch_to_deploy()
        elif isinstance(module,DEConv) and hasattr(module,"conv1_1"):module.switch_to_deploy()
    return model


def count_ops(model,inputs):
    counts={};handles=[]
    def hook(name,module,args,out):
        total=0
        if isinstance(module,nn.Conv2d):
            total=out.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]*2
        elif isinstance(module,nn.Linear):total=out.numel()*module.in_features*2
        elif isinstance(module,DEConv) and hasattr(module,"conv1_1"):
            total=out.numel()*out.shape[1]*9*2
        elif isinstance(module,GLMContext):
            # Two 1 x (C/groups) by (C/groups) x HW matrix products.
            total=4*args[0].numel()
        counts[name]=counts.get(name,0)+int(total)
    for name,module in model.named_modules():
        if isinstance(module,(nn.Conv2d,nn.Linear,DEConv,GLMContext)):
            handles.append(module.register_forward_hook(lambda m,a,o,n=name:hook(n,m,a,o)))
    try:
        with torch.inference_mode():model(*inputs)
    finally:
        for h in handles:h.remove()
    return sum(counts.values()),counts


def profile_model(model,size=640):
    train_params=sum(p.numel() for p in model.parameters())
    deploy=to_deploy(model)
    parameter=next(deploy.parameters())
    x=torch.zeros(1,3,size,size,device=parameter.device,dtype=parameter.dtype)
    flops,_=count_ops(deploy,(x,))
    return {'input':[1,3,size,size],'training_parameters':train_params,
            'deployment_parameters':sum(p.numel() for p in deploy.parameters()),
            'deployment_flops':flops,'deployment_gflops':flops/1e9,
            'convention':'2 FLOPs/MAC; convolution + GEMM; includes functional DEConv; excludes norm/activation/pooling/interpolation/elementwise/decode/NMS'}
