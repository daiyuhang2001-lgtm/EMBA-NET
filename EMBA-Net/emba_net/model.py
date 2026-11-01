"""Focused builder for the supplied EMBA-Net.yaml graph.

Derived from the model parsing logic in the supplied Ultralytics fork, AGPL-3.0.
Paper-facing module names are used; legacy tensor checkpoints are migrated explicitly.
"""
from copy import deepcopy
from pathlib import Path
import math
import re
import torch
from torch import nn
import yaml

from .common import Conv, C2f, SPPF
from .glm_paper import GLM_P_B
from .figure_bgsa import BGSA
from .original_bgsa import BGSA as OriginalBGSA
from .lsd_paper import LSDheadreplacement
from .glm_pb import GLM_P_B as OriginalGLM
from .lsd_head import LSDheadreplacement as OriginalHead
from .backbone import PVFeaturePerceptionNet, DetectorGraph

MODULES = {c.__name__: c for c in (Conv, C2f, SPPF, GLM_P_B, BGSA, LSDheadreplacement)}
LEGACY_MODULE_NAMES = {"SBA": "BGSA", "C2f_EMA": "GLM_P_B", "Detect_LSDECD": "LSDheadreplacement"}
PAPER_NAMES = {"GLM_P_B": "GLM-P B", "BGSA": "BGSA", "LSDheadreplacement": "LSDheadreplacement"}


def build_graph(config, channels=3):
    """Keep the original parser's scaling and repeat semantics for these six modules."""
    cfg = deepcopy(config)
    original = cfg.get("implementation") == "original"
    glm_cls = OriginalGLM if original else GLM_P_B
    head_cls = OriginalHead if original else LSDheadreplacement
    bgsa_cls = OriginalBGSA if original else BGSA
    modules = {**MODULES, "GLM_P_B": glm_cls, "LSDheadreplacement": head_cls,"BGSA":bgsa_cls}
    # A named backbone expands to the same ten stages. Head indices retain the
    # original global numbering (10..20), rather than shifting to 1..11.
    spec = cfg["backbone"]
    if len(spec) == 1 and spec[0][2] == "PVFeaturePerceptionNet":
        if spec[0] != [-1, 1, "PVFeaturePerceptionNet", []]:
            raise ValueError("PVFeaturePerceptionNet expects [-1, 1, PVFeaturePerceptionNet, []].")
        cfg["backbone"] = PVFeaturePerceptionNet.stage_spec()
    backbone_stages = len(cfg["backbone"])
    scale = cfg.get("scale") or next(iter(cfg["scales"]))
    depth, width, maximum = cfg["scales"][scale]
    nc = cfg["nc"]
    ch, layers, saves = [channels], [], []
    for i, (f, repeat, name, raw_args) in enumerate(cfg["backbone"] + cfg["head"]):
        name = LEGACY_MODULE_NAMES.get(name, name)
        if name not in MODULES:
            raise ValueError(f"This focused builder does not support module {name!r}.")
        cls = modules[name]
        args = [nc if a == "nc" else a for a in raw_args]
        n = max(round(repeat * depth), 1) if repeat > 1 else repeat
        if cls in (Conv, C2f, SPPF, glm_cls):
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = math.ceil(min(c2, maximum) * width / 8) * 8
            args = [c1, c2, *args[1:]]
            if cls is GLM_P_B:
                hidden = cfg.get("glm_hidden", {}).get(i, int(c2*0.5))
                branches=cfg.get("rep_branches",1)
                branches=branches.get(i,1) if isinstance(branches,dict) else branches
                args = [c1,c2,n,False,1,hidden/c2,branches]
                n = 1
            # The supplied parser inserts repeats into C2f, but NOT GLM_P_B.
            # Preserve that distinction, including for non-n scales.
            if cls is C2f:
                args.insert(2, n)
                n = 1
        elif cls is bgsa_cls:
            incoming = [ch[j] for j in f]
            c2 = incoming[-1]
            args = [incoming, c2]
        else:
            args.append([ch[j] for j in f])
            args[1] = math.ceil(min(args[1], maximum) * width / 8) * 8
            c2 = ch[f[-1]]  # final-layer bookkeeping only
        layer = nn.Sequential(*(cls(*args) for _ in range(n))) if n > 1 else cls(*args)
        layer.i, layer.f, layer.type = i, f, name
        layer.np = sum(p.numel() for p in layer.parameters())
        saves.extend(j % i for j in ([f] if isinstance(f, int) else f) if j != -1)
        layers.append(layer)
        if i == 0:
            ch = []
        ch.append(c2)
    if not isinstance(layers[-1], (LSDheadreplacement,OriginalHead)):
        raise ValueError("The selected graph must end in LSDheadreplacement.")
    return DetectorGraph(layers, backbone_stages), sorted(set(saves))


class EMBANet(nn.Module):
    """Application-facing name for the supplied model, with documented YOLO provenance."""

    def __init__(self, config, nc=None, scale=None):
        super().__init__()
        if isinstance(config, (str, Path)):
            with Path(config).open(encoding="utf-8-sig") as f:
                config = yaml.safe_load(f)
        self.yaml = deepcopy(config)
        for layer in self.yaml["backbone"] + self.yaml["head"]:
            layer[2] = LEGACY_MODULE_NAMES.get(layer[2], layer[2])
        if nc is not None:
            if not isinstance(nc, int) or nc < 1:
                raise ValueError("nc must be a positive integer.")
            self.yaml["nc"] = nc
        if scale is not None:
            self.yaml["scale"] = scale
        self.nc = self.yaml["nc"]
        self.model, self.save = build_graph(self.yaml, self.yaml.get("ch", 3))
        self.names = {i: str(i) for i in range(self.nc)}
        if self.nc == 1:
            self.names = {0: "BirdDrop"}
        # Match the supplied constructor: stride probe in train mode, then bias
        # initialization, then BN eps/momentum and activation initialization.
        with torch.no_grad():
            raw = self._forward_once(torch.zeros(2, self.yaml.get("ch", 3), 640, 640))
        head = self.model[-1]
        head.stride = torch.tensor([640 / f.shape[-2] for f in raw])
        self.stride = head.stride
        head.bias_init()
        for m in self.modules():
            if type(m) is nn.BatchNorm2d:
                m.eps, m.momentum = 1e-3, 0.03
            elif type(m) in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU):
                m.inplace = True

    def _forward_once(self, x):
        features = self.model.backbone(x)
        outputs = [None] * len(self.model.backbone.layers)
        for index, feature in features.items():
            if index in self.save:
                outputs[index] = feature
        x = features[9]
        for layer in self.model.head:
            if layer.f != -1:
                x = outputs[layer.f] if isinstance(layer.f, int) else [x if j == -1 else outputs[j] for j in layer.f]
            x = layer(x)
            outputs.append(x if layer.i in self.save else None)
        return x

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.yaml.get("ch", 3):
            raise ValueError("Expected BCHW input with the configured channel count.")
        if x.shape[-1] % 32 or x.shape[-2] % 32:
            raise ValueError("Input height and width must be multiples of 32.")
        return self._forward_once(x)

    def _apply(self, fn):
        super()._apply(fn)
        head = self.model[-1]
        for name in ("stride", "anchors", "strides"):
            setattr(head, name, fn(getattr(head, name)))
        self.stride = head.stride
        return self

    def load_weights(self, path):
        """Load tensor-only checkpoints strictly; never silently drop mismatched layers."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError("Expected a state_dict or a tensor-only checkpoint dictionary.")
        if checkpoint.get("deployment",False):
            self.switch_to_deploy()
        state = checkpoint.get("state_dict", checkpoint)
        # Migrate flat legacy model indices to the named backbone/head containers.
        mapped = {}
        for key, value in state.items():
            target = key.replace(".EMA.", ".context.")
            legacy = re.match(r"^model\.(\d+)\.(.+)$", target)
            if legacy:
                index, suffix = int(legacy.group(1)), legacy.group(2)
                target = (f"model.backbone.layers.{index}.{suffix}" if index < 10
                          else f"model.head.{index-10}.{suffix}")
            if target in mapped:
                raise ValueError(f"Conflicting legacy/new checkpoint keys: {target}")
            mapped[target] = value
        state = mapped
        self.load_state_dict(state, strict=True)
        if "names" in checkpoint:
            self.names = checkpoint["names"]
        return self

    def save_weights(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"state_dict": self.state_dict(), "config": self.yaml, "names": self.names,
                      "deployment":getattr(self,"deployment",False),**metadata}
        torch.save(checkpoint, path)

    def switch_to_deploy(self):
        from .complexity import to_deploy
        self.eval()
        to_deploy(self,inplace=True)
        self.deployment=True
        return self

    def summary(self, detail=False):
        rows = [{"index": m.i, "from": m.f, "module": m.type,
                 "paper_name": PAPER_NAMES.get(m.type, m.type), "parameters": m.np} for m in self.model]
        backbone = {"index": 0, "through_index": 9, "from": -1, "module": "PVFeaturePerceptionNet",
                    "paper_name": "Feature Perception", "outputs": list(PVFeaturePerceptionNet.output_names),
                    "parameters": sum(p.numel() for p in self.model.backbone.parameters())}
        return {"classes": self.nc, "scale": self.yaml.get("scale", "n"),
                "backbone": "PVFeaturePerceptionNet",
                "backbone_display_name": "Feature Perception",
                "parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "strides": self.stride.tolist(),
                "layers": rows if detail else [backbone, *rows[10:]]}
