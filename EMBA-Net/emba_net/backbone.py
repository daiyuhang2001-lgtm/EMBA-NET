"""PVFeaturePerceptionNet: an application name for the existing backbone, AGPL-3.0.

The name groups the supplied Ultralytics-derived Conv/C2f/SPPF stages; it does
not introduce a new primitive operator or change the backbone computation.
"""
from copy import deepcopy
from torch import nn


class PVFeaturePerceptionNet(nn.Module):
    """Existing ten-stage backbone with P2/P3/P4/P5 feature outputs."""

    _STAGES = [
        [-1, 1, "Conv", [64, 3, 2]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 3, "C2f", [128, True]],
        [-1, 1, "Conv", [256, 3, 2]],
        [-1, 6, "C2f", [256, True]],
        [-1, 1, "Conv", [512, 3, 2]],
        [-1, 6, "C2f", [512, True]],
        [-1, 1, "Conv", [1024, 3, 2]],
        [-1, 3, "C2f", [1024, True]],
        [-1, 1, "SPPF", [1024, 5]],
    ]
    output_indices = (2, 4, 6, 9)
    output_names = ("P2", "P3", "P4", "P5")

    @classmethod
    def stage_spec(cls):
        """Expand the original stage definitions before applying scale factors."""
        return deepcopy(cls._STAGES)

    def __init__(self, layers):
        super().__init__()
        if len(layers) != 10 or any(layer.f != -1 for layer in layers):
            raise ValueError("PVFeaturePerceptionNet requires the supplied ten sequential backbone stages.")
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        features = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.output_indices:
                features[i] = x
        return features


class DetectorGraph(nn.Module):
    """Registered backbone/head containers with legacy global-index access."""

    def __init__(self, layers, backbone_stages):
        super().__init__()
        if backbone_stages != 10:
            raise ValueError("This focused graph supports the supplied ten-stage backbone only.")
        self.backbone = PVFeaturePerceptionNet(layers[:backbone_stages])
        self.head = nn.Sequential(*layers[backbone_stages:])

    def __iter__(self):
        yield from self.backbone.layers
        yield from self.head

    def __getitem__(self, index):
        return tuple(iter(self))[index]

    def __len__(self):
        return len(self.backbone.layers) + len(self.head)
