"""
models.py - Timmy-backed EfficientNet B0-B5 + MobileNet V3 Large-0.75 + ConvNeXtV2-Base Factory
============================================================================================
Unified factory exposing EfficientNet-B0…B5, MobileNet V3 Large (0.75), and ConvNeXtV2-Base via a
single 'getModel(config)' interface. Relies on **timm** for all ImageNet-pretrained models,
avoiding external back-ends. Supports optional meta-feature branches via 'modify_meta()'.

Minimal usage
-------------
'''python
from models import getModel
cfg = {
    "model_type": "convnextv2_base",  # or "efficientnet_b3" / "mobilenet_v3_large_075"
    "numClasses": 8,
}
builder = getModel(cfg)
model = builder()
'''
"""

from __future__ import annotations
import functools
from typing import Callable, Dict
import torch.nn as nn
import timm
from torchvision import models as tv
from torchvision.models import MobileNet_V3_Large_Weights


# 1.  Back-bone builders via timm: EfficientNet B0-B5, ConvNeXtV2-Base
#                + torchvision MobileNet V3 Large-0.75

def _timm_model(name: str, cfg: Dict) -> nn.Module:
    """Helper: create a timm model with custom classifier head."""
    num_classes = int(cfg.get("numClasses") or cfg.get("num_classes", 1000))
    model = timm.create_model(name, pretrained=True, num_classes=num_classes)
    return model

# EfficientNet variants (timm)

def efficientnet_b0(cfg: Dict) -> nn.Module:
    return _timm_model("efficientnet_b0", cfg)

def efficientnet_b1(cfg: Dict) -> nn.Module:
    return _timm_model("efficientnet_b1", cfg)

def efficientnet_b2(cfg: Dict) -> nn.Module:
    return _timm_model("efficientnet_b2", cfg)

def efficientnet_b3(cfg: Dict) -> nn.Module:
    return _timm_model("efficientnet_b3", cfg)

def efficientnet_b4(cfg: Dict) -> nn.Module:
    return _timm_model("efficientnet_b4", cfg)

def efficientnet_b5(cfg: Dict) -> nn.Module:
    return _timm_model("efficientnet_b5", cfg)

# ConvNeXtV2-Base (timm)

def convnextv2_base(cfg: Dict) -> nn.Module:
    return _timm_model("convnextv2_base", cfg)

# MobileNet V3 Large-0.75 (torchvision)

def mobilenet_v3_large_075(cfg: Dict) -> nn.Module:
    model = tv.mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        width_mult=0.75
    )
    num_classes = int(cfg.get("numClasses") or cfg.get("num_classes", 1000))
    if isinstance(model.classifier, nn.Sequential):
        # last layer is Linear
        in_f = model.classifier[-1].in_features  # type: ignore[index]
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    else:
        raise RuntimeError("Unexpected classifier structure")
    return model


# 2.  Meta-feature injector (optional)

def modify_meta(mdlParams: Dict, model: nn.Module) -> nn.Module:
    n_meta = mdlParams["meta_array"].shape[1]
    layers: list[nn.Module] = []
    in_dim = n_meta
    for dim in mdlParams["fc_layers_before"]:
        layers += [
            nn.Linear(in_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(p=mdlParams.get("dropout_meta", 0.5))
        ]
        in_dim = dim
    model.meta_before = nn.Sequential(*layers)
    return model


# 3.  Registry

model_map: dict[str, Callable[[Dict], nn.Module]] = {
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b3": efficientnet_b3,
    "efficientnet_b4": efficientnet_b4,
    "efficientnet_b5": efficientnet_b5,
    "convnextv2_base": convnextv2_base,
    "mobilenet_v3_large_075": mobilenet_v3_large_075,
}


# 4.  Public factory helper

def getModel(config: Dict) -> Callable[[], nn.Module]:
    model_type = (config.get("model_type") or config.get("model_name", "")).lower()
    if model_type not in model_map:
        raise ValueError(f"Unknown model '{model_type}'. Available: {list(model_map)}")
    builder_fn = model_map[model_type]
    @functools.wraps(builder_fn)
    def builder() -> nn.Module:
        model = builder_fn(config)
        if "meta_array" in config:
            model = modify_meta(config, model)
        return model
    return builder


# 5.  Quick sanity test

if __name__ == "__main__":
    cfg = {"model_type": "convnextv2_base", "numClasses": 2}
    net = getModel(cfg)()
    print(
        "✓", cfg["model_type"],
        "→ params:", sum(p.numel() for p in net.parameters()) // 1_000_000, "M"
    )
