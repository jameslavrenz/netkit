"""ResNet-18 architecture (BasicBlock stages, torchvision-compatible topology)."""

from __future__ import annotations

from typing import Any

from .arch_writer import _out_dim, _resnet_output_spatial

# (in_channels, out_channels, stride) for each BasicBlock after the stem.
RESNET18_BLOCKS: list[tuple[int, int, int]] = [
    (64, 64, 1),
    (64, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 256, 2),
    (256, 256, 1),
    (256, 512, 2),
    (512, 512, 1),
]


def build_resnet18_arch(
    *,
    height: int,
    width: int,
    channels: int = 3,
    num_classes: int = 10,
    include_head: bool = True,
) -> dict[str, Any]:
    """Build full ResNet-18 as a netkit CNN arch dict."""
    h, w, ch = height, width, channels
    layers: list[dict[str, Any]] = [
        {
            "type": "conv2d",
            "kernel_size": 7,
            "stride": 2,
            "filters": 64,
            "pad_h": 3,
            "pad_w": 3,
            "activation": "relu",
        },
        {"type": "max_pool2d", "pool_size": 3, "stride": 2, "pad_h": 1, "pad_w": 1},
    ]
    h = _out_dim(h, 7, 2, 3)
    w = _out_dim(w, 7, 2, 3)
    h = _out_dim(h, 3, 2, 1)
    w = _out_dim(w, 3, 2, 1)
    ch = 64

    for in_c, out_c, stride in RESNET18_BLOCKS:
        layer = {
            "type": "resnet_basic_block",
            "in_channels": in_c,
            "out_channels": out_c,
            "stride": stride,
        }
        layers.append(layer)
        h, w = _resnet_output_spatial(h, w, layer)
        ch = out_c

    if include_head:
        if h < 1 or w < 1:
            raise ValueError(f"input {height}x{width} is too small for ResNet-18")
        layers.append({"type": "avg_pool2d", "pool_size": h, "stride": h})
        layers.append({"type": "flatten"})
        layers.append({"type": "dense", "units": num_classes, "activation": "none"})

    return {"network": "cnn", "input": [height, width, channels], "layers": layers}
