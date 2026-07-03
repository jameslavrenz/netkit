"""ConvNeXt V2-Atto architecture (facebookresearch/ConvNeXt-V2 convnextv2_atto)."""

from __future__ import annotations

from typing import Any

from .arch_writer import _out_dim

CONVNEXTV2_ATTO_DEPTHS: list[int] = [2, 2, 6, 2]
CONVNEXTV2_ATTO_DIMS: list[int] = [40, 80, 160, 320]
CONVNEXTV2_ATTO_EPS = 1e-6


def _stem_conv(filters: int) -> dict[str, Any]:
    return {
        "type": "conv2d",
        "kernel_size": 4,
        "stride": 4,
        "filters": filters,
        "pad_h": 0,
        "pad_w": 0,
        "activation": "none",
    }


def _downsample_conv(filters: int) -> dict[str, Any]:
    return {
        "type": "conv2d",
        "kernel_size": 2,
        "stride": 2,
        "filters": filters,
        "pad_h": 0,
        "pad_w": 0,
        "activation": "none",
    }


def _layernorm(channels: int) -> dict[str, Any]:
    return {"type": "layernorm2d", "channels": channels, "eps": CONVNEXTV2_ATTO_EPS}


def _block(channels: int) -> dict[str, Any]:
    return {"type": "convnextv2_block", "channels": channels, "eps": CONVNEXTV2_ATTO_EPS}


def build_convnextv2_atto_arch(
    *,
    height: int,
    width: int,
    channels: int = 3,
    num_classes: int = 10,
    include_head: bool = True,
) -> dict[str, Any]:
    """Build full ConvNeXt V2-Atto as a netkit CNN arch dict."""
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("ConvNeXt V2-Atto requires height and width divisible by 32")

    h, w, ch = height, width, channels
    layers: list[dict[str, Any]] = []

    layers.append(_stem_conv(CONVNEXTV2_ATTO_DIMS[0]))
    h = _out_dim(h, 4, 4, 0)
    w = _out_dim(w, 4, 4, 0)
    layers.append(_layernorm(CONVNEXTV2_ATTO_DIMS[0]))
    ch = CONVNEXTV2_ATTO_DIMS[0]

    for stage_i, depth in enumerate(CONVNEXTV2_ATTO_DEPTHS):
        if stage_i > 0:
            layers.append(_layernorm(CONVNEXTV2_ATTO_DIMS[stage_i - 1]))
            layers.append(_downsample_conv(CONVNEXTV2_ATTO_DIMS[stage_i]))
            h = _out_dim(h, 2, 2, 0)
            w = _out_dim(w, 2, 2, 0)
            ch = CONVNEXTV2_ATTO_DIMS[stage_i]

        for _ in range(depth):
            layers.append(_block(ch))

    if include_head:
        if h < 1 or w < 1:
            raise ValueError(f"input {height}x{width} is too small for ConvNeXt V2-Atto")
        layers.append(_layernorm(ch))
        layers.append({"type": "avg_pool2d", "pool_size": h, "stride": h})
        layers.append({"type": "flatten"})
        layers.append({"type": "dense", "units": num_classes, "activation": "none"})

    return {"network": "cnn", "input": [height, width, channels], "layers": layers}
