"""MobileNetV4-Conv-Small architecture (matches d-li14/mobilenetv4.pytorch block_specs)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .arch_writer import _make_divisible, _out_dim, _uib_output_spatial

# (conv_bn, kernel, stride, out_ch) or (uib, start_dw, middle_dw, stride, out_ch, expand_ratio)
MOBILENETV4_CONV_SMALL_BLOCKS: list[tuple] = [
    ("conv_bn", 3, 2, 32),
    ("conv_bn", 3, 2, 32),
    ("conv_bn", 1, 1, 32),
    ("conv_bn", 3, 2, 96),
    ("conv_bn", 1, 1, 64),
    ("uib", 5, 5, 2, 96, 3.0),
    ("uib", 0, 3, 1, 96, 2.0),
    ("uib", 0, 3, 1, 96, 2.0),
    ("uib", 0, 3, 1, 96, 2.0),
    ("uib", 0, 3, 1, 96, 2.0),
    ("uib", 3, 0, 1, 96, 4.0),
    ("uib", 3, 3, 2, 128, 6.0),
    ("uib", 5, 5, 1, 128, 4.0),
    ("uib", 0, 5, 1, 128, 4.0),
    ("uib", 0, 5, 1, 128, 3.0),
    ("uib", 0, 3, 1, 128, 4.0),
    ("uib", 0, 3, 1, 128, 4.0),
    ("conv_bn", 1, 1, 960),
]


def _conv_bn_layer(kernel: int, stride: int, filters: int) -> dict[str, Any]:
    pad = (kernel - 1) // 2
    return {
        "type": "conv2d",
        "kernel_size": kernel,
        "stride": stride,
        "filters": filters,
        "pad_h": pad,
        "pad_w": pad,
        "activation": "relu",
    }


def _uib_layer(
    *,
    in_channels: int,
    out_channels: int,
    start_dw_kernel: int,
    middle_dw_kernel: int,
    stride: int,
    expand_ratio: float,
) -> dict[str, Any]:
    return {
        "type": "mobilenetv4_uib",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "start_dw_kernel": start_dw_kernel,
        "middle_dw_kernel": middle_dw_kernel,
        "stride": stride,
        "expand_ratio": expand_ratio,
        "middle_dw_downsample": 1,
    }


def build_mobilenetv4_small_arch(
    *,
    height: int,
    width: int,
    channels: int = 3,
    num_classes: int = 10,
    include_head: bool = True,
) -> dict[str, Any]:
    """Build full MNv4-Conv-Small as a netkit CNN arch dict."""
    h, w, ch = height, width, channels
    layers: list[dict[str, Any]] = []

    for block in MOBILENETV4_CONV_SMALL_BLOCKS:
        if block[0] == "conv_bn":
            _, kernel, stride, out_ch = block
            layers.append(_conv_bn_layer(kernel, stride, out_ch))
            pad = (kernel - 1) // 2
            h = _out_dim(h, kernel, stride, pad)
            w = _out_dim(w, kernel, stride, pad)
            ch = out_ch
        elif block[0] == "uib":
            _, start_dw, middle_dw, stride, out_ch, expand_ratio = block
            layer = _uib_layer(
                in_channels=ch,
                out_channels=out_ch,
                start_dw_kernel=start_dw,
                middle_dw_kernel=middle_dw,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            layers.append(layer)
            h, w = _uib_output_spatial(h, w, layer)
            ch = out_ch

    if include_head:
        if h < 1 or w < 1:
            raise ValueError(f"input {height}x{width} is too small for MobileNetV4-Conv-Small")
        layers.append({"type": "avg_pool2d", "pool_size": h, "stride": h})
        layers.append(
            {
                "type": "conv2d",
                "kernel_size": 1,
                "stride": 1,
                "filters": 1280,
                "pad_h": 0,
                "pad_w": 0,
                "activation": "relu",
            }
        )
        layers.append({"type": "flatten"})
        layers.append({"type": "dense", "units": num_classes, "activation": "none"})

    return {"network": "cnn", "input": [height, width, channels], "layers": layers}


def pack_uib_weights_flat(
    rng: np.random.Generator,
    *,
    in_channels: int,
    out_channels: int,
    start_dw_kernel: int,
    middle_dw_kernel: int,
    expand_ratio: float,
    scale: float = 0.05,
) -> list[np.ndarray]:
    expand_c = _make_divisible(in_channels * expand_ratio, 8)
    parts: list[np.ndarray] = []

    if start_dw_kernel:
        parts.append(rng.standard_normal((in_channels, start_dw_kernel, start_dw_kernel), dtype=np.float32).reshape(-1) * scale)
        parts.append(rng.standard_normal(in_channels, dtype=np.float32) * 0.01)
        parts.append(np.ones(in_channels, dtype=np.float32))
        parts.append(np.zeros(in_channels, dtype=np.float32))

    parts.append(rng.standard_normal((expand_c, in_channels), dtype=np.float32).reshape(-1) * scale)
    parts.append(rng.standard_normal(expand_c, dtype=np.float32) * 0.01)
    parts.append(np.ones(expand_c, dtype=np.float32))
    parts.append(np.zeros(expand_c, dtype=np.float32))

    if middle_dw_kernel:
        parts.append(
            rng.standard_normal((expand_c, middle_dw_kernel, middle_dw_kernel), dtype=np.float32).reshape(-1) * scale
        )
        parts.append(rng.standard_normal(expand_c, dtype=np.float32) * 0.01)
        parts.append(np.ones(expand_c, dtype=np.float32))
        parts.append(np.zeros(expand_c, dtype=np.float32))

    parts.append(rng.standard_normal((out_channels, expand_c), dtype=np.float32).reshape(-1) * scale)
    parts.append(rng.standard_normal(out_channels, dtype=np.float32) * 0.01)
    parts.append(np.ones(out_channels, dtype=np.float32))
    parts.append(np.zeros(out_channels, dtype=np.float32))
    return parts
