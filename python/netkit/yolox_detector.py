"""Single-scale YOLOX decoupled detection head + MobileNetV4-Small detector builder."""

from __future__ import annotations

from typing import Any

import numpy as np

from .mobilenetv4_small import build_mobilenetv4_small_arch
from .reference_forward import _activate, _conv_nhwc, forward_cnn

YOLOX_MAX_STACKED_CONVS = 4
MNv4_SMALL_BACKBONE_OUT_CHANNELS = 960


def _silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def yolox_head_output_channels(num_classes: int) -> int:
    return 4 + 1 + int(num_classes)


def build_yolox_mnv4_small_detector(
    *,
    height: int,
    width: int,
    channels: int = 3,
    num_classes: int = 10,
    hidden_dim: int = 256,
    num_convs: int = 2,
) -> dict[str, Any]:
    """MobileNetV4-Conv-Small backbone + fused YOLOX decoupled head (single scale)."""
    if num_convs < 1 or num_convs > YOLOX_MAX_STACKED_CONVS:
        raise ValueError(f"num_convs must be in 1..{YOLOX_MAX_STACKED_CONVS}, got {num_convs}")

    arch = build_mobilenetv4_small_arch(
        height=height,
        width=width,
        channels=channels,
        include_head=False,
    )
    arch["layers"].append(
        {
            "type": "yolox_decoupled_head",
            "in_channels": MNv4_SMALL_BACKBONE_OUT_CHANNELS,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "num_convs": num_convs,
        }
    )
    return arch


def backbone_arch_from_detector(arch: dict[str, Any]) -> dict[str, Any]:
    """Return the MobileNetV4-Small backbone portion of a detector arch (no YOLOX head)."""
    if not arch.get("layers") or arch["layers"][-1].get("type") != "yolox_decoupled_head":
        raise ValueError("expected a detector arch ending with yolox_decoupled_head")
    return {"network": "cnn", "input": list(arch["input"]), "layers": arch["layers"][:-1]}


def head_weight_offset(arch: dict[str, Any]) -> int:
    """Flat weight index where the fused YOLOX head tensors begin."""
    from .arch_writer import count_packed_cnn_weight_floats

    return count_packed_cnn_weight_floats(arch, num_layers=len(arch["layers"]) - 1)


def forward_yolox_backbone(
    flat_input: np.ndarray,
    arch: dict[str, Any],
    weights: np.ndarray,
) -> list[float]:
    """Forward through the detector backbone layers only."""
    offset = head_weight_offset(arch)
    return forward_cnn(flat_input, backbone_arch_from_detector(arch), weights[:offset])


def forward_yolox_head_nhwc(
    features: np.ndarray,
    head_layer: dict[str, Any],
    weights: np.ndarray,
    *,
    offset: int | None = None,
) -> np.ndarray:
    """Run the fused YOLOX head on an NHWC backbone feature map."""
    if offset is None:
        raise ValueError("offset is required when weights contain backbone tensors")
    out, _ = yolox_decoupled_head_forward_nhwc(
        np.asarray(features, dtype=np.float32),
        in_channels=int(head_layer["in_channels"]),
        hidden_dim=int(head_layer["hidden_dim"]),
        num_classes=int(head_layer["num_classes"]),
        num_convs=int(head_layer["num_convs"]),
        weights=weights,
        offset=offset,
    )
    return out


def pack_yolox_head_weights_flat(
    rng: np.random.Generator,
    *,
    in_channels: int,
    hidden_dim: int,
    num_classes: int,
    num_convs: int,
    scale: float = 0.05,
) -> list[np.ndarray]:
    """Flat W/B pairs in .nk catalog order for one YOLOX decoupled head."""
    parts: list[np.ndarray] = []

    parts.append(rng.standard_normal(hidden_dim * in_channels, dtype=np.float32) * scale)
    parts.append(rng.standard_normal(hidden_dim, dtype=np.float32) * 0.01)

    branch_elems = hidden_dim * hidden_dim * 9
    for _ in range(num_convs):
        parts.append(rng.standard_normal(branch_elems, dtype=np.float32) * scale)
        parts.append(rng.standard_normal(hidden_dim, dtype=np.float32) * 0.01)
    for _ in range(num_convs):
        parts.append(rng.standard_normal(branch_elems, dtype=np.float32) * scale)
        parts.append(rng.standard_normal(hidden_dim, dtype=np.float32) * 0.01)

    parts.append(rng.standard_normal(num_classes * hidden_dim, dtype=np.float32) * scale)
    parts.append(rng.standard_normal(num_classes, dtype=np.float32) * 0.01)
    parts.append(rng.standard_normal(4 * hidden_dim, dtype=np.float32) * scale)
    parts.append(rng.standard_normal(4, dtype=np.float32) * 0.01)
    parts.append(rng.standard_normal(hidden_dim, dtype=np.float32) * scale)
    parts.append(rng.standard_normal(1, dtype=np.float32) * 0.01)
    return parts


def _run_branch(
    stem: np.ndarray,
    *,
    num_convs: int,
    hidden_dim: int,
    weights: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, int]:
    feat = stem
    for _ in range(num_convs):
        branch_elems = hidden_dim * hidden_dim * 9
        kernel = weights[offset : offset + branch_elems].reshape(hidden_dim, 3, 3, hidden_dim)
        offset += branch_elems
        bias = weights[offset : offset + hidden_dim]
        offset += hidden_dim
        feat = _conv_nhwc(feat, kernel, bias, stride=1, pad_h=1, pad_w=1)
        feat = _silu(feat)
    return feat, offset


def yolox_decoupled_head_forward_nhwc(
    inp: np.ndarray,
    *,
    in_channels: int,
    hidden_dim: int,
    num_classes: int,
    num_convs: int,
    weights: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, int]:
    """Reference forward for fused YOLOX head; returns (H,W,4+1+num_classes)."""
    h, w, _ = inp.shape

    stem_w = weights[offset : offset + hidden_dim * in_channels].reshape(hidden_dim, 1, 1, in_channels)
    offset += hidden_dim * in_channels
    stem_b = weights[offset : offset + hidden_dim]
    offset += hidden_dim
    stem = _conv_nhwc(inp, stem_w, stem_b, stride=1, pad_h=0, pad_w=0)
    stem = _silu(stem)

    cls_feat, offset = _run_branch(stem, num_convs=num_convs, hidden_dim=hidden_dim, weights=weights, offset=offset)
    reg_feat, offset = _run_branch(stem, num_convs=num_convs, hidden_dim=hidden_dim, weights=weights, offset=offset)

    cls_w = weights[offset : offset + num_classes * hidden_dim].reshape(num_classes, 1, 1, hidden_dim)
    offset += num_classes * hidden_dim
    cls_b = weights[offset : offset + num_classes]
    offset += num_classes
    reg_w = weights[offset : offset + 4 * hidden_dim].reshape(4, 1, 1, hidden_dim)
    offset += 4 * hidden_dim
    reg_b = weights[offset : offset + 4]
    offset += 4
    obj_w = weights[offset : offset + hidden_dim].reshape(1, 1, 1, hidden_dim)
    offset += hidden_dim
    obj_b = weights[offset : offset + 1]
    offset += 1

    cls_pred = _conv_nhwc(cls_feat, cls_w, cls_b, stride=1, pad_h=0, pad_w=0)
    reg_pred = _conv_nhwc(reg_feat, reg_w, reg_b, stride=1, pad_h=0, pad_w=0)
    obj_pred = _conv_nhwc(reg_feat, obj_w, obj_b, stride=1, pad_h=0, pad_w=0)

    out_c = yolox_head_output_channels(num_classes)
    out = np.zeros((h, w, out_c), dtype=np.float32)
    out[..., 0:4] = reg_pred
    out[..., 4:5] = obj_pred
    out[..., 5:] = cls_pred
    return out, offset
