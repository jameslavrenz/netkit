"""Shared depthwise conv helpers for arch dicts and .nk packing."""

from __future__ import annotations

from typing import Any

from .pad_encoding import decode_pad_extra


def depthwise_kernel_hw(layer: dict[str, Any]) -> tuple[int, int]:
    """Return (kernel_h, kernel_w) for depthwise_conv2d arch layers."""
    try:
        return int(layer["kernel_h"]), int(layer["kernel_w"])
    except KeyError as exc:
        raise ValueError("depthwise_conv2d requires kernel_h and kernel_w") from exc


def reconcile_depthwise_kernel(
    *,
    kernel_h: int,
    kernel_w_byte: int,
    pad_h: int,
    pad_w: int,
    channels: int,
    weight_elems: int,
) -> tuple[int, int, int, int]:
    """Resolve depthwise kernel width and asymmetric pads from header byte + weights."""
    if weight_elems % channels != 0:
        raise ValueError(
            f"depthwise weight count {weight_elems} not divisible by channels {channels}"
        )
    kernel_area = weight_elems // channels

    def _pad_from_byte(kw_byte: int) -> tuple[int, int, int, int]:
        top, left, bottom, right = decode_pad_extra(pad_h, pad_w, kw_byte)
        return kernel_h, top, left, bottom, right

    candidates: list[tuple[int, int, int, int, int]] = []
    if kernel_w_byte == kernel_h and kernel_h * kernel_h == kernel_area:
        candidates.append((kernel_h, pad_h, pad_w, pad_h, pad_w))
    if kernel_h * kernel_h == kernel_area and kernel_w_byte != kernel_h:
        candidates.append(_pad_from_byte(kernel_w_byte))
    literal_kw = kernel_w_byte if kernel_w_byte else kernel_h
    if kernel_h * literal_kw == kernel_area:
        candidates.append((literal_kw, pad_h, pad_w, pad_h, pad_w))
    if kernel_h != literal_kw and kernel_area % kernel_h == 0:
        actual_kw = kernel_area // kernel_h
        if actual_kw != kernel_h and kernel_h * kernel_w_byte != kernel_area:
            top, left, bottom, right = decode_pad_extra(pad_h, pad_w, kernel_w_byte)
            if bottom != top or right != left:
                candidates.append((actual_kw, top, left, bottom, right))

    if not candidates:
        raise ValueError(
            f"depthwise kernel metadata mismatch: kh={kernel_h} kw_byte={kernel_w_byte} "
            f"weight_elems={weight_elems} channels={channels}"
        )

    kw, top, left, bottom, right = candidates[0]
    for candidate in candidates[1:]:
        c_kw, c_top, c_left, c_bottom, c_right = candidate
        c_pad_score = int(c_bottom != c_top) + int(c_right != c_left)
        kw_score = int(c_kw == kernel_h)
        best_pad_score = int(bottom != top) + int(right != left)
        best_kw_score = int(kw == kernel_h)
        if (c_pad_score, kw_score) > (best_pad_score, best_kw_score):
            kw, top, left, bottom, right = candidate
    return kw, top, left, bottom, right


def conv2d_input_channels(
    layer: dict[str, Any],
    tensor_channels: int,
    *,
    weight_elems: int | None = None,
) -> int:
    """Resolve conv2d input channels from layer metadata or weight tensor size."""
    stored = int(layer.get("in_channels", 0))
    if stored > 0:
        return stored
    if weight_elems is not None:
        k = int(layer["kernel_size"])
        out_c = int(layer["filters"])
        denom = k * k * out_c
        if denom <= 0 or weight_elems % denom != 0:
            raise ValueError(
                f"conv2d weight count {weight_elems} does not match "
                f"kernel={k} filters={out_c}"
            )
        return weight_elems // denom
    return int(tensor_channels)


def _uib_subop_weight_tensor_count(layer: dict[str, Any]) -> int:
    """Weight tensors for one UIB when BN is folded (int8 / folded float packs)."""
    count = 2  # expand + project
    if int(layer.get("start_dw_kernel", 0)):
        count += 1
    if int(layer.get("middle_dw_kernel", 0)):
        count += 1
    return count


def _layer_weight_tensor_count(layer: dict[str, Any], *, bn_folded: bool = False) -> int:
    layer_type = layer.get("type")
    if layer_type in {"conv2d", "depthwise_conv2d", "dense", "batch_norm2d", "layernorm2d"}:
        return 1
    if layer_type == "convnextv2_block":
        return 5
    if layer_type == "mobilenetv4_uib":
        if bn_folded:
            return _uib_subop_weight_tensor_count(layer)
        count = 0
        if int(layer.get("start_dw_kernel", 0)):
            count += 2
        count += 2  # expand conv + bn
        if int(layer.get("middle_dw_kernel", 0)):
            count += 2
        count += 2  # project conv + bn
        return count
    if layer_type == "resnet_basic_block":
        count = 4
        if int(layer.get("stride", 1)) != 1 or int(layer["in_channels"]) != int(layer["out_channels"]):
            count += 2
        return count
    if layer_type == "yolox_decoupled_head":
        return 3 + 2 * int(layer.get("num_convs", 2))
    if layer_type == "feature_tap":
        return 0
    if layer_type == "yolox_pafpn_multiscale":
        per_head = 3 + 2 * int(layer.get("num_convs", 2))
        return 11 + 3 * per_head
    return 0


def depthwise_arch_entry(layer: dict[str, Any]) -> dict[str, Any]:
    """Build arch dict entry from a parsed depthwise layer descriptor."""
    entry: dict[str, Any] = {
        "type": "depthwise_conv2d",
        "kernel_h": layer["kernel_h"],
        "kernel_w": layer["kernel_w"],
        "stride": layer["stride"],
        "filters": layer["filters"],
        "activation": layer.get("activation", "none"),
    }
    if layer.get("pad_h", 0):
        entry["pad_h"] = layer["pad_h"]
    if layer.get("pad_w", 0):
        entry["pad_w"] = layer["pad_w"]
    if layer.get("pad_h_end", layer.get("pad_h", 0)) != layer.get("pad_h", 0):
        entry["pad_h_end"] = layer["pad_h_end"]
    if layer.get("pad_w_end", layer.get("pad_w", 0)) != layer.get("pad_w", 0):
        entry["pad_w_end"] = layer["pad_w_end"]
    if layer.get("activation") == "leaky_relu":
        entry["alpha"] = float(layer.get("alpha", 0.01))
    return entry
