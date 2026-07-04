"""Build .nk files from in-memory architecture dicts and flat weight arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .cnn_layers import depthwise_arch_entry, depthwise_kernel_hw
from .format import MAX_LAYERS, activation_from_name
from .writer import LayerSpec, ModelSpec, RegressionCase, RegressionSuite, write_nk, write_nk_bytes


def _make_divisible(value: float, divisor: int = 8) -> int:
    rounded = int(value + divisor / 2) // divisor * divisor
    result = max(divisor, rounded)
    if result < int(0.9 * value):
        result += divisor
    return result


def _uib_output_spatial(height: int, width: int, layer: dict) -> tuple[int, int]:
    start_k = int(layer.get("start_dw_kernel", 0))
    middle_k = int(layer.get("middle_dw_kernel", 0))
    stride = int(layer.get("stride", 1))
    middle_down = bool(layer.get("middle_dw_downsample", 1))
    h, w = height, width
    if start_k:
        pad = (start_k - 1) // 2
        dw_stride = 1 if middle_down and middle_k > 0 else stride
        h = _out_dim(h, start_k, dw_stride, pad)
        w = _out_dim(w, start_k, dw_stride, pad)
    if middle_k:
        pad = (middle_k - 1) // 2
        mid_stride = stride if middle_down else 1
        h = _out_dim(h, middle_k, mid_stride, pad)
        w = _out_dim(w, middle_k, mid_stride, pad)
    return h, w


def _resnet_output_spatial(height: int, width: int, layer: dict) -> tuple[int, int]:
    stride = int(layer.get("stride", 1))
    pad = 1
    h = _out_dim(height, 3, stride, pad)
    w = _out_dim(width, 3, stride, pad)
    return h, w


def _append_resnet_basic_block_weights(
    parts: list[np.ndarray],
    rng: np.random.Generator,
    *,
    in_c: int,
    out_c: int,
    stride: int,
    scale: float,
) -> None:
    conv1 = rng.standard_normal((out_c, 3, 3, in_c), dtype=np.float32) * scale
    parts.append(conv1.reshape(-1))
    parts.append(rng.standard_normal(out_c, dtype=np.float32) * 0.01)
    parts.append(np.ones(out_c, dtype=np.float32))
    parts.append(np.zeros(out_c, dtype=np.float32))

    conv2 = rng.standard_normal((out_c, 3, 3, out_c), dtype=np.float32) * scale
    parts.append(conv2.reshape(-1))
    parts.append(rng.standard_normal(out_c, dtype=np.float32) * 0.01)
    parts.append(np.ones(out_c, dtype=np.float32))
    parts.append(np.zeros(out_c, dtype=np.float32))

    if stride != 1 or in_c != out_c:
        shortcut = rng.standard_normal((out_c, 1, 1, in_c), dtype=np.float32) * scale
        parts.append(shortcut.reshape(-1))
        parts.append(rng.standard_normal(out_c, dtype=np.float32) * 0.01)
        parts.append(np.ones(out_c, dtype=np.float32))
        parts.append(np.zeros(out_c, dtype=np.float32))


def _out_dim(in_dim: int, kernel: int, stride: int, pad: int = 0) -> int:
    return (in_dim + 2 * pad - kernel) // stride + 1


def _split_mlp_weights(arch: dict, weights: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    offset = 0
    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []
    in_features = arch["input"][1]

    for layer in arch["layers"]:
        out_features = layer["units"]
        w_size = in_features * out_features
        w = weights[offset : offset + w_size].reshape(out_features, in_features)
        offset += w_size
        b = weights[offset : offset + out_features]
        offset += out_features
        weight_tensors.append(w.astype(np.float32))
        bias_tensors.append(b.astype(np.float32))
        in_features = out_features

    return weight_tensors, bias_tensors


def _split_cnn_weights(arch: dict, weights: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    offset = 0
    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []
    height, width, channels = arch["input"]
    dense_in = 0

    for layer in arch["layers"]:
        layer_type = layer["type"]
        if layer_type == "conv2d":
            k = layer["kernel_size"]
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            out_c = layer["filters"]
            kernel_elems = k * k * channels
            w_flat = weights[offset : offset + kernel_elems * out_c]
            offset += kernel_elems * out_c
            b = weights[offset : offset + out_c]
            offset += out_c
            kernel = w_flat.reshape(out_c, k, k, channels)
            weight_tensors.append(kernel.astype(np.float32))
            bias_tensors.append(b.astype(np.float32))
            height = _out_dim(height, k, stride, pad_h)
            width = _out_dim(width, k, stride, pad_w)
            channels = out_c
        elif layer_type == "depthwise_conv2d":
            kh, kw = depthwise_kernel_hw(layer)
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            ch = layer["filters"]
            kernel_elems = kh * kw * ch
            w_flat = weights[offset : offset + kernel_elems]
            offset += kernel_elems
            b = weights[offset : offset + ch]
            offset += ch
            kernel = w_flat.reshape(ch, kh, kw)
            weight_tensors.append(kernel.astype(np.float32))
            bias_tensors.append(b.astype(np.float32))
            height = _out_dim(height, kh, stride, pad_h)
            width = _out_dim(width, kw, stride, pad_w)
        elif layer_type == "max_pool2d":
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            height = _out_dim(height, pool, stride, pad_h)
            width = _out_dim(width, pool, stride, pad_w)
        elif layer_type == "avg_pool2d":
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            height = _out_dim(height, pool, stride, pad_h)
            width = _out_dim(width, pool, stride, pad_w)
        elif layer_type == "batch_norm2d":
            ch = layer["channels"]
            scale = weights[offset : offset + ch]
            offset += ch
            bias = weights[offset : offset + ch]
            offset += ch
            weight_tensors.append(scale.astype(np.float32))
            bias_tensors.append(bias.astype(np.float32))
        elif layer_type == "layernorm2d":
            ch = layer["channels"]
            ln_w = weights[offset : offset + ch]
            offset += ch
            ln_b = weights[offset : offset + ch]
            offset += ch
            weight_tensors.append(ln_w.astype(np.float32))
            bias_tensors.append(ln_b.astype(np.float32))
        elif layer_type == "convnextv2_block":
            ch = layer["channels"]
            expanded = ch * 4
            pairs: list[tuple[tuple[int, ...], int]] = [
                ((ch, 7, 7), ch),
                ((ch,), ch),
                ((expanded, ch), expanded),
                ((expanded,), expanded),
                ((ch, expanded), ch),
            ]
            for w_shape, b_elems in pairs:
                w_elems = int(np.prod(w_shape))
                w_arr = weights[offset : offset + w_elems].reshape(w_shape)
                offset += w_elems
                b_arr = weights[offset : offset + b_elems]
                offset += b_elems
                weight_tensors.append(w_arr.astype(np.float32))
                bias_tensors.append(b_arr.astype(np.float32))
        elif layer_type == "mobilenetv4_uib":
            in_c = layer["in_channels"]
            out_c = layer["out_channels"]
            start_k = int(layer.get("start_dw_kernel", 0))
            middle_k = int(layer.get("middle_dw_kernel", 0))
            expand_c = _make_divisible(in_c * float(layer["expand_ratio"]), 8)
            if start_k:
                dw_elems = start_k * start_k * in_c
                w_arr = weights[offset : offset + dw_elems].reshape(in_c, start_k, start_k)
                offset += dw_elems
                b_arr = weights[offset : offset + in_c]
                offset += in_c
                weight_tensors.append(w_arr.astype(np.float32))
                bias_tensors.append(b_arr.astype(np.float32))
                scale = weights[offset : offset + in_c]
                offset += in_c
                beta = weights[offset : offset + in_c]
                offset += in_c
                weight_tensors.append(scale.astype(np.float32))
                bias_tensors.append(beta.astype(np.float32))

            expand_elems = expand_c * in_c
            expand_w = weights[offset : offset + expand_elems].reshape(expand_c, 1, 1, in_c)
            offset += expand_elems
            expand_b = weights[offset : offset + expand_c]
            offset += expand_c
            weight_tensors.append(expand_w.astype(np.float32))
            bias_tensors.append(expand_b.astype(np.float32))
            expand_scale = weights[offset : offset + expand_c]
            offset += expand_c
            expand_beta = weights[offset : offset + expand_c]
            offset += expand_c
            weight_tensors.append(expand_scale.astype(np.float32))
            bias_tensors.append(expand_beta.astype(np.float32))

            if middle_k:
                dw_elems = middle_k * middle_k * expand_c
                w_arr = weights[offset : offset + dw_elems].reshape(expand_c, middle_k, middle_k)
                offset += dw_elems
                b_arr = weights[offset : offset + expand_c]
                offset += expand_c
                weight_tensors.append(w_arr.astype(np.float32))
                bias_tensors.append(b_arr.astype(np.float32))
                scale = weights[offset : offset + expand_c]
                offset += expand_c
                beta = weights[offset : offset + expand_c]
                offset += expand_c
                weight_tensors.append(scale.astype(np.float32))
                bias_tensors.append(beta.astype(np.float32))

            proj_elems = out_c * expand_c
            proj_w = weights[offset : offset + proj_elems].reshape(out_c, 1, 1, expand_c)
            offset += proj_elems
            proj_b = weights[offset : offset + out_c]
            offset += out_c
            weight_tensors.append(proj_w.astype(np.float32))
            bias_tensors.append(proj_b.astype(np.float32))
            proj_scale = weights[offset : offset + out_c]
            offset += out_c
            proj_beta = weights[offset : offset + out_c]
            offset += out_c
            weight_tensors.append(proj_scale.astype(np.float32))
            bias_tensors.append(proj_beta.astype(np.float32))

            height, width = _uib_output_spatial(height, width, layer)
            channels = out_c
        elif layer_type == "resnet_basic_block":
            in_c = layer["in_channels"]
            out_c = layer["out_channels"]
            stride = int(layer.get("stride", 1))
            identity = stride == 1 and in_c == out_c

            conv1_elems = out_c * 3 * 3 * in_c
            conv1_w = weights[offset : offset + conv1_elems].reshape(out_c, 3, 3, in_c)
            offset += conv1_elems
            conv1_b = weights[offset : offset + out_c]
            offset += out_c
            weight_tensors.append(conv1_w.astype(np.float32))
            bias_tensors.append(conv1_b.astype(np.float32))
            bn1_scale = weights[offset : offset + out_c]
            offset += out_c
            bn1_beta = weights[offset : offset + out_c]
            offset += out_c
            weight_tensors.append(bn1_scale.astype(np.float32))
            bias_tensors.append(bn1_beta.astype(np.float32))

            conv2_elems = out_c * 3 * 3 * out_c
            conv2_w = weights[offset : offset + conv2_elems].reshape(out_c, 3, 3, out_c)
            offset += conv2_elems
            conv2_b = weights[offset : offset + out_c]
            offset += out_c
            weight_tensors.append(conv2_w.astype(np.float32))
            bias_tensors.append(conv2_b.astype(np.float32))
            bn2_scale = weights[offset : offset + out_c]
            offset += out_c
            bn2_beta = weights[offset : offset + out_c]
            offset += out_c
            weight_tensors.append(bn2_scale.astype(np.float32))
            bias_tensors.append(bn2_beta.astype(np.float32))

            if not identity:
                shortcut_elems = out_c * in_c
                shortcut_w = weights[offset : offset + shortcut_elems].reshape(out_c, 1, 1, in_c)
                offset += shortcut_elems
                shortcut_b = weights[offset : offset + out_c]
                offset += out_c
                weight_tensors.append(shortcut_w.astype(np.float32))
                bias_tensors.append(shortcut_b.astype(np.float32))
                shortcut_scale = weights[offset : offset + out_c]
                offset += out_c
                shortcut_beta = weights[offset : offset + out_c]
                offset += out_c
                weight_tensors.append(shortcut_scale.astype(np.float32))
                bias_tensors.append(shortcut_beta.astype(np.float32))

            height, width = _resnet_output_spatial(height, width, layer)
            channels = out_c
        elif layer_type == "flatten":
            dense_in = height * width * channels
        elif layer_type == "dense":
            out_f = layer["units"]
            w_size = dense_in * out_f
            dense_w = weights[offset : offset + w_size].reshape(out_f, dense_in)
            offset += w_size
            b = weights[offset : offset + out_f]
            offset += out_f
            weight_tensors.append(dense_w.astype(np.float32))
            bias_tensors.append(b.astype(np.float32))
            dense_in = out_f

    if offset != len(weights):
        raise ValueError(f"weight count mismatch: used {offset}, file has {len(weights)}")

    return weight_tensors, bias_tensors


def pack_random_cnn_weights(arch: dict, rng: np.random.Generator, scale: float = 0.05) -> np.ndarray:
    """Build a flat weight blob matching arch layer order (for fixtures)."""
    parts: list[np.ndarray] = []
    height, width, channels = arch["input"]
    dense_in = 0

    for layer in arch["layers"]:
        layer_type = layer["type"]
        if layer_type == "conv2d":
            k = layer["kernel_size"]
            out_c = layer["filters"]
            kernel_elems = k * k * channels
            parts.append(rng.standard_normal(kernel_elems * out_c, dtype=np.float32) * scale)
            parts.append(rng.standard_normal(out_c, dtype=np.float32) * 0.01)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            stride = layer.get("stride", 1)
            height = _out_dim(height, k, stride, pad_h)
            width = _out_dim(width, k, stride, pad_w)
            channels = out_c
        elif layer_type == "depthwise_conv2d":
            kh, kw = depthwise_kernel_hw(layer)
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            ch = layer["filters"]
            kernel_elems = kh * kw * ch
            parts.append(rng.standard_normal(kernel_elems, dtype=np.float32) * scale)
            parts.append(rng.standard_normal(ch, dtype=np.float32) * 0.01)
            height = _out_dim(height, kh, stride, pad_h)
            width = _out_dim(width, kw, stride, pad_w)
        elif layer_type in ("max_pool2d", "avg_pool2d"):
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            height = _out_dim(height, pool, stride, pad_h)
            width = _out_dim(width, pool, stride, pad_w)
        elif layer_type == "batch_norm2d":
            ch = layer["channels"]
            parts.append(np.ones(ch, dtype=np.float32))
            parts.append(np.zeros(ch, dtype=np.float32))
        elif layer_type == "layernorm2d":
            ch = layer["channels"]
            parts.append(np.ones(ch, dtype=np.float32))
            parts.append(np.zeros(ch, dtype=np.float32))
        elif layer_type == "convnextv2_block":
            ch = layer["channels"]
            expanded = ch * 4
            parts.append(rng.standard_normal(ch * 49, dtype=np.float32) * scale)
            parts.append(rng.standard_normal(ch, dtype=np.float32) * 0.01)
            parts.append(np.ones(ch, dtype=np.float32))
            parts.append(np.zeros(ch, dtype=np.float32))
            parts.append(rng.standard_normal(expanded * ch, dtype=np.float32) * scale)
            parts.append(rng.standard_normal(expanded, dtype=np.float32) * 0.01)
            parts.append(np.zeros(expanded, dtype=np.float32))
            parts.append(np.zeros(expanded, dtype=np.float32))
            parts.append(rng.standard_normal(ch * expanded, dtype=np.float32) * scale)
            parts.append(rng.standard_normal(ch, dtype=np.float32) * 0.01)
        elif layer_type == "mobilenetv4_uib":
            from .mobilenetv4_small import pack_uib_weights_flat

            parts.extend(
                pack_uib_weights_flat(
                    rng,
                    in_channels=layer["in_channels"],
                    out_channels=layer["out_channels"],
                    start_dw_kernel=int(layer.get("start_dw_kernel", 0)),
                    middle_dw_kernel=int(layer.get("middle_dw_kernel", 0)),
                    expand_ratio=float(layer["expand_ratio"]),
                    scale=scale,
                )
            )
            height, width = _uib_output_spatial(height, width, layer)
            channels = layer["out_channels"]
        elif layer_type == "resnet_basic_block":
            _append_resnet_basic_block_weights(
                parts,
                rng,
                in_c=layer["in_channels"],
                out_c=layer["out_channels"],
                stride=int(layer.get("stride", 1)),
                scale=scale,
            )
            height, width = _resnet_output_spatial(height, width, layer)
            channels = layer["out_channels"]
        elif layer_type == "flatten":
            dense_in = height * width * channels
        elif layer_type == "dense":
            out_f = layer["units"]
            parts.append(rng.standard_normal(dense_in * out_f, dtype=np.float32) * scale)
            parts.append(rng.standard_normal(out_f, dtype=np.float32) * 0.01)
            dense_in = out_f

    return np.concatenate(parts).astype(np.float32)


def _arch_to_spec(arch: dict, weights: np.ndarray) -> ModelSpec:
    layers: list[LayerSpec] = []
    if len(arch["layers"]) > MAX_LAYERS:
        raise ValueError(f"arch has {len(arch['layers'])} layers; max is {MAX_LAYERS}")
    for layer in arch["layers"]:
        layer_type = layer["type"]
        act = activation_from_name(layer.get("activation", "none"))
        alpha = float(layer.get("alpha", 0.01))
        if layer_type == "dense":
            layers.append(LayerSpec(kind="dense", units=layer["units"], activation=act, alpha=alpha))
        elif layer_type == "conv2d":
            layers.append(
                LayerSpec(
                    kind="conv2d",
                    kernel_size=layer["kernel_size"],
                    stride=layer.get("stride", 1),
                    filters=layer["filters"],
                    activation=act,
                    alpha=alpha,
                    pad_h=layer.get("pad_h", 0),
                    pad_w=layer.get("pad_w", 0),
                    pad_h_end=layer.get("pad_h_end", layer.get("pad_h", 0)),
                    pad_w_end=layer.get("pad_w_end", layer.get("pad_w", 0)),
                )
            )
        elif layer_type == "depthwise_conv2d":
            kh, kw = depthwise_kernel_hw(layer)
            layers.append(
                LayerSpec(
                    kind="depthwise_conv2d",
                    kernel_h=kh,
                    kernel_w=kw,
                    stride=layer.get("stride", 1),
                    filters=layer["filters"],
                    activation=act,
                    alpha=alpha,
                    pad_h=layer.get("pad_h", 0),
                    pad_w=layer.get("pad_w", 0),
                    pad_h_end=layer.get("pad_h_end", layer.get("pad_h", 0)),
                    pad_w_end=layer.get("pad_w_end", layer.get("pad_w", 0)),
                )
            )
        elif layer_type == "max_pool2d":
            layers.append(
                LayerSpec(
                    kind="max_pool2d",
                    pool_size=layer["pool_size"],
                    stride=layer.get("stride", layer["pool_size"]),
                    pad_h=layer.get("pad_h", 0),
                    pad_w=layer.get("pad_w", 0),
                    pool_w=layer.get("pool_w", layer["pool_size"]),
                    pad_h_end=layer.get("pad_h_end", layer.get("pad_h", 0)),
                    pad_w_end=layer.get("pad_w_end", layer.get("pad_w", 0)),
                )
            )
        elif layer_type == "avg_pool2d":
            layers.append(
                LayerSpec(
                    kind="avg_pool2d",
                    pool_size=layer["pool_size"],
                    stride=layer.get("stride", layer["pool_size"]),
                    pad_h=layer.get("pad_h", 0),
                    pad_w=layer.get("pad_w", 0),
                    pool_w=layer.get("pool_w", layer["pool_size"]),
                    pad_h_end=layer.get("pad_h_end", layer.get("pad_h", 0)),
                    pad_w_end=layer.get("pad_w_end", layer.get("pad_w", 0)),
                )
            )
        elif layer_type == "batch_norm2d":
            layers.append(LayerSpec(kind="batch_norm2d", channels=layer["channels"]))
        elif layer_type == "layernorm2d":
            layers.append(
                LayerSpec(
                    kind="layernorm2d",
                    channels=layer["channels"],
                    eps=float(layer.get("eps", 1e-6)),
                )
            )
        elif layer_type == "convnextv2_block":
            layers.append(
                LayerSpec(
                    kind="convnextv2_block",
                    channels=layer["channels"],
                    eps=float(layer.get("eps", 1e-6)),
                )
            )
        elif layer_type == "mobilenetv4_uib":
            layers.append(
                LayerSpec(
                    kind="mobilenetv4_uib",
                    in_channels=layer["in_channels"],
                    out_channels=layer["out_channels"],
                    start_dw_kernel=int(layer.get("start_dw_kernel", 0)),
                    middle_dw_kernel=int(layer.get("middle_dw_kernel", 0)),
                    stride=int(layer.get("stride", 1)),
                    middle_dw_downsample=int(layer.get("middle_dw_downsample", 1)),
                    expand_ratio=float(layer["expand_ratio"]),
                )
            )
        elif layer_type == "resnet_basic_block":
            layers.append(
                LayerSpec(
                    kind="resnet_basic_block",
                    in_channels=layer["in_channels"],
                    out_channels=layer["out_channels"],
                    stride=int(layer.get("stride", 1)),
                )
            )
        elif layer_type == "flatten":
            layers.append(LayerSpec(kind="flatten"))
        else:
            raise ValueError(f"unsupported layer type: {layer_type}")

    if arch["network"] == "mlp":
        weight_tensors, bias_tensors = _split_mlp_weights(arch, weights)
    else:
        weight_tensors, bias_tensors = _split_cnn_weights(arch, weights)

    return ModelSpec(
        network=arch["network"],
        input_shape=list(arch["input"]),
        layers=layers,
        weight_tensors=weight_tensors,
        bias_tensors=bias_tensors,
    )


def write_nk_from_arch(
    arch: dict,
    weights: np.ndarray,
    output_path: str | Path,
    tests: RegressionSuite | None = None,
) -> Path:
    output_path = Path(output_path)
    spec = _arch_to_spec(arch, weights)
    spec.tests = tests
    write_nk(output_path, spec)
    return output_path


def arch_to_nk_bytes(
    arch: dict,
    weights: np.ndarray,
    tests: RegressionSuite | None = None,
) -> bytes:
    spec = _arch_to_spec(arch, weights)
    spec.tests = tests
    return write_nk_bytes(spec)
