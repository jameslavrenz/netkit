"""NumPy reference forward pass matching netkit C++ runtime layout (NHWC conv, [out,in] dense)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .cnn_layers import depthwise_kernel_hw


def _activate(x: np.ndarray, activation: str, *, alpha: float = 0.01) -> np.ndarray:
    if activation == "relu":
        return np.maximum(x, 0.0)
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if activation == "tanh":
        return np.tanh(x)
    if activation == "leaky_relu":
        return np.where(x > 0.0, x, alpha * x)
    if activation == "relu6":
        return np.clip(x, 0.0, 6.0)
    if activation == "softmax":
        shifted = x - np.max(x)
        exp = np.exp(shifted)
        return exp / np.sum(exp)
    return x


def _out_dim(in_dim: int, kernel: int, stride: int, pad_before: int = 0, pad_after: int | None = None) -> int:
    if pad_after is None:
        pad_after = pad_before
    return (in_dim + pad_before + pad_after - kernel) // stride + 1


def pack_mlp_weights(layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for w, b in layers:
        parts.append(w.astype(np.float32).reshape(-1))
        parts.append(b.astype(np.float32).reshape(-1))
    return np.concatenate(parts)


def pack_cnn_weights(tensors: list[tuple[np.ndarray, np.ndarray] | None]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for item in tensors:
        if item is None:
            continue
        w, b = item
        parts.append(w.astype(np.float32).reshape(-1))
        parts.append(b.astype(np.float32).reshape(-1))
    return np.concatenate(parts)


def _conv_nhwc(
    inp: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    *,
    stride: int,
    pad_h: int = 0,
    pad_w: int = 0,
    pad_h_end: int | None = None,
    pad_w_end: int | None = None,
) -> np.ndarray:
    """kernel shape (out_c, k, k, in_c); inp (H, W, C)."""
    if pad_h_end is None:
        pad_h_end = pad_h
    if pad_w_end is None:
        pad_w_end = pad_w
    h, w, in_c = inp.shape
    out_c, k, _, _ = kernel.shape
    out_h = _out_dim(h, k, stride, pad_h, pad_h_end)
    out_w = _out_dim(w, k, stride, pad_w, pad_w_end)
    out = np.zeros((out_h, out_w, out_c), dtype=np.float32)
    for oc in range(out_c):
        for oh in range(out_h):
            for ow in range(out_w):
                total = float(bias[oc])
                for kh in range(k):
                    for kw in range(k):
                        for ic in range(in_c):
                            ih = oh * stride + kh - pad_h
                            iw = ow * stride + kw - pad_w
                            if ih < 0 or iw < 0 or ih >= h or iw >= w:
                                continue
                            total += inp[ih, iw, ic] * kernel[oc, kh, kw, ic]
                out[oh, ow, oc] = total
    return out


def _depthwise_conv_nhwc(
    inp: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    *,
    stride: int,
    pad_h: int = 0,
    pad_w: int = 0,
    pad_h_end: int | None = None,
    pad_w_end: int | None = None,
) -> np.ndarray:
    """kernel shape (channels, kh, kw); inp (H, W, C)."""
    if pad_h_end is None:
        pad_h_end = pad_h
    if pad_w_end is None:
        pad_w_end = pad_w
    h, w, channels = inp.shape
    _, kernel_h, kernel_w = kernel.shape
    out_h = _out_dim(h, kernel_h, stride, pad_h, pad_h_end)
    out_w = _out_dim(w, kernel_w, stride, pad_w, pad_w_end)
    out = np.zeros((out_h, out_w, channels), dtype=np.float32)
    for c in range(channels):
        for oh in range(out_h):
            for ow in range(out_w):
                total = float(bias[c])
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        ih = oh * stride + kh - pad_h
                        iw = ow * stride + kw - pad_w
                        if ih < 0 or iw < 0 or ih >= h or iw >= w:
                            continue
                        total += inp[ih, iw, c] * kernel[c, kh, kw]
                out[oh, ow, c] = total
    return out


def _max_pool_nhwc(
    inp: np.ndarray,
    *,
    pool_h: int,
    pool_w: int | None = None,
    stride: int,
    pad_h: int = 0,
    pad_w: int = 0,
    pad_h_end: int | None = None,
    pad_w_end: int | None = None,
) -> np.ndarray:
    if pool_w is None:
        pool_w = pool_h
    if pad_h_end is None:
        pad_h_end = pad_h
    if pad_w_end is None:
        pad_w_end = pad_w
    h, w, channels = inp.shape
    out_h = _out_dim(h, pool_h, stride, pad_h, pad_h_end)
    out_w = _out_dim(w, pool_w, stride, pad_w, pad_w_end)
    out = np.full((out_h, out_w, channels), -np.finfo(np.float32).max, dtype=np.float32)
    for c in range(channels):
        for oh in range(out_h):
            for ow in range(out_w):
                max_val = -np.finfo(np.float32).max
                for kh in range(pool_h):
                    for kw in range(pool_w):
                        ih = oh * stride + kh - pad_h
                        iw = ow * stride + kw - pad_w
                        if ih < 0 or iw < 0 or ih >= h or iw >= w:
                            continue
                        max_val = max(max_val, inp[ih, iw, c])
                out[oh, ow, c] = max_val
    return out


def _avg_pool_nhwc(
    inp: np.ndarray,
    *,
    pool_h: int,
    pool_w: int | None = None,
    stride: int,
    pad_h: int = 0,
    pad_w: int = 0,
    pad_h_end: int | None = None,
    pad_w_end: int | None = None,
) -> np.ndarray:
    if pool_w is None:
        pool_w = pool_h
    if pad_h_end is None:
        pad_h_end = pad_h
    if pad_w_end is None:
        pad_w_end = pad_w
    h, w, channels = inp.shape
    out_h = _out_dim(h, pool_h, stride, pad_h, pad_h_end)
    out_w = _out_dim(w, pool_w, stride, pad_w, pad_w_end)
    out = np.zeros((out_h, out_w, channels), dtype=np.float32)
    for c in range(channels):
        for oh in range(out_h):
            for ow in range(out_w):
                total = 0.0
                count = 0
                for kh in range(pool_h):
                    for kw in range(pool_w):
                        ih = oh * stride + kh - pad_h
                        iw = ow * stride + kw - pad_w
                        if ih < 0 or iw < 0 or ih >= h or iw >= w:
                            continue
                        total += float(inp[ih, iw, c])
                        count += 1
                out[oh, ow, c] = total / count if count > 0 else 0.0
    return out


def _batch_norm_nhwc(inp: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    out = np.empty_like(inp, dtype=np.float32)
    channels = inp.shape[2]
    flat = inp.reshape(-1, channels)
    out_flat = out.reshape(-1, channels)
    for c in range(channels):
        out_flat[:, c] = flat[:, c] * scale[c] + bias[c]
    return out


def _gelu(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))).astype(np.float32)


def _layer_norm_nhwc(inp: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    out = np.empty_like(inp, dtype=np.float32)
    height, width, channels = inp.shape
    for oh in range(height):
        for ow in range(width):
            pixel = inp[oh, ow]
            mean = float(np.mean(pixel))
            variance = float(np.mean((pixel - mean) ** 2))
            inv_std = 1.0 / np.sqrt(variance + eps)
            out[oh, ow] = (pixel - mean) * inv_std * weight + bias
    return out


def _grn_nhwc(inp: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    out = inp.copy()
    height, width, channels = inp.shape
    spatial = height * width
    flat = inp.reshape(spatial, channels)
    norms = np.sqrt(np.sum(flat * flat, axis=0))
    mean_norm = float(np.mean(norms))
    denom = mean_norm + eps
    for c in range(channels):
        nx = norms[c] / denom
        out[:, :, c] = gamma[c] * (inp[:, :, c] * nx) + beta[c] + inp[:, :, c]
    return out.astype(np.float32)


def _convnextv2_block_nhwc(
    inp: np.ndarray,
    *,
    channels: int,
    eps: float,
    dw_w: np.ndarray,
    dw_b: np.ndarray,
    ln_w: np.ndarray,
    ln_b: np.ndarray,
    pw1_w: np.ndarray,
    pw1_b: np.ndarray,
    grn_gamma: np.ndarray,
    grn_beta: np.ndarray,
    pw2_w: np.ndarray,
    pw2_b: np.ndarray,
) -> np.ndarray:
    expanded = channels * 4
    branch = _depthwise_conv_nhwc(inp, dw_w, dw_b, stride=1, pad_h=3, pad_w=3)
    branch = _layer_norm_nhwc(branch, ln_w, ln_b, eps)
    height, width, _ = branch.shape
    mlp = np.zeros((height, width, expanded), dtype=np.float32)
    for oh in range(height):
        for ow in range(width):
            mlp[oh, ow] = pw1_w @ branch[oh, ow] + pw1_b
    mlp = _gelu(mlp)
    mlp = _grn_nhwc(mlp, grn_gamma, grn_beta, eps)
    out = np.zeros_like(branch, dtype=np.float32)
    for oh in range(height):
        for ow in range(width):
            out[oh, ow] = pw2_w @ mlp[oh, ow] + pw2_b
    return (out + inp).astype(np.float32)


def _make_divisible(value: float, divisor: int = 8) -> int:
    rounded = int(value + divisor / 2) // divisor * divisor
    result = max(divisor, rounded)
    if result < int(0.9 * value):
        result += divisor
    return result


def _uib_dw_stride(stride: int, middle_k: int, middle_dw_downsample: bool) -> int:
    return 1 if middle_dw_downsample and middle_k > 0 else stride


def _uib_middle_stride(stride: int, middle_k: int, middle_dw_downsample: bool) -> int:
    if middle_k <= 0:
        return 1
    return stride if middle_dw_downsample else 1


def _mobilenetv4_uib_nhwc(
    inp: np.ndarray,
    *,
    in_channels: int,
    out_channels: int,
    start_dw_kernel: int,
    middle_dw_kernel: int,
    stride: int,
    expand_ratio: float,
    middle_dw_downsample: bool,
    start_dw_w: np.ndarray | None,
    start_dw_b: np.ndarray | None,
    start_bn_scale: np.ndarray | None,
    start_bn_bias: np.ndarray | None,
    expand_w: np.ndarray,
    expand_b: np.ndarray,
    expand_bn_scale: np.ndarray,
    expand_bn_bias: np.ndarray,
    middle_dw_w: np.ndarray | None,
    middle_dw_b: np.ndarray | None,
    middle_bn_scale: np.ndarray | None,
    middle_bn_bias: np.ndarray | None,
    proj_w: np.ndarray,
    proj_b: np.ndarray,
    proj_bn_scale: np.ndarray,
    proj_bn_bias: np.ndarray,
) -> np.ndarray:
    expand_c = _make_divisible(in_channels * expand_ratio, 8)
    x = inp.astype(np.float32)
    residual = x.copy() if stride == 1 and in_channels == out_channels else None

    if start_dw_kernel:
        pad = (start_dw_kernel - 1) // 2
        dw_stride = _uib_dw_stride(stride, middle_dw_kernel, middle_dw_downsample)
        x = _depthwise_conv_nhwc(
            x, start_dw_w, start_dw_b, stride=dw_stride, pad_h=pad, pad_w=pad
        )
        x = _batch_norm_nhwc(x, start_bn_scale, start_bn_bias)

    expand_kernel = expand_w.reshape(expand_c, 1, 1, in_channels)
    x = _conv_nhwc(x, expand_kernel, expand_b, stride=1, pad_h=0, pad_w=0)
    x = _batch_norm_nhwc(x, expand_bn_scale, expand_bn_bias)
    x = np.maximum(x, 0.0)

    if middle_dw_kernel:
        pad = (middle_dw_kernel - 1) // 2
        mid_stride = _uib_middle_stride(stride, middle_dw_kernel, middle_dw_downsample)
        x = _depthwise_conv_nhwc(
            x, middle_dw_w, middle_dw_b, stride=mid_stride, pad_h=pad, pad_w=pad
        )
        x = _batch_norm_nhwc(x, middle_bn_scale, middle_bn_bias)
        x = np.maximum(x, 0.0)

    proj_kernel = proj_w.reshape(out_channels, 1, 1, expand_c)
    x = _conv_nhwc(x, proj_kernel, proj_b, stride=1, pad_h=0, pad_w=0)
    x = _batch_norm_nhwc(x, proj_bn_scale, proj_bn_bias)

    if residual is not None:
        x = x + residual
    return x.astype(np.float32)


def _resnet_basic_block_nhwc(
    inp: np.ndarray,
    *,
    in_channels: int,
    out_channels: int,
    stride: int,
    conv1_w: np.ndarray,
    conv1_b: np.ndarray,
    bn1_scale: np.ndarray,
    bn1_bias: np.ndarray,
    conv2_w: np.ndarray,
    conv2_b: np.ndarray,
    bn2_scale: np.ndarray,
    bn2_bias: np.ndarray,
    shortcut_w: np.ndarray | None,
    shortcut_b: np.ndarray | None,
    shortcut_bn_scale: np.ndarray | None,
    shortcut_bn_bias: np.ndarray | None,
) -> np.ndarray:
    x = inp.astype(np.float32)
    identity = stride == 1 and in_channels == out_channels

    conv1_kernel = conv1_w.reshape(out_channels, 3, 3, in_channels)
    out = _conv_nhwc(x, conv1_kernel, conv1_b, stride=stride, pad_h=1, pad_w=1)
    out = _batch_norm_nhwc(out, bn1_scale, bn1_bias)
    out = np.maximum(out, 0.0)

    conv2_kernel = conv2_w.reshape(out_channels, 3, 3, out_channels)
    out = _conv_nhwc(out, conv2_kernel, conv2_b, stride=1, pad_h=1, pad_w=1)
    out = _batch_norm_nhwc(out, bn2_scale, bn2_bias)

    if identity:
        residual = x
    else:
        assert shortcut_w is not None and shortcut_b is not None
        assert shortcut_bn_scale is not None and shortcut_bn_bias is not None
        shortcut_kernel = shortcut_w.reshape(out_channels, 1, 1, in_channels)
        residual = _conv_nhwc(x, shortcut_kernel, shortcut_b, stride=stride, pad_h=0, pad_w=0)
        residual = _batch_norm_nhwc(residual, shortcut_bn_scale, shortcut_bn_bias)

    out = out + residual
    return np.maximum(out, 0.0).astype(np.float32)


def forward_mlp(flat_input: np.ndarray, arch: dict[str, Any], weights: np.ndarray) -> list[float]:
    x = np.asarray(flat_input, dtype=np.float32).reshape(-1)
    offset = 0
    in_features = arch["input"][1]

    for layer in arch["layers"]:
        out_features = layer["units"]
        w_size = in_features * out_features
        w = weights[offset : offset + w_size].reshape(out_features, in_features)
        offset += w_size
        b = weights[offset : offset + out_features]
        offset += out_features
        x = w @ x + b
        x = _activate(x, layer.get("activation", "none"), alpha=float(layer.get("alpha", 0.01)))
        in_features = out_features

    return x.astype(np.float32).reshape(-1).tolist()


def forward_cnn(flat_input: np.ndarray, arch: dict[str, Any], weights: np.ndarray) -> list[float]:
    h, w, channels = arch["input"]
    x = np.asarray(flat_input, dtype=np.float32).reshape(h, w, channels)
    offset = 0
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
            kernel = weights[offset : offset + kernel_elems * out_c].reshape(out_c, k, k, channels)
            offset += kernel_elems * out_c
            bias = weights[offset : offset + out_c]
            offset += out_c
            pad_h_end = layer.get("pad_h_end", pad_h)
            pad_w_end = layer.get("pad_w_end", pad_w)
            x = _conv_nhwc(
                x,
                kernel,
                bias,
                stride=stride,
                pad_h=pad_h,
                pad_w=pad_w,
                pad_h_end=pad_h_end,
                pad_w_end=pad_w_end,
            )
            x = _activate(x, layer.get("activation", "none"), alpha=float(layer.get("alpha", 0.01)))
            h, w, channels = x.shape
        elif layer_type == "depthwise_conv2d":
            kh, kw = depthwise_kernel_hw(layer)
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            ch = layer["filters"]
            kernel_elems = kh * kw * ch
            kernel = weights[offset : offset + kernel_elems].reshape(ch, kh, kw)
            offset += kernel_elems
            bias = weights[offset : offset + ch]
            offset += ch
            pad_h_end = layer.get("pad_h_end", pad_h)
            pad_w_end = layer.get("pad_w_end", pad_w)
            x = _depthwise_conv_nhwc(
                x,
                kernel,
                bias,
                stride=stride,
                pad_h=pad_h,
                pad_w=pad_w,
                pad_h_end=pad_h_end,
                pad_w_end=pad_w_end,
            )
            x = _activate(x, layer.get("activation", "none"), alpha=float(layer.get("alpha", 0.01)))
            h, w, channels = x.shape
        elif layer_type == "max_pool2d":
            pool_h = layer["pool_size"]
            pool_w = layer.get("pool_w", pool_h)
            stride = layer.get("stride", pool_h)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            pad_h_end = layer.get("pad_h_end", pad_h)
            pad_w_end = layer.get("pad_w_end", pad_w)
            x = _max_pool_nhwc(
                x,
                pool_h=pool_h,
                pool_w=pool_w,
                stride=stride,
                pad_h=pad_h,
                pad_w=pad_w,
                pad_h_end=pad_h_end,
                pad_w_end=pad_w_end,
            )
            h, w, channels = x.shape
        elif layer_type == "avg_pool2d":
            pool_h = layer["pool_size"]
            pool_w = layer.get("pool_w", pool_h)
            stride = layer.get("stride", pool_h)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            pad_h_end = layer.get("pad_h_end", pad_h)
            pad_w_end = layer.get("pad_w_end", pad_w)
            x = _avg_pool_nhwc(
                x,
                pool_h=pool_h,
                pool_w=pool_w,
                stride=stride,
                pad_h=pad_h,
                pad_w=pad_w,
                pad_h_end=pad_h_end,
                pad_w_end=pad_w_end,
            )
            h, w, channels = x.shape
        elif layer_type == "batch_norm2d":
            ch = layer["channels"]
            scale = weights[offset : offset + ch]
            offset += ch
            bias = weights[offset : offset + ch]
            offset += ch
            x = _batch_norm_nhwc(x, scale, bias)
        elif layer_type == "layernorm2d":
            ch = layer["channels"]
            ln_w = weights[offset : offset + ch]
            offset += ch
            ln_b = weights[offset : offset + ch]
            offset += ch
            x = _layer_norm_nhwc(x, ln_w, ln_b, float(layer.get("eps", 1e-6)))
        elif layer_type == "convnextv2_block":
            ch = layer["channels"]
            expanded = ch * 4
            dw_w = weights[offset : offset + ch * 49].reshape(ch, 7, 7)
            offset += ch * 49
            dw_b = weights[offset : offset + ch]
            offset += ch
            ln_w = weights[offset : offset + ch]
            offset += ch
            ln_b = weights[offset : offset + ch]
            offset += ch
            pw1_w = weights[offset : offset + expanded * ch].reshape(expanded, ch)
            offset += expanded * ch
            pw1_b = weights[offset : offset + expanded]
            offset += expanded
            grn_gamma = weights[offset : offset + expanded]
            offset += expanded
            grn_beta = weights[offset : offset + expanded]
            offset += expanded
            pw2_w = weights[offset : offset + ch * expanded].reshape(ch, expanded)
            offset += ch * expanded
            pw2_b = weights[offset : offset + ch]
            offset += ch
            x = _convnextv2_block_nhwc(
                x,
                channels=ch,
                eps=float(layer.get("eps", 1e-6)),
                dw_w=dw_w,
                dw_b=dw_b,
                ln_w=ln_w,
                ln_b=ln_b,
                pw1_w=pw1_w,
                pw1_b=pw1_b,
                grn_gamma=grn_gamma,
                grn_beta=grn_beta,
                pw2_w=pw2_w,
                pw2_b=pw2_b,
            )
            h, w, channels = x.shape
        elif layer_type == "mobilenetv4_uib":
            in_c = layer["in_channels"]
            out_c = layer["out_channels"]
            start_k = int(layer.get("start_dw_kernel", 0))
            middle_k = int(layer.get("middle_dw_kernel", 0))
            expand_c = _make_divisible(in_c * float(layer["expand_ratio"]), 8)
            middle_down = bool(layer.get("middle_dw_downsample", 1))

            start_dw_w = start_dw_b = start_bn_scale = start_bn_bias = None
            if start_k:
                start_dw_w = weights[offset : offset + in_c * start_k * start_k].reshape(in_c, start_k, start_k)
                offset += in_c * start_k * start_k
                start_dw_b = weights[offset : offset + in_c]
                offset += in_c
                start_bn_scale = weights[offset : offset + in_c]
                offset += in_c
                start_bn_bias = weights[offset : offset + in_c]
                offset += in_c

            expand_w = weights[offset : offset + expand_c * in_c].reshape(expand_c, in_c)
            offset += expand_c * in_c
            expand_b = weights[offset : offset + expand_c]
            offset += expand_c
            expand_bn_scale = weights[offset : offset + expand_c]
            offset += expand_c
            expand_bn_bias = weights[offset : offset + expand_c]
            offset += expand_c

            middle_dw_w = middle_dw_b = middle_bn_scale = middle_bn_bias = None
            if middle_k:
                middle_dw_w = weights[offset : offset + expand_c * middle_k * middle_k].reshape(
                    expand_c, middle_k, middle_k
                )
                offset += expand_c * middle_k * middle_k
                middle_dw_b = weights[offset : offset + expand_c]
                offset += expand_c
                middle_bn_scale = weights[offset : offset + expand_c]
                offset += expand_c
                middle_bn_bias = weights[offset : offset + expand_c]
                offset += expand_c

            proj_w = weights[offset : offset + out_c * expand_c].reshape(out_c, expand_c)
            offset += out_c * expand_c
            proj_b = weights[offset : offset + out_c]
            offset += out_c
            proj_bn_scale = weights[offset : offset + out_c]
            offset += out_c
            proj_bn_bias = weights[offset : offset + out_c]
            offset += out_c

            x = _mobilenetv4_uib_nhwc(
                x,
                in_channels=in_c,
                out_channels=out_c,
                start_dw_kernel=start_k,
                middle_dw_kernel=middle_k,
                stride=int(layer.get("stride", 1)),
                expand_ratio=float(layer["expand_ratio"]),
                middle_dw_downsample=middle_down,
                start_dw_w=start_dw_w,
                start_dw_b=start_dw_b,
                start_bn_scale=start_bn_scale,
                start_bn_bias=start_bn_bias,
                expand_w=expand_w,
                expand_b=expand_b,
                expand_bn_scale=expand_bn_scale,
                expand_bn_bias=expand_bn_bias,
                middle_dw_w=middle_dw_w,
                middle_dw_b=middle_dw_b,
                middle_bn_scale=middle_bn_scale,
                middle_bn_bias=middle_bn_bias,
                proj_w=proj_w,
                proj_b=proj_b,
                proj_bn_scale=proj_bn_scale,
                proj_bn_bias=proj_bn_bias,
            )
            h, w, channels = x.shape
        elif layer_type == "resnet_basic_block":
            in_c = layer["in_channels"]
            out_c = layer["out_channels"]
            stride = int(layer.get("stride", 1))
            identity = stride == 1 and in_c == out_c

            conv1_elems = out_c * 3 * 3 * in_c
            conv1_w = weights[offset : offset + conv1_elems]
            offset += conv1_elems
            conv1_b = weights[offset : offset + out_c]
            offset += out_c
            bn1_scale = weights[offset : offset + out_c]
            offset += out_c
            bn1_bias = weights[offset : offset + out_c]
            offset += out_c

            conv2_elems = out_c * 3 * 3 * out_c
            conv2_w = weights[offset : offset + conv2_elems]
            offset += conv2_elems
            conv2_b = weights[offset : offset + out_c]
            offset += out_c
            bn2_scale = weights[offset : offset + out_c]
            offset += out_c
            bn2_bias = weights[offset : offset + out_c]
            offset += out_c

            shortcut_w = shortcut_b = shortcut_bn_scale = shortcut_bn_bias = None
            if not identity:
                shortcut_elems = out_c * in_c
                shortcut_w = weights[offset : offset + shortcut_elems]
                offset += shortcut_elems
                shortcut_b = weights[offset : offset + out_c]
                offset += out_c
                shortcut_bn_scale = weights[offset : offset + out_c]
                offset += out_c
                shortcut_bn_bias = weights[offset : offset + out_c]
                offset += out_c

            x = _resnet_basic_block_nhwc(
                x,
                in_channels=in_c,
                out_channels=out_c,
                stride=stride,
                conv1_w=conv1_w,
                conv1_b=conv1_b,
                bn1_scale=bn1_scale,
                bn1_bias=bn1_bias,
                conv2_w=conv2_w,
                conv2_b=conv2_b,
                bn2_scale=bn2_scale,
                bn2_bias=bn2_bias,
                shortcut_w=shortcut_w,
                shortcut_b=shortcut_b,
                shortcut_bn_scale=shortcut_bn_scale,
                shortcut_bn_bias=shortcut_bn_bias,
            )
            h, w, channels = x.shape
        elif layer_type == "flatten":
            x = x.reshape(-1)
            dense_in = x.size
        elif layer_type == "dense":
            out_f = layer["units"]
            w_size = dense_in * out_f
            w = weights[offset : offset + w_size].reshape(out_f, dense_in)
            offset += w_size
            b = weights[offset : offset + out_f]
            offset += out_f
            x = w @ x + b
            x = _activate(x, layer.get("activation", "none"), alpha=float(layer.get("alpha", 0.01)))
            dense_in = out_f

    if offset != len(weights):
        raise ValueError(f"weight count mismatch: used {offset}, file has {len(weights)}")

    return x.astype(np.float32).reshape(-1).tolist()
