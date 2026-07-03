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


def _out_dim(in_dim: int, kernel: int, stride: int, pad: int = 0) -> int:
    return (in_dim + 2 * pad - kernel) // stride + 1


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
) -> np.ndarray:
    """kernel shape (out_c, k, k, in_c); inp (H, W, C)."""
    h, w, in_c = inp.shape
    out_c, k, _, _ = kernel.shape
    out_h = _out_dim(h, k, stride, pad_h)
    out_w = _out_dim(w, k, stride, pad_w)
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
) -> np.ndarray:
    """kernel shape (channels, kh, kw); inp (H, W, C)."""
    h, w, channels = inp.shape
    _, kernel_h, kernel_w = kernel.shape
    out_h = _out_dim(h, kernel_h, stride, pad_h)
    out_w = _out_dim(w, kernel_w, stride, pad_w)
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
    pool_size: int,
    stride: int,
    pad_h: int = 0,
    pad_w: int = 0,
) -> np.ndarray:
    h, w, channels = inp.shape
    out_h = _out_dim(h, pool_size, stride, pad_h)
    out_w = _out_dim(w, pool_size, stride, pad_w)
    out = np.full((out_h, out_w, channels), -np.finfo(np.float32).max, dtype=np.float32)
    for c in range(channels):
        for oh in range(out_h):
            for ow in range(out_w):
                max_val = -np.finfo(np.float32).max
                for kh in range(pool_size):
                    for kw in range(pool_size):
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
    pool_size: int,
    stride: int,
    pad_h: int = 0,
    pad_w: int = 0,
) -> np.ndarray:
    h, w, channels = inp.shape
    out_h = _out_dim(h, pool_size, stride, pad_h)
    out_w = _out_dim(w, pool_size, stride, pad_w)
    out = np.zeros((out_h, out_w, channels), dtype=np.float32)
    for c in range(channels):
        for oh in range(out_h):
            for ow in range(out_w):
                total = 0.0
                count = 0
                for kh in range(pool_size):
                    for kw in range(pool_size):
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
            x = _conv_nhwc(x, kernel, bias, stride=stride, pad_h=pad_h, pad_w=pad_w)
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
            x = _depthwise_conv_nhwc(x, kernel, bias, stride=stride, pad_h=pad_h, pad_w=pad_w)
            x = _activate(x, layer.get("activation", "none"), alpha=float(layer.get("alpha", 0.01)))
            h, w, channels = x.shape
        elif layer_type == "max_pool2d":
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            x = _max_pool_nhwc(x, pool_size=pool, stride=stride, pad_h=pad_h, pad_w=pad_w)
            h, w, channels = x.shape
        elif layer_type == "avg_pool2d":
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            x = _avg_pool_nhwc(x, pool_size=pool, stride=stride, pad_h=pad_h, pad_w=pad_w)
            h, w, channels = x.shape
        elif layer_type == "batch_norm2d":
            ch = layer["channels"]
            scale = weights[offset : offset + ch]
            offset += ch
            bias = weights[offset : offset + ch]
            offset += ch
            x = _batch_norm_nhwc(x, scale, bias)
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
