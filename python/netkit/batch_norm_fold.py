"""BatchNorm folding helpers for PyTorch and ONNX export."""

from __future__ import annotations

import numpy as np


def fold_batch_norm_params(
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    *,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold BN running stats into per-channel scale/bias (netkit batch_norm2d layout)."""
    inv_std = 1.0 / np.sqrt(var.astype(np.float64) + eps)
    scale = (gamma.astype(np.float64) * inv_std).astype(np.float32)
    bias = (beta.astype(np.float64) - mean.astype(np.float64) * scale).astype(np.float32)
    return scale, bias


def fold_conv_batch_norm(
    conv_w: np.ndarray,
    conv_b: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    *,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold BN into conv weights (netkit conv layout [O,Kh,Kw,I] or [O,Kh,Kw])."""
    scale, bias = fold_batch_norm_params(gamma, beta, mean, var, eps=eps)
    if conv_w.ndim == 4:
        w = conv_w * scale.reshape(-1, 1, 1, 1)
    elif conv_w.ndim == 3:
        w = conv_w * scale.reshape(-1, 1, 1)
    else:
        raise ValueError(f"unsupported conv weight rank: {conv_w.ndim}")
    b = conv_b.astype(np.float32) * scale + bias
    return w.astype(np.float32), b.astype(np.float32)
