"""Pack PyTorch weights into netkit flat layout and verify against the reference forward."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .reference_forward import forward_cnn, forward_mlp
from .torch_models import SpeechKwsCnn, TutorialCnn28, TutorialMlp784


def netkit_softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax — matches netkit Ops::Softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def pack_dense(linear: nn.Linear) -> tuple[np.ndarray, np.ndarray]:
    """PyTorch Linear [out, in] -> netkit W [out, in] row-major + bias [out]."""
    w = linear.weight.detach().cpu().numpy().astype(np.float32)
    b = linear.bias.detach().cpu().numpy().astype(np.float32)
    return w, b


def pack_conv2d(conv: nn.Conv2d) -> tuple[np.ndarray, np.ndarray]:
    """PyTorch Conv2d OIHW -> netkit [out, kh, kw, in] + bias [out]."""
    w = conv.weight.detach().cpu().numpy()
    w_nk = np.transpose(w, (0, 2, 3, 1)).astype(np.float32)
    if conv.bias is not None:
        b = conv.bias.detach().cpu().numpy().astype(np.float32)
    else:
        b = np.zeros(conv.out_channels, dtype=np.float32)
    return w_nk, b


def pack_depthwise_conv2d(conv: nn.Conv2d) -> tuple[np.ndarray, np.ndarray]:
    """PyTorch depthwise Conv2d [C,1,Kh,Kw] -> netkit [C,Kh,Kw] + bias [C]."""
    w = conv.weight.detach().cpu().numpy()
    if w.shape[1] != 1:
        raise ValueError("pack_depthwise_conv2d expects groups == in_channels")
    w_nk = w.reshape(w.shape[0], w.shape[2], w.shape[3]).astype(np.float32)
    b = conv.bias.detach().cpu().numpy().astype(np.float32)
    return w_nk, b


def pack_tutorial_mlp(model: TutorialMlp784) -> np.ndarray:
    parts: list[np.ndarray] = []
    for layer in (model.fc1, model.fc2):
        w, b = pack_dense(layer)
        parts.extend([w.reshape(-1), b])
    return np.concatenate(parts).astype(np.float32)


def pack_tutorial_cnn(model: TutorialCnn28) -> np.ndarray:
    parts: list[np.ndarray] = []
    for layer in (model.conv1, model.conv2):
        w, b = pack_conv2d(layer)
        parts.extend([w.reshape(-1), b])
    for layer in (model.fc1, model.fc2):
        w, b = pack_dense(layer)
        parts.extend([w.reshape(-1), b])
    return np.concatenate(parts).astype(np.float32)


def pack_speech_kws(model: SpeechKwsCnn) -> np.ndarray:
    parts: list[np.ndarray] = []
    for layer in (model.stem,):
        w, b = pack_conv2d(layer)
        parts.extend([w.reshape(-1), b])
    w, b = pack_depthwise_conv2d(model.dw1)
    parts.extend([w.reshape(-1), b])
    w, b = pack_conv2d(model.pw1)
    parts.extend([w.reshape(-1), b])
    w, b = pack_depthwise_conv2d(model.dw2)
    parts.extend([w.reshape(-1), b])
    w, b = pack_conv2d(model.pw2)
    parts.extend([w.reshape(-1), b])
    w, b = pack_dense(model.fc)
    parts.extend([w.reshape(-1), b])
    return np.concatenate(parts).astype(np.float32)


@torch.no_grad()
def forward_mlp_netkit(model: TutorialMlp784, x_flat: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.asarray(x_flat, dtype=np.float32))
    if x.ndim == 1:
        x = x.unsqueeze(0)
    logits = model.forward_logits(x)
    return netkit_softmax(logits.cpu().numpy())


@torch.no_grad()
def forward_cnn_netkit(model: TutorialCnn28, x_flat: np.ndarray, *, img_h: int = 28, img_w: int = 28) -> np.ndarray:
    x = np.asarray(x_flat, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nchw = x.reshape(-1, img_h, img_w, 1).transpose(0, 3, 1, 2)
    tensor = torch.from_numpy(nchw.copy())
    logits = model.forward_logits(tensor)
    return netkit_softmax(logits.cpu().numpy())


@torch.no_grad()
def forward_speech_kws_netkit(model: SpeechKwsCnn, x_flat: np.ndarray, *, img_h: int = 16, img_w: int = 10) -> np.ndarray:
    x = np.asarray(x_flat, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nchw = x.reshape(-1, img_h, img_w, 1).transpose(0, 3, 1, 2)
    tensor = torch.from_numpy(nchw.copy())
    logits = model.forward_logits(tensor)
    return logits.cpu().numpy()


def assert_packed_matches_reference(
    arch: dict,
    weights: np.ndarray,
    torch_forward: Callable[[np.ndarray], np.ndarray],
    *,
    samples: int = 8,
    seed: int = 0,
    atol: float = 1e-5,
) -> None:
    """Verify packed weights produce the same outputs as the PyTorch netkit forward."""
    rng = np.random.default_rng(seed)
    network = arch["network"]

    for _ in range(samples):
        if network == "mlp":
            features = arch["input"][1]
            inp = rng.uniform(-0.5, 0.5, features).astype(np.float32)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                ref = np.asarray(forward_mlp(inp, arch, weights), dtype=np.float32)
        elif network == "cnn":
            h, w, c = arch["input"]
            inp = rng.uniform(-0.5, 0.5, h * w * c).astype(np.float32)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                ref = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
        else:
            raise ValueError(f"unsupported network: {network}")

        out = torch_forward(inp)
        if out.ndim == 2:
            out = out.reshape(-1)
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=atol)
