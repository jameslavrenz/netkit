"""Pack PyTorch backbone checkpoints into netkit flat CNN weights."""

from __future__ import annotations

from typing import Any

import numpy as np

from .batch_norm_fold import fold_batch_norm_params, fold_conv_batch_norm
from .resnet18 import RESNET18_BLOCKS, build_resnet18_arch
from .torch_pack import pack_conv2d, pack_dense

try:
    import torch.nn as nn
except ImportError:  # pragma: no cover
    nn = None  # type: ignore[assignment,misc]


def _require_torch() -> None:
    if nn is None:
        raise SystemExit('Requires torch: pip install -e "python[train]"')


def _bn_arrays(module: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        module.weight.detach().cpu().numpy(),
        module.bias.detach().cpu().numpy(),
        module.running_mean.detach().cpu().numpy(),
        module.running_var.detach().cpu().numpy(),
    )


def pack_resnet_basic_block_tensors(block: nn.Module) -> list[np.ndarray]:
    """Pack one torchvision BasicBlock into netkit resnet_basic_block tensor order."""
    _require_torch()
    parts: list[np.ndarray] = []

    w1, b1 = pack_conv2d(block.conv1)
    g1, b1_beta, m1, v1 = _bn_arrays(block.bn1)
    s1, bn1_bias = fold_batch_norm_params(g1, b1_beta, m1, v1, eps=block.bn1.eps)
    parts.extend([w1.reshape(-1), b1, s1, bn1_bias])

    w2, b2 = pack_conv2d(block.conv2)
    g2, b2_beta, m2, v2 = _bn_arrays(block.bn2)
    s2, bn2_bias = fold_batch_norm_params(g2, b2_beta, m2, v2, eps=block.bn2.eps)
    parts.extend([w2.reshape(-1), b2, s2, bn2_bias])

    downsample = getattr(block, "downsample", None)
    if downsample is not None:
        ds_conv = downsample[0]
        ds_bn = downsample[1]
        ws, bs = pack_conv2d(ds_conv)
        gs, bs_beta, ms, vs = _bn_arrays(ds_bn)
        ss, shortcut_bias = fold_batch_norm_params(gs, bs_beta, ms, vs, eps=ds_bn.eps)
        parts.extend([ws.reshape(-1), bs, ss, shortcut_bias])

    return parts


def pack_resnet18_tensors(model: nn.Module) -> np.ndarray:
    """Pack torchvision ResNet-18 weights for build_resnet18_arch()."""
    _require_torch()
    parts: list[np.ndarray] = []

    w_stem, b_stem = pack_conv2d(model.conv1)
    g, beta, mean, var = _bn_arrays(model.bn1)
    w_stem, b_stem = fold_conv_batch_norm(w_stem, b_stem, g, beta, mean, var, eps=model.bn1.eps)
    parts.extend([w_stem.reshape(-1), b_stem])

    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in layer:
            parts.extend(pack_resnet_basic_block_tensors(block))

    w_fc, b_fc = pack_dense(model.fc)
    parts.extend([w_fc.reshape(-1), b_fc])
    return np.concatenate(parts).astype(np.float32)


def pack_resnet18_from_torch(
    model: nn.Module,
    *,
    height: int,
    width: int,
    num_classes: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return (arch, weights) for a torchvision ResNet-18 module."""
    _require_torch()
    classes = num_classes if num_classes is not None else int(model.fc.out_features)
    arch = build_resnet18_arch(
        height=height,
        width=width,
        channels=3,
        num_classes=classes,
        include_head=True,
    )
    weights = pack_resnet18_tensors(model)
    return arch, weights


def resnet18_block_specs() -> list[tuple[int, int, int]]:
    """Expose block topology for ONNX fusion parity checks."""
    return list(RESNET18_BLOCKS)
