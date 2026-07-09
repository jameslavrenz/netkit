"""Pack PyTorch backbone checkpoints into netkit flat CNN weights."""

from __future__ import annotations

from typing import Any

import numpy as np

from .batch_norm_fold import fold_batch_norm_params, fold_conv_batch_norm
from .convnextv2_atto import build_convnextv2_atto_arch
from .mobilenetv4_small import MOBILENETV4_CONV_SMALL_BLOCKS, build_mobilenetv4_small_arch
from .resnet18 import RESNET18_BLOCKS, build_resnet18_arch
from .torch_pack import pack_conv2d, pack_dense, pack_depthwise_conv2d

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
    """Pack one timm/torch ResNet BasicBlock into netkit resnet_basic_block tensor order."""
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
    """Pack timm resnet18 weights for build_resnet18_arch()."""
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


def pack_resnet18_from_timm(
    model: nn.Module,
    *,
    height: int,
    width: int,
    num_classes: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return (arch, weights) for a timm resnet18 module."""
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


# Backward-compatible alias.
pack_resnet18_from_torch = pack_resnet18_from_timm


def resnet18_block_specs() -> list[tuple[int, int, int]]:
    """Expose block topology for ONNX fusion parity checks."""
    return list(RESNET18_BLOCKS)


def pack_layernorm2d_tensors(module: nn.Module) -> list[np.ndarray]:
    """Pack LayerNorm2d gamma/beta (netkit layernorm2d layout)."""
    _require_torch()
    weight = module.weight.detach().cpu().numpy().astype(np.float32)
    bias = module.bias.detach().cpu().numpy().astype(np.float32)
    return [weight, bias]


def pack_conv_bn_folded_tensors(conv: nn.Module, bn: nn.Module) -> list[np.ndarray]:
    """Fold BatchNorm into conv and return flat [W, B] for netkit conv2d+ReLU layers."""
    _require_torch()
    w, b = pack_conv2d(conv)
    gamma, beta, mean, var = _bn_arrays(bn)
    w, b = fold_conv_batch_norm(w, b, gamma, beta, mean, var, eps=bn.eps)
    return [w.reshape(-1), b]


def pack_convnextv2_block_tensors(block: nn.Module) -> list[np.ndarray]:
    """Pack one timm ConvNeXtBlock into netkit convnextv2_block tensor order."""
    _require_torch()
    parts: list[np.ndarray] = []

    dw_w, dw_b = pack_depthwise_conv2d(block.conv_dw)
    parts.extend([dw_w.reshape(-1), dw_b])
    parts.extend(pack_layernorm2d_tensors(block.norm))

    pw1_w, pw1_b = pack_conv2d(block.mlp.fc1)
    parts.extend([pw1_w.reshape(-1), pw1_b])

    grn = block.mlp.grn
    parts.append(grn.weight.detach().cpu().numpy().astype(np.float32))
    parts.append(grn.bias.detach().cpu().numpy().astype(np.float32))

    pw2_w, pw2_b = pack_conv2d(block.mlp.fc2)
    parts.extend([pw2_w.reshape(-1), pw2_b])
    return parts


def pack_mobilenetv4_uib_tensors(uib: nn.Module) -> list[np.ndarray]:
    """Pack one timm UniversalInvertedResidual into netkit mobilenetv4_uib tensor order."""
    _require_torch()
    parts: list[np.ndarray] = []

    def append_conv_norm_act(cna: nn.Module, *, depthwise: bool) -> None:
        conv = cna.conv
        bn = cna.bn
        if depthwise:
            w, b = pack_depthwise_conv2d(conv)
        else:
            w, b = pack_conv2d(conv)
        gamma, beta, mean, var = _bn_arrays(bn)
        scale, bn_bias = fold_batch_norm_params(gamma, beta, mean, var, eps=bn.eps)
        parts.extend([w.reshape(-1), b, scale, bn_bias])

    if hasattr(uib.dw_start, "conv"):
        append_conv_norm_act(uib.dw_start, depthwise=True)
    append_conv_norm_act(uib.pw_exp, depthwise=False)
    if hasattr(uib.dw_mid, "conv"):
        append_conv_norm_act(uib.dw_mid, depthwise=True)
    append_conv_norm_act(uib.pw_proj, depthwise=False)
    return parts


def _flatten_mobilenetv4_small_modules(model: nn.Module) -> list[nn.Module | tuple[nn.Module, nn.Module]]:
    """Walk timm mobilenetv4_conv_small in netkit layer order."""
    _require_torch()
    flat: list[nn.Module | tuple[nn.Module, nn.Module]] = [(model.conv_stem, model.bn1)]
    for seq in model.blocks:
        for mod in seq:
            flat.append(mod)
    if len(flat) != len(MOBILENETV4_CONV_SMALL_BLOCKS):
        raise ValueError(
            f"expected {len(MOBILENETV4_CONV_SMALL_BLOCKS)} backbone modules, got {len(flat)}"
        )
    return flat


def pack_convnextv2_atto_tensors(model: nn.Module) -> np.ndarray:
    """Pack timm convnextv2_atto weights for build_convnextv2_atto_arch()."""
    _require_torch()
    parts: list[np.ndarray] = []

    stem_conv = model.stem[0]
    stem_norm = model.stem[1]
    w, b = pack_conv2d(stem_conv)
    parts.extend([w.reshape(-1), b])
    parts.extend(pack_layernorm2d_tensors(stem_norm))

    for stage_i, stage in enumerate(model.stages):
        if stage_i > 0:
            down = stage.downsample
            parts.extend(pack_layernorm2d_tensors(down[0]))
            w, b = pack_conv2d(down[1])
            parts.extend([w.reshape(-1), b])
        for block in stage.blocks:
            parts.extend(pack_convnextv2_block_tensors(block))

    parts.extend(pack_layernorm2d_tensors(model.head.norm))
    w_fc, b_fc = pack_dense(model.head.fc)
    parts.extend([w_fc.reshape(-1), b_fc])
    return np.concatenate(parts).astype(np.float32)


def pack_convnextv2_atto_from_timm(
    model: nn.Module,
    *,
    height: int,
    width: int,
    num_classes: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return (arch, weights) for a timm convnextv2_atto module."""
    _require_torch()
    classes = num_classes if num_classes is not None else int(model.head.fc.out_features)
    arch = build_convnextv2_atto_arch(
        height=height,
        width=width,
        channels=3,
        num_classes=classes,
        include_head=True,
    )
    weights = pack_convnextv2_atto_tensors(model)
    return arch, weights


def pack_mobilenetv4_small_tensors(model: nn.Module) -> np.ndarray:
    """Pack timm mobilenetv4_conv_small weights for build_mobilenetv4_small_arch()."""
    _require_torch()
    parts: list[np.ndarray] = []

    for spec, mod in zip(MOBILENETV4_CONV_SMALL_BLOCKS, _flatten_mobilenetv4_small_modules(model)):
        if spec[0] == "conv_bn":
            if isinstance(mod, tuple):
                conv, bn = mod
            else:
                conv, bn = mod.conv, mod.bn1
            parts.extend(pack_conv_bn_folded_tensors(conv, bn))
        else:
            parts.extend(pack_mobilenetv4_uib_tensors(mod))

    w_head, b_head = pack_conv_bn_folded_tensors(model.conv_head, model.norm_head)
    parts.extend([w_head, b_head])
    w_fc, b_fc = pack_dense(model.classifier)
    parts.extend([w_fc.reshape(-1), b_fc])
    return np.concatenate(parts).astype(np.float32)


def pack_mobilenetv4_small_from_timm(
    model: nn.Module,
    *,
    height: int,
    width: int,
    num_classes: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return (arch, weights) for a timm mobilenetv4_conv_small module."""
    _require_torch()
    classes = num_classes if num_classes is not None else int(model.classifier.out_features)
    arch = build_mobilenetv4_small_arch(
        height=height,
        width=width,
        channels=3,
        num_classes=classes,
        include_head=True,
    )
    weights = pack_mobilenetv4_small_tensors(model)
    return arch, weights


TIMM_BACKBONE_NAMES: dict[str, str] = {
    "resnet18": "resnet18",
    "convnextv2_atto": "convnextv2_atto",
    "mobilenetv4_small": "mobilenetv4_conv_small",
}

PACK_ARCH_DEFAULTS: dict[str, tuple[int, int]] = {
    "resnet18": (56, 56),
    "convnextv2_atto": (32, 32),
    "mobilenetv4_small": (56, 56),
}


def load_backbone_model(
    arch_name: str,
    *,
    num_classes: int,
    pretrained: bool = False,
) -> nn.Module:
    """Instantiate a timm backbone for packing (random or ImageNet-pretrained)."""
    _require_torch()
    try:
        import timm
    except ImportError as exc:
        raise SystemExit('pack requires timm: pip install -e "python[train]"') from exc

    timm_name = TIMM_BACKBONE_NAMES.get(arch_name)
    if timm_name is None:
        raise ValueError(f"unsupported arch: {arch_name}")
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    model.eval()
    return model


def pack_backbone_from_torch(
    arch_name: str,
    model: nn.Module,
    *,
    height: int,
    width: int,
    num_classes: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Pack a loaded backbone module into (arch, weights)."""
    if arch_name == "resnet18":
        return pack_resnet18_from_timm(model, height=height, width=width, num_classes=num_classes)
    if arch_name == "convnextv2_atto":
        return pack_convnextv2_atto_from_timm(model, height=height, width=width, num_classes=num_classes)
    if arch_name == "mobilenetv4_small":
        return pack_mobilenetv4_small_from_timm(model, height=height, width=width, num_classes=num_classes)
    raise ValueError(f"unsupported arch: {arch_name}")


def backbone_torch_forward(
    model: nn.Module,
    inp: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Run timm backbone forward on NHWC-flat input."""
    _require_torch()
    import torch

    x = torch.from_numpy(inp.reshape(1, height, width, 3).transpose(0, 3, 1, 2).copy())
    with torch.no_grad():
        logits = model(x)
    return logits.cpu().numpy().reshape(-1)
