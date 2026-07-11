#!/usr/bin/env python3
"""Pack a MiniDetector torch checkpoint into a PAFPN detector .nk (not the CI fixture).

Default output: models/yolox_mnv4_pafpn_trained.nk (keeps models/yolox_mnv4_small.nk as random TCAS).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch
import torch.nn as nn

from netkit.arch_writer import count_packed_cnn_weight_floats, write_nk_from_arch
from netkit.mobilenetv4_small import MOBILENETV4_CONV_SMALL_BLOCKS
from netkit.reference_forward import forward_cnn
from netkit.torch_backbone_pack import (
    _flatten_mobilenetv4_small_modules,
    pack_conv_bn_folded_tensors,
    pack_mobilenetv4_uib_tensors,
)
from netkit.writer import RegressionCase, RegressionSuite
from netkit.yolox_pafpn import build_yolox_mnv4_small_pafpn_detector
from train_yolox_mnv4_pafpn_mini import MiniDetector


def _conv_oihw_flat(w: torch.Tensor) -> np.ndarray:
    return w.detach().float().cpu().numpy().reshape(-1).astype(np.float32)


def _bias_flat(b: torch.Tensor) -> np.ndarray:
    return b.detach().float().cpu().numpy().reshape(-1).astype(np.float32)


def pack_head(module: nn.Module) -> list[np.ndarray]:
    parts: list[np.ndarray] = []
    parts.append(_conv_oihw_flat(module.stem.weight))
    parts.append(_bias_flat(module.stem.bias))
    for conv in module.cls_convs:
        parts.append(_conv_oihw_flat(conv.weight))
        parts.append(_bias_flat(conv.bias))
    for conv in module.reg_convs:
        parts.append(_conv_oihw_flat(conv.weight))
        parts.append(_bias_flat(conv.bias))
    parts.append(_conv_oihw_flat(module.cls_pred.weight))
    parts.append(_bias_flat(module.cls_pred.bias))
    parts.append(_conv_oihw_flat(module.reg_pred.weight))
    parts.append(_bias_flat(module.reg_pred.bias))
    parts.append(_conv_oihw_flat(module.obj_pred.weight))
    parts.append(_bias_flat(module.obj_pred.bias))
    return parts


def pack_dwpw(module: nn.Module) -> list[np.ndarray]:
    return [
        _conv_oihw_flat(module.dw.weight),
        _bias_flat(module.dw.bias),
        _conv_oihw_flat(module.pw.weight),
        _bias_flat(module.pw.bias),
    ]


def pack_neck(neck: nn.Module) -> list[np.ndarray]:
    parts: list[np.ndarray] = []
    for lat in (neck.lat3, neck.lat4, neck.lat5):
        parts.append(_conv_oihw_flat(lat.weight))
        parts.append(_bias_flat(lat.bias))
    for block in (neck.td_p4, neck.td_p3, neck.bu_n4, neck.bu_n5):
        parts.extend(pack_dwpw(block))
    for head in neck.heads:
        parts.extend(pack_head(head))
    return parts


def pack_backbone_blocks(full_model: nn.Module) -> np.ndarray:
    parts: list[np.ndarray] = []
    modules = _flatten_mobilenetv4_small_modules(full_model)
    for spec, mod in zip(MOBILENETV4_CONV_SMALL_BLOCKS, modules):
        if spec[0] == "conv_bn":
            conv, bn = mod  # type: ignore[misc]
            parts.extend(pack_conv_bn_folded_tensors(conv, bn))
        else:
            parts.extend(pack_mobilenetv4_uib_tensors(mod))  # type: ignore[arg-type]
    return np.concatenate(parts).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "yolox_mnv4_pafpn_coco_val.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / "yolox_mnv4_pafpn_trained.nk",
    )
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--classes", type=int, default=80)
    args = parser.parse_args()

    if not args.ckpt.is_file():
        raise SystemExit(f"missing checkpoint {args.ckpt}")

    import timm

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = MiniDetector(hidden=args.hidden, freeze_backbone=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    full = timm.create_model("mobilenetv4_conv_small", pretrained=False)
    feat_sd = model.backbone.state_dict()
    full_sd = full.state_dict()
    mapped = 0
    for k, v in feat_sd.items():
        if k in full_sd and full_sd[k].shape == v.shape:
            full_sd[k] = v
            mapped += 1
    full.load_state_dict(full_sd, strict=False)
    print(f"mapped backbone tensors into full model: {mapped}")

    backbone_flat = pack_backbone_blocks(full)
    neck_parts = pack_neck(model.neck)
    weights = np.concatenate([backbone_flat, *neck_parts]).astype(np.float32)

    arch = build_yolox_mnv4_small_pafpn_detector(
        height=args.height,
        width=args.width,
        num_classes=args.classes,
        hidden_dim=args.hidden,
    )
    expected_n = count_packed_cnn_weight_floats(arch)
    if weights.size != expected_n:
        raise SystemExit(
            f"weight count mismatch: packed {weights.size} vs arch expects {expected_n}"
        )

    rng = np.random.default_rng(0)
    x = rng.standard_normal(args.height * args.width * 3, dtype=np.float32) * 0.1
    y = forward_cnn(x, arch, weights)
    write_nk_from_arch(
        arch,
        weights,
        args.out,
        RegressionSuite(
            tolerance=1e-3,
            cases=[
                RegressionCase(
                    name="YOLOX MNv4-Small PAFPN packed checkpoint",
                    input=x,
                    expected=y,
                )
            ],
        ),
    )
    print(f"wrote {args.out} weights={weights.size} output_elems={len(y)}")


if __name__ == "__main__":
    main()
