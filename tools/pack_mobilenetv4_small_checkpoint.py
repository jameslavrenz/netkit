#!/usr/bin/env python3
"""Pack a timm MobileNetV4-Conv-Small checkpoint into a .nk file."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    import torch  # noqa: F401
except ImportError as exc:
    raise SystemExit('Requires torch/timm: pip install -e "python[train]"') from exc

from netkit.arch_writer import write_nk_from_arch
from netkit.reference_forward import forward_cnn
from netkit.torch_backbone_pack import (
    backbone_torch_forward,
    load_backbone_model,
    pack_backbone_from_torch,
)
from netkit.torch_pack import assert_packed_matches_reference
from netkit.writer import RegressionCase, RegressionSuite


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pack timm MobileNetV4-Conv-Small into .nk")
    parser.add_argument("-o", "--output", required=True, help="Output .nk path")
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load ImageNet-pretrained timm weights",
    )
    args = parser.parse_args()

    model = load_backbone_model(
        "mobilenetv4_small", num_classes=args.num_classes, pretrained=args.pretrained
    )
    arch, weights = pack_backbone_from_torch(
        "mobilenetv4_small",
        model,
        height=args.height,
        width=args.width,
        num_classes=args.num_classes,
    )

    def torch_forward(inp: np.ndarray) -> np.ndarray:
        return backbone_torch_forward(model, inp, height=args.height, width=args.width)

    assert_packed_matches_reference(arch, weights, torch_forward, seed=42, atol=1e-4, samples=3)

    rng = np.random.default_rng(0)
    inp = rng.standard_normal(args.height * args.width * 3, dtype=np.float32) * 0.1
    expected = forward_cnn(inp, arch, weights)

    out = Path(args.output)
    write_nk_from_arch(
        arch,
        weights,
        out,
        RegressionSuite(
            tolerance=1e-4,
            cases=[RegressionCase(name="MobileNetV4 Small packed checkpoint", input=inp, expected=expected)],
        ),
    )
    print(f"Wrote {out} ({len(arch['layers'])} layers, {weights.nbytes} bytes)")


if __name__ == "__main__":
    main()
