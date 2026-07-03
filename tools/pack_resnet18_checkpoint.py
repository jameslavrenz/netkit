#!/usr/bin/env python3
"""Pack a torchvision ResNet-18 checkpoint into a .nk file."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    import torch
    from torchvision.models import resnet18
except ImportError as exc:
    raise SystemExit('Requires torch/torchvision: pip install -e "python[train]" torchvision') from exc

from netkit.arch_writer import write_nk_from_arch
from netkit.reference_forward import forward_cnn
from netkit.torch_backbone_pack import pack_resnet18_from_torch
from netkit.torch_pack import assert_packed_matches_reference
from netkit.writer import RegressionCase, RegressionSuite


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pack torchvision ResNet-18 into .nk")
    parser.add_argument("-o", "--output", required=True, help="Output .nk path")
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    model = resnet18(weights=None)
    model.eval()
    if args.num_classes != model.fc.out_features:
        model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)

    arch, weights = pack_resnet18_from_torch(
        model,
        height=args.height,
        width=args.width,
        num_classes=args.num_classes,
    )

    img_h, img_w = args.height, args.width

    def torch_forward(inp: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(inp.reshape(1, img_h, img_w, 3).transpose(0, 3, 1, 2).copy())
        with torch.no_grad():
            logits = model(x)
        return logits.cpu().numpy().reshape(-1)

    assert_packed_matches_reference(arch, weights, torch_forward, seed=42, atol=1e-4)

    rng = np.random.default_rng(0)
    inp = rng.standard_normal(img_h * img_w * 3, dtype=np.float32) * 0.1
    expected = forward_cnn(inp, arch, weights)

    out = Path(args.output)
    write_nk_from_arch(
        arch,
        weights,
        out,
        RegressionSuite(
            tolerance=1e-4,
            cases=[RegressionCase(name="ResNet-18 packed checkpoint", input=inp, expected=expected)],
        ),
    )
    print(f"Wrote {out} ({len(arch['layers'])} layers, {weights.nbytes} bytes)")


if __name__ == "__main__":
    main()
