#!/usr/bin/env python3
"""Write full ResNet-18 .nk fixture (stem + 8 BasicBlocks + classification head)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from netkit.arch_writer import pack_random_cnn_weights, write_nk_from_arch
from netkit.reference_forward import forward_cnn
from netkit.resnet18 import build_resnet18_arch
from netkit.writer import RegressionCase, RegressionSuite

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

IMG_H = 56
IMG_W = 56
NUM_CLASSES = 10


def main() -> None:
    rng = np.random.default_rng(42)
    arch = build_resnet18_arch(
        height=IMG_H, width=IMG_W, channels=3, num_classes=NUM_CLASSES, include_head=True
    )
    weights = pack_random_cnn_weights(arch, rng)
    inp = rng.standard_normal(IMG_H * IMG_W * 3, dtype=np.float32) * 0.2
    expected = forward_cnn(inp, arch, weights)

    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[
            RegressionCase(
                name="ResNet-18 forward",
                input=inp.tolist(),
                expected=expected,
            )
        ],
    )

    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / "resnet18.nk"
    write_nk_from_arch(arch, weights, out, tests=suite)
    print(
        f"Wrote {out} ({len(arch['layers'])} layers, {len(weights)} weights, 1 TCAS case)"
    )


if __name__ == "__main__":
    main()
