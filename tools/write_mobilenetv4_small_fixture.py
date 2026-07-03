#!/usr/bin/env python3
"""Write full MobileNetV4-Conv-Small .nk fixture (all backbone + classification head layers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from netkit.arch_writer import pack_random_cnn_weights, write_nk_from_arch
from netkit.mobilenetv4_small import build_mobilenetv4_small_arch
from netkit.reference_forward import forward_cnn
from netkit.writer import RegressionCase, RegressionSuite

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# 56×56 fits MNv4-Conv-Small topology and uses the MNIST-scale regression arena (≥784 inputs).
IMG_H = 56
IMG_W = 56
NUM_CLASSES = 10


def main() -> None:
    rng = np.random.default_rng(42)
    arch = build_mobilenetv4_small_arch(
        height=IMG_H, width=IMG_W, channels=3, num_classes=NUM_CLASSES, include_head=True
    )
    weights = pack_random_cnn_weights(arch, rng)
    inp = rng.standard_normal(IMG_H * IMG_W * 3, dtype=np.float32) * 0.2
    expected = forward_cnn(inp, arch, weights)

    suite = RegressionSuite(
        tolerance=1e-5,
        cases=[
            RegressionCase(
                name="MobileNetV4-Conv-Small forward",
                input=inp.tolist(),
                expected=expected,
            )
        ],
    )

    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / "mobilenetv4_small.nk"
    write_nk_from_arch(arch, weights, out, tests=suite)
    print(
        f"Wrote {out} ({len(arch['layers'])} layers, {len(weights)} weights, 1 TCAS case)"
    )


if __name__ == "__main__":
    main()
