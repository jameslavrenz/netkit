#!/usr/bin/env python3
"""Write yolox_head_only.nk fixture (isolated YOLOX decoupled head on 2×2×960 features)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np

from netkit.arch_writer import pack_random_cnn_weights, write_nk_from_arch
from netkit.reference_forward import forward_cnn
from netkit.writer import RegressionCase, RegressionSuite
from netkit.yolox_detector import MNv4_SMALL_BACKBONE_OUT_CHANNELS

HEIGHT = 2
WIDTH = 2
IN_CHANNELS = MNv4_SMALL_BACKBONE_OUT_CHANNELS
NUM_CLASSES = 5
HIDDEN_DIM = 32
NUM_CONVS = 2


def main() -> None:
    arch = {
        "network": "cnn",
        "input": [HEIGHT, WIDTH, IN_CHANNELS],
        "layers": [
            {
                "type": "yolox_decoupled_head",
                "in_channels": IN_CHANNELS,
                "hidden_dim": HIDDEN_DIM,
                "num_classes": NUM_CLASSES,
                "num_convs": NUM_CONVS,
            }
        ],
    }
    rng = np.random.default_rng(77)
    weights = pack_random_cnn_weights(arch, rng, scale=0.02)
    inp = rng.standard_normal(HEIGHT * WIDTH * IN_CHANNELS, dtype=np.float32) * 0.05
    expected = forward_cnn(inp, arch, weights)

    out = ROOT / "models" / "yolox_head_only.nk"
    write_nk_from_arch(
        arch,
        weights,
        out,
        RegressionSuite(
            tolerance=1e-4,
            cases=[
                RegressionCase(
                    name="YOLOX head only on 2x2x960",
                    input=inp,
                    expected=expected,
                )
            ],
        ),
    )
    print(
        f"Wrote {out} (1 layer, {weights.nbytes} bytes, "
        f"output={len(expected)} floats = {HEIGHT}x{WIDTH}x{yolox_out_channels()})"
    )


def yolox_out_channels() -> int:
    return 4 + 1 + NUM_CLASSES


if __name__ == "__main__":
    main()
