#!/usr/bin/env python3
"""Write a hand-checked ResNet-18 BasicBlock fixture (identity shortcut)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from netkit.arch_writer import write_nk_from_arch
from netkit.reference_forward import forward_cnn
from netkit.writer import RegressionCase, RegressionSuite

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

IMG_H = 4
IMG_W = 4
CHANNELS = 4
STRIDE = 1


def build_arch() -> dict:
    return {
        "network": "cnn",
        "input": [IMG_H, IMG_W, CHANNELS],
        "layers": [
            {
                "type": "resnet_basic_block",
                "in_channels": CHANNELS,
                "out_channels": CHANNELS,
                "stride": STRIDE,
            }
        ],
    }


def pack_block_weights(rng: np.random.Generator) -> np.ndarray:
    in_c = CHANNELS
    out_c = CHANNELS
    parts: list[np.ndarray] = []

    parts.append(rng.standard_normal((out_c, 3, 3, in_c), dtype=np.float32).reshape(-1) * 0.05)
    parts.append(rng.standard_normal(out_c, dtype=np.float32) * 0.01)
    parts.append(np.ones(out_c, dtype=np.float32))
    parts.append(np.zeros(out_c, dtype=np.float32))

    parts.append(rng.standard_normal((out_c, 3, 3, out_c), dtype=np.float32).reshape(-1) * 0.05)
    parts.append(rng.standard_normal(out_c, dtype=np.float32) * 0.01)
    parts.append(np.ones(out_c, dtype=np.float32))
    parts.append(np.zeros(out_c, dtype=np.float32))

    return np.concatenate(parts).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(42)
    arch = build_arch()
    weights = pack_block_weights(rng)
    inp = rng.standard_normal(IMG_H * IMG_W * CHANNELS, dtype=np.float32) * 0.2
    expected = forward_cnn(inp, arch, weights)

    suite = RegressionSuite(
        tolerance=1e-5,
        cases=[
            RegressionCase(
                name="ResNet-18 BasicBlock (hand)",
                input=inp.tolist(),
                expected=expected,
            )
        ],
    )

    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / "resnet18_basic_block.nk"
    write_nk_from_arch(arch, weights, out, tests=suite)
    print(f"Wrote {out} ({len(weights)} weights, 1 TCAS case)")


if __name__ == "__main__":
    main()
