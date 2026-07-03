#!/usr/bin/env python3
"""Write a hand-checked ConvNeXt V2 block fixture (Atto-compatible block op)."""

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
CHANNELS = 2


def build_arch() -> dict:
    return {
        "network": "cnn",
        "input": [IMG_H, IMG_W, CHANNELS],
        "layers": [{"type": "convnextv2_block", "channels": CHANNELS, "eps": 1e-6}],
    }


def pack_block_weights(rng: np.random.Generator) -> np.ndarray:
    ch = CHANNELS
    expanded = ch * 4
    parts = [
        rng.standard_normal((ch, 7, 7), dtype=np.float32).reshape(-1) * 0.05,
        rng.standard_normal(ch, dtype=np.float32) * 0.01,
        np.ones(ch, dtype=np.float32),
        np.zeros(ch, dtype=np.float32),
        rng.standard_normal((expanded, ch), dtype=np.float32).reshape(-1) * 0.05,
        rng.standard_normal(expanded, dtype=np.float32) * 0.01,
        np.zeros(expanded, dtype=np.float32),
        np.zeros(expanded, dtype=np.float32),
        rng.standard_normal((ch, expanded), dtype=np.float32).reshape(-1) * 0.05,
        rng.standard_normal(ch, dtype=np.float32) * 0.01,
    ]
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
                name="ConvNeXt V2 Atto block (hand)",
                input=inp.tolist(),
                expected=expected,
            )
        ],
    )

    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / "convnextv2_atto_block.nk"
    write_nk_from_arch(arch, weights, out, tests=suite)
    print(f"Wrote {out} ({len(weights)} weights, 1 TCAS case)")


if __name__ == "__main__":
    main()
