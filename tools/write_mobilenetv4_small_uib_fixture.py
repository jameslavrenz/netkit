#!/usr/bin/env python3
"""Write a hand-checked MobileNetV4 UIB fixture (MNv4-Conv-Small IB block)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from netkit.arch_writer import _make_divisible, write_nk_from_arch
from netkit.reference_forward import forward_cnn
from netkit.writer import RegressionCase, RegressionSuite

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

IMG_H = 4
IMG_W = 4
IN_CHANNELS = 4
OUT_CHANNELS = 4
START_DW = 0
MIDDLE_DW = 3
STRIDE = 1
EXPAND_RATIO = 2.0


def build_arch() -> dict:
    return {
        "network": "cnn",
        "input": [IMG_H, IMG_W, IN_CHANNELS],
        "layers": [
            {
                "type": "mobilenetv4_uib",
                "in_channels": IN_CHANNELS,
                "out_channels": OUT_CHANNELS,
                "start_dw_kernel": START_DW,
                "middle_dw_kernel": MIDDLE_DW,
                "stride": STRIDE,
                "expand_ratio": EXPAND_RATIO,
                "middle_dw_downsample": 1,
            }
        ],
    }


def pack_uib_weights(rng: np.random.Generator) -> np.ndarray:
    in_c = IN_CHANNELS
    out_c = OUT_CHANNELS
    expand_c = _make_divisible(in_c * EXPAND_RATIO, 8)
    parts: list[np.ndarray] = []

    if START_DW:
        parts.append(rng.standard_normal((in_c, START_DW, START_DW), dtype=np.float32).reshape(-1) * 0.05)
        parts.append(rng.standard_normal(in_c, dtype=np.float32) * 0.01)
        parts.append(np.ones(in_c, dtype=np.float32))
        parts.append(np.zeros(in_c, dtype=np.float32))

    parts.append(rng.standard_normal((expand_c, in_c), dtype=np.float32).reshape(-1) * 0.05)
    parts.append(rng.standard_normal(expand_c, dtype=np.float32) * 0.01)
    parts.append(np.ones(expand_c, dtype=np.float32))
    parts.append(np.zeros(expand_c, dtype=np.float32))

    if MIDDLE_DW:
        parts.append(
            rng.standard_normal((expand_c, MIDDLE_DW, MIDDLE_DW), dtype=np.float32).reshape(-1) * 0.05
        )
        parts.append(rng.standard_normal(expand_c, dtype=np.float32) * 0.01)
        parts.append(np.ones(expand_c, dtype=np.float32))
        parts.append(np.zeros(expand_c, dtype=np.float32))

    parts.append(rng.standard_normal((out_c, expand_c), dtype=np.float32).reshape(-1) * 0.05)
    parts.append(rng.standard_normal(out_c, dtype=np.float32) * 0.01)
    parts.append(np.ones(out_c, dtype=np.float32))
    parts.append(np.zeros(out_c, dtype=np.float32))

    return np.concatenate(parts).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(42)
    arch = build_arch()
    weights = pack_uib_weights(rng)
    inp = rng.standard_normal(IMG_H * IMG_W * IN_CHANNELS, dtype=np.float32) * 0.2
    expected = forward_cnn(inp, arch, weights)

    suite = RegressionSuite(
        tolerance=1e-5,
        cases=[
            RegressionCase(
                name="MobileNetV4 Small UIB (hand)",
                input=inp.tolist(),
                expected=expected,
            )
        ],
    )

    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / "mobilenetv4_small_uib.nk"
    write_nk_from_arch(arch, weights, out, tests=suite)
    print(f"Wrote {out} ({len(weights)} weights, 1 TCAS case)")


if __name__ == "__main__":
    main()
