#!/usr/bin/env python3
"""Write float32 .bin weight files for mlp_hand and cnn_hand models.

Run from repo root:
    python3 tools/write_hand_models.py

See docs/MODEL_FORMAT.md for weight layout.
"""

from __future__ import annotations

import struct
from pathlib import Path


def write_floats(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def mlp_hand_weights() -> list[float]:
    # Layer 1: W[3x4] row-major, then b[4]
    w1 = [
        [1.0, -1.0, 1.0, 0.0],
        [1.0, 1.0, -1.0, 1.0],
        [0.0, 1.0, 1.0, -1.0],
    ]
    b1 = [0.0, 0.0, 0.0, 0.0]

    # Layer 2: W[4x2] row-major, then b[2]
    w2 = [
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ]
    b2 = [1.0, -1.0]

    weights: list[float] = []
    for row in w1:
        weights.extend(row)
    weights.extend(b1)
    for row in w2:
        weights.extend(row)
    weights.extend(b2)
    return weights


def cnn_hand_weights() -> list[float]:
    weights: list[float] = []

    # Layer 1, filter 0: all ones (2x2x2)
    weights.extend([1.0] * 8)
    # Layer 1, filter 1: ic=0 -> 1, ic=1 -> 2 at each kernel position
    for _ in range(4):
        weights.extend([1.0, 2.0])
    weights.extend([0.0, -1.0])  # biases

    # Layer 2, 1x1x2 -> 1 filter
    weights.extend([1.0, 2.0])
    weights.append(0.0)  # bias

    return weights


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "models"
    write_floats(root / "mlp_hand.bin", mlp_hand_weights())
    write_floats(root / "cnn_hand.bin", cnn_hand_weights())
    print("Wrote mlp_hand.bin and cnn_hand.bin")


if __name__ == "__main__":
    main()
