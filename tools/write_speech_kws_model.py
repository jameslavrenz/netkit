#!/usr/bin/env python3
"""Write a compact Speech Commands–style KWS .nk model with fixed coefficients.

The classifier expects a 16×10×1 float feature map (downsampled MFCC / spectrogram
patch). Weights are hand-authored — not trained — but the topology matches typical
MCU keyword-spotting CNNs (conv → pool → conv → pool → dense logits).

Regression inputs are precomputed feature grids; expected outputs are computed via
the NumPy reference forward pass and embedded in the .nk TCAS section (no JSON).

Run from repo root:
    python3 tools/write_speech_kws_model.py
    make export-speech-kws
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit import RegressionCase, RegressionSuite, write_nk_from_arch
from netkit.reference_forward import forward_cnn, pack_cnn_weights

MODELS = ROOT / "models"

# 16 time frames × 10 mel bins — compact vs full 49×40 micro_speech, still MCU-sized.
INPUT_H = 16
INPUT_W = 10
INPUT_C = 1
NUM_KEYWORDS = 12  # yes, no, up, down, left, right, on, off, stop, go, unknown, silence


def _fixed_conv1() -> tuple[np.ndarray, np.ndarray]:
    """Four 3×3 filters on a single input channel (spectral / temporal edges)."""
    kernel = np.zeros((4, 3, 3, 1), dtype=np.float32)
    kernel[0, :, :, 0] = [
        [-0.20, -0.20, -0.20],
        [0.00, 0.00, 0.00],
        [0.20, 0.20, 0.20],
    ]
    kernel[1, :, :, 0] = [
        [-0.15, 0.00, 0.15],
        [-0.15, 0.00, 0.15],
        [-0.15, 0.00, 0.15],
    ]
    kernel[2, :, :, 0] = 0.10
    kernel[3, 1, 1, 0] = 0.80
    bias = np.array([0.00, 0.05, -0.05, 0.10], dtype=np.float32)
    return kernel, bias


def _fixed_conv2() -> tuple[np.ndarray, np.ndarray]:
    """Eight 3×3 filters mixing four feature channels."""
    kernel = np.zeros((8, 3, 3, 4), dtype=np.float32)
    taps = np.array(
        [
            [0.12, -0.08, 0.04],
            [0.10, 0.10, 0.10],
            [-0.06, 0.14, -0.06],
        ],
        dtype=np.float32,
    )
    for oc in range(8):
        ic = oc % 4
        sign = 1.0 if oc < 4 else -1.0
        kernel[oc, :, :, ic] = sign * taps
        kernel[oc, 1, 1, (ic + 1) % 4] += 0.20
    bias = np.linspace(-0.08, 0.08, 8, dtype=np.float32)
    return kernel, bias


def _fixed_dense() -> tuple[np.ndarray, np.ndarray]:
    """12 keyword logits from 16 pooled features."""
    weight = np.zeros((NUM_KEYWORDS, 16), dtype=np.float32)
    for row in range(NUM_KEYWORDS):
        weight[row, row % 16] = 0.45
        weight[row, (row * 5 + 3) % 16] = 0.30
        weight[row, (row * 7 + 1) % 16] = 0.15
    bias = np.array(
        [
            0.20,
            -0.15,
            0.10,
            -0.10,
            0.05,
            -0.05,
            0.12,
            -0.12,
            0.08,
            -0.08,
            0.00,
            -0.20,
        ],
        dtype=np.float32,
    )
    return weight, bias


def _speech_arch() -> dict:
    return {
        "network": "cnn",
        "input": [INPUT_H, INPUT_W, INPUT_C],
        "layers": [
            {"type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 4, "activation": "relu"},
            {"type": "max_pool2d", "pool_size": 2, "stride": 2},
            {"type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 8, "activation": "relu"},
            {"type": "max_pool2d", "pool_size": 2, "stride": 2},
            {"type": "flatten"},
            {"type": "dense", "units": NUM_KEYWORDS, "activation": "none"},
        ],
    }


def _pack_weights() -> np.ndarray:
    conv1 = _fixed_conv1()
    conv2 = _fixed_conv2()
    dense = _fixed_dense()
    return pack_cnn_weights([conv1, conv2, None, dense])


def _feature_grid(name: str) -> np.ndarray:
    """Synthetic MFCC-like patches (time × mel). Values in [0, 1] unless noted."""
    grid = np.zeros((INPUT_H, INPUT_W), dtype=np.float32)
    if name == "silence":
        pass
    elif name == "low_band":
        grid[:, 0:3] = 0.85
    elif name == "mid_band":
        grid[:, 3:7] = 0.75
    elif name == "high_band":
        grid[:, 7:10] = 0.90
    elif name == "onset":
        grid[0:4, :] = 0.70
    elif name == "offset":
        grid[12:16, :] = 0.65
    elif name == "yes_like":
        grid[2:10, 1:4] = 0.80
        grid[4:12, 4:6] = 0.35
    elif name == "no_like":
        grid[1:9, 6:9] = 0.75
        grid[6:14, 2:5] = 0.40
    else:
        raise ValueError(f"unknown feature pattern: {name}")
    return grid.reshape(INPUT_H, INPUT_W, INPUT_C)


def _case(name: str, pattern: str, arch: dict, weights: np.ndarray) -> RegressionCase:
    flat = _feature_grid(pattern).reshape(-1)
    expected = forward_cnn(flat, arch, weights)
    return RegressionCase(name=name, input=flat.tolist(), expected=expected)


def build_speech_kws() -> tuple[dict, np.ndarray, RegressionSuite]:
    arch = _speech_arch()
    weights = _pack_weights()
    patterns = [
        ("silence", "silence"),
        ("low mel band", "low_band"),
        ("mid mel band", "mid_band"),
        ("high mel band", "high_band"),
        ("early onset", "onset"),
        ("late offset", "offset"),
        ("yes-like patch", "yes_like"),
        ("no-like patch", "no_like"),
    ]
    cases = [_case(label, pattern, arch, weights) for label, pattern in patterns]
    return arch, weights, RegressionSuite(tolerance=1e-5, cases=cases)


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    arch, weights, suite = build_speech_kws()
    out = MODELS / "speech_kws.nk"
    write_nk_from_arch(arch, weights, out, suite)
    nbytes = out.stat().st_size
    print(
        f"Wrote {out} ({len(suite.cases)} TCAS cases, "
        f"{len(weights)} weights, {nbytes} bytes)"
    )


if __name__ == "__main__":
    main()
