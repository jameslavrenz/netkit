#!/usr/bin/env python3
"""Write synthetic op-coverage and deep-chain .nk regression models.

Run from repo root:
    python3 tools/write_op_matrix_models.py
    make export-op-matrix
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit import RegressionCase, RegressionSuite, write_nk_from_arch
from netkit.reference_forward import forward_cnn, forward_mlp, pack_cnn_weights, pack_mlp_weights

MODELS = ROOT / "models"


def _case(name: str, inp: list[float], arch: dict, weights: np.ndarray, *, tol: float = 1e-5) -> RegressionCase:
    if arch["network"] == "mlp":
        expected = forward_mlp(np.asarray(inp, dtype=np.float32), arch, weights)
    else:
        expected = forward_cnn(np.asarray(inp, dtype=np.float32), arch, weights)
    return RegressionCase(name=name, input=inp, expected=expected)


def build_op_matrix_mlp() -> tuple[dict, np.ndarray, RegressionSuite]:
    arch = {
        "network": "mlp",
        "input": [1, 5],
        "layers": [
            {"type": "dense", "units": 4, "activation": "relu"},
            {"type": "dense", "units": 4, "activation": "sigmoid"},
            {"type": "dense", "units": 4, "activation": "tanh"},
            {"type": "dense", "units": 4, "activation": "leaky_relu", "alpha": 0.25},
            {"type": "dense", "units": 4, "activation": "relu6"},
            {"type": "dense", "units": 3, "activation": "none"},
        ],
    }
    rng = np.random.default_rng(11)
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    in_f = 5
    for layer in arch["layers"]:
        out_f = layer["units"]
        w = rng.uniform(-0.5, 0.5, size=(out_f, in_f)).astype(np.float32)
        b = rng.uniform(-0.25, 0.25, size=(out_f,)).astype(np.float32)
        layers.append((w, b))
        in_f = out_f
    weights = pack_mlp_weights(layers)

    inputs = [
        ("mixed signs", [-2.0, -0.5, 0.0, 1.0, 3.0]),
        ("all positive", [0.25, 0.5, 1.0, 1.5, 2.0]),
        ("relu6 plateau", [4.0, 5.0, 6.0, 7.0, 8.0]),
        ("negative heavy", [-3.0, -2.0, -1.0, -0.5, -0.1]),
        ("zero input", [0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    cases = [_case(name, list(inp), arch, weights) for name, inp in inputs]
    return arch, weights, RegressionSuite(tolerance=1e-5, cases=cases)


def build_op_matrix_cnn() -> tuple[dict, np.ndarray, RegressionSuite]:
    arch = {
        "network": "cnn",
        "input": [5, 5, 1],
        "layers": [
            {"type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 2, "activation": "relu"},
            {"type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 2, "activation": "sigmoid"},
            {"type": "conv2d", "kernel_size": 1, "stride": 1, "filters": 3, "activation": "tanh"},
            {"type": "conv2d", "kernel_size": 1, "stride": 1, "filters": 3, "activation": "leaky_relu", "alpha": 0.2},
            {"type": "conv2d", "kernel_size": 1, "stride": 1, "filters": 2, "activation": "relu6"},
            {"type": "flatten"},
            {"type": "dense", "units": 2, "activation": "none"},
        ],
    }
    rng = np.random.default_rng(23)
    tensors: list[tuple[np.ndarray, np.ndarray] | None] = []
    height, width, channels = 5, 5, 1
    dense_in = 0
    for layer in arch["layers"]:
        if layer["type"] == "conv2d":
            k = layer["kernel_size"]
            stride = layer.get("stride", 1)
            out_c = layer["filters"]
            kernel = rng.uniform(-0.4, 0.4, size=(out_c, k, k, channels)).astype(np.float32)
            bias = rng.uniform(-0.2, 0.2, size=(out_c,)).astype(np.float32)
            tensors.append((kernel, bias))
            height = (height - k) // stride + 1
            width = (width - k) // stride + 1
            channels = out_c
        elif layer["type"] == "flatten":
            tensors.append(None)
            dense_in = height * width * channels
        elif layer["type"] == "dense":
            out_f = layer["units"]
            w_arr = rng.uniform(-0.5, 0.5, size=(out_f, dense_in)).astype(np.float32)
            b_arr = rng.uniform(-0.2, 0.2, size=(out_f,)).astype(np.float32)
            tensors.append((w_arr, b_arr))
            dense_in = out_f

    weights = pack_cnn_weights(tensors)

    base = np.linspace(0.1, 1.0, 25, dtype=np.float32).reshape(5, 5, 1)
    inputs = [
        ("graded ramp", base.reshape(-1).tolist()),
        ("checkerboard", np.where((np.add.outer(np.arange(5), np.arange(5)) % 2) == 0, 1.0, -1.0).reshape(-1).tolist()),
        ("center impulse", [0.0] * 12 + [5.0] + [0.0] * 12),
        ("negative corners", [(-1.0 if (r in (0, 4) and c in (0, 4)) else 0.5) for r in range(5) for c in range(5)]),
        ("uniform", [0.75] * 25),
    ]
    cases = [_case(name, list(inp), arch, weights) for name, inp in inputs]
    return arch, weights, RegressionSuite(tolerance=1e-5, cases=cases)


def build_deep_mlp() -> tuple[dict, np.ndarray, RegressionSuite]:
    arch = {
        "network": "mlp",
        "input": [1, 6],
        "layers": [
            {"type": "dense", "units": 6, "activation": "relu"},
            {"type": "dense", "units": 6, "activation": "relu"},
            {"type": "dense", "units": 6, "activation": "relu"},
            {"type": "dense", "units": 6, "activation": "relu"},
            {"type": "dense", "units": 6, "activation": "relu"},
            {"type": "dense", "units": 4, "activation": "relu"},
            {"type": "dense", "units": 2, "activation": "none"},
        ],
    }
    rng = np.random.default_rng(37)
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    in_f = 6
    for layer in arch["layers"]:
        out_f = layer["units"]
        w = rng.normal(0.0, 0.15, size=(out_f, in_f)).astype(np.float32)
        b = rng.normal(0.0, 0.05, size=(out_f,)).astype(np.float32)
        layers.append((w, b))
        in_f = out_f
    weights = pack_mlp_weights(layers)

    inputs = [
        ("unit ramp", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("alternating", [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        ("sparse", [0.0, 0.0, 3.0, 0.0, -2.0, 0.0]),
    ]
    cases = [_case(name, list(inp), arch, weights) for name, inp in inputs]
    return arch, weights, RegressionSuite(tolerance=1e-5, cases=cases)


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    specs = [
        ("op_matrix_mlp.nk", *build_op_matrix_mlp()),
        ("op_matrix_cnn.nk", *build_op_matrix_cnn()),
        ("deep_mlp.nk", *build_deep_mlp()),
    ]
    for filename, arch, weights, suite in specs:
        path = write_nk_from_arch(arch, weights, MODELS / filename, suite)
        print(f"Wrote {path} ({len(suite.cases)} cases)")


if __name__ == "__main__":
    main()
