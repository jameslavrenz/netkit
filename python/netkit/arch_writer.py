"""Build .nk files from in-memory architecture dicts and flat weight arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .format import activation_from_name
from .writer import LayerSpec, ModelSpec, RegressionCase, RegressionSuite, write_nk


def _out_dim(in_dim: int, kernel: int, stride: int, pad: int = 0) -> int:
    return (in_dim + 2 * pad - kernel) // stride + 1


def _split_mlp_weights(arch: dict, weights: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    offset = 0
    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []
    in_features = arch["input"][1]

    for layer in arch["layers"]:
        out_features = layer["units"]
        w_size = in_features * out_features
        w = weights[offset : offset + w_size].reshape(out_features, in_features)
        offset += w_size
        b = weights[offset : offset + out_features]
        offset += out_features
        weight_tensors.append(w.astype(np.float32))
        bias_tensors.append(b.astype(np.float32))
        in_features = out_features

    return weight_tensors, bias_tensors


def _split_cnn_weights(arch: dict, weights: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    offset = 0
    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []
    height, width, channels = arch["input"]
    dense_in = 0

    for layer in arch["layers"]:
        layer_type = layer["type"]
        if layer_type == "conv2d":
            k = layer["kernel_size"]
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            out_c = layer["filters"]
            kernel_elems = k * k * channels
            w_flat = weights[offset : offset + kernel_elems * out_c]
            offset += kernel_elems * out_c
            b = weights[offset : offset + out_c]
            offset += out_c
            kernel = w_flat.reshape(out_c, k, k, channels)
            weight_tensors.append(kernel.astype(np.float32))
            bias_tensors.append(b.astype(np.float32))
            height = _out_dim(height, k, stride, pad_h)
            width = _out_dim(width, k, stride, pad_w)
            channels = out_c
        elif layer_type == "max_pool2d":
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            height = _out_dim(height, pool, stride, pad_h)
            width = _out_dim(width, pool, stride, pad_w)
        elif layer_type == "avg_pool2d":
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            height = _out_dim(height, pool, stride, pad_h)
            width = _out_dim(width, pool, stride, pad_w)
        elif layer_type == "batch_norm2d":
            ch = layer["channels"]
            scale = weights[offset : offset + ch]
            offset += ch
            bias = weights[offset : offset + ch]
            offset += ch
            weight_tensors.append(scale.astype(np.float32))
            bias_tensors.append(bias.astype(np.float32))
        elif layer_type == "flatten":
            dense_in = height * width * channels
        elif layer_type == "dense":
            out_f = layer["units"]
            w_size = dense_in * out_f
            dense_w = weights[offset : offset + w_size].reshape(out_f, dense_in)
            offset += w_size
            b = weights[offset : offset + out_f]
            offset += out_f
            weight_tensors.append(dense_w.astype(np.float32))
            bias_tensors.append(b.astype(np.float32))
            dense_in = out_f

    if offset != len(weights):
        raise ValueError(f"weight count mismatch: used {offset}, file has {len(weights)}")

    return weight_tensors, bias_tensors


def _arch_to_spec(arch: dict, weights: np.ndarray) -> ModelSpec:
    layers: list[LayerSpec] = []
    for layer in arch["layers"]:
        layer_type = layer["type"]
        act = activation_from_name(layer.get("activation", "none"))
        alpha = float(layer.get("alpha", 0.01))
        if layer_type == "dense":
            layers.append(LayerSpec(kind="dense", units=layer["units"], activation=act, alpha=alpha))
        elif layer_type == "conv2d":
            layers.append(
                LayerSpec(
                    kind="conv2d",
                    kernel_size=layer["kernel_size"],
                    stride=layer.get("stride", 1),
                    filters=layer["filters"],
                    activation=act,
                    alpha=alpha,
                    pad_h=layer.get("pad_h", 0),
                    pad_w=layer.get("pad_w", 0),
                )
            )
        elif layer_type == "max_pool2d":
            layers.append(
                LayerSpec(
                    kind="max_pool2d",
                    pool_size=layer["pool_size"],
                    stride=layer.get("stride", layer["pool_size"]),
                    pad_h=layer.get("pad_h", 0),
                    pad_w=layer.get("pad_w", 0),
                )
            )
        elif layer_type == "avg_pool2d":
            layers.append(
                LayerSpec(
                    kind="avg_pool2d",
                    pool_size=layer["pool_size"],
                    stride=layer.get("stride", layer["pool_size"]),
                    pad_h=layer.get("pad_h", 0),
                    pad_w=layer.get("pad_w", 0),
                )
            )
        elif layer_type == "batch_norm2d":
            layers.append(LayerSpec(kind="batch_norm2d", channels=layer["channels"]))
        elif layer_type == "flatten":
            layers.append(LayerSpec(kind="flatten"))
        else:
            raise ValueError(f"unsupported layer type: {layer_type}")

    if arch["network"] == "mlp":
        weight_tensors, bias_tensors = _split_mlp_weights(arch, weights)
    else:
        weight_tensors, bias_tensors = _split_cnn_weights(arch, weights)

    return ModelSpec(
        network=arch["network"],
        input_shape=list(arch["input"]),
        layers=layers,
        weight_tensors=weight_tensors,
        bias_tensors=bias_tensors,
    )


def write_nk_from_arch(
    arch: dict,
    weights: np.ndarray,
    output_path: str | Path,
    tests: RegressionSuite | None = None,
) -> Path:
    output_path = Path(output_path)
    spec = _arch_to_spec(arch, weights)
    spec.tests = tests
    write_nk(output_path, spec)
    return output_path
