"""Write netkit .nk binary model files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .format import (
    FLAG_HAS_TESTS,
    Activation,
    DType,
    NetworkKind,
    pack_avg_pool_layer,
    pack_batch_norm_layer,
    pack_convnextv2_block_layer,
    pack_layernorm2d_layer,
    pack_mobilenetv4_uib_layer,
    pack_resnet_basic_block_layer,
    pack_conv_layer,
    pack_depthwise_conv_layer,
    pack_dense_layer,
    pack_flatten_layer,
    pack_header,
    pack_pool_layer,
    pack_tensor_desc,
    pack_test_section,
)


@dataclass
class LayerSpec:
    kind: str
    units: int = 0
    activation: Activation = Activation.NONE
    alpha: float = 0.01
    kernel_size: int = 1
    kernel_h: int = 1
    kernel_w: int = 1
    stride: int = 1
    filters: int = 0
    pad_h: int = 0
    pad_w: int = 0
    pool_size: int = 2
    channels: int = 0
    eps: float = 1e-6
    in_channels: int = 0
    out_channels: int = 0
    start_dw_kernel: int = 0
    middle_dw_kernel: int = 0
    middle_dw_downsample: int = 1
    expand_ratio: float = 1.0


@dataclass
class RegressionCase:
    name: str
    input: list[float] | np.ndarray
    expected: list[float] | np.ndarray
    label: int = -1


@dataclass
class RegressionSuite:
    tolerance: float = 1e-5
    cases: list[RegressionCase] = field(default_factory=list)


@dataclass
class ModelSpec:
    network: str
    input_shape: list[int]
    layers: list[LayerSpec] = field(default_factory=list)
    weight_tensors: list[np.ndarray] = field(default_factory=list)
    bias_tensors: list[np.ndarray] = field(default_factory=list)
    tests: RegressionSuite | None = None


def write_nk(path: str | Path, spec: ModelSpec) -> None:
    Path(path).write_bytes(write_nk_bytes(spec))


def write_nk_bytes(spec: ModelSpec) -> bytes:
    network_kind = NetworkKind.MLP if spec.network == "mlp" else NetworkKind.CNN
    input_rank = len(spec.input_shape)

    weights_blob = b"".join(
        np.ascontiguousarray(w, dtype=np.float32).tobytes() for w in spec.weight_tensors
    )
    biases_blob = b"".join(
        np.ascontiguousarray(b, dtype=np.float32).tobytes() for b in spec.bias_tensors
    )

    flags = FLAG_HAS_TESTS if spec.tests and spec.tests.cases else 0
    header = pack_header(
        network_kind=network_kind,
        input_rank=input_rank,
        input_shape=spec.input_shape,
        num_layers=len(spec.layers),
        num_weight_tensors=len(spec.weight_tensors),
        num_bias_tensors=len(spec.bias_tensors),
        weights_bytes=len(weights_blob),
        biases_bytes=len(biases_blob),
        flags=flags,
    )

    layer_bytes = bytearray()
    for layer in spec.layers:
        if layer.kind == "dense":
            layer_bytes += pack_dense_layer(
                units=layer.units, activation=layer.activation, alpha=layer.alpha
            )
        elif layer.kind == "conv2d":
            layer_bytes += pack_conv_layer(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                filters=layer.filters,
                activation=layer.activation,
                alpha=layer.alpha,
                pad_h=layer.pad_h,
                pad_w=layer.pad_w,
            )
        elif layer.kind == "depthwise_conv2d":
            layer_bytes += pack_depthwise_conv_layer(
                kernel_h=layer.kernel_h,
                kernel_w=layer.kernel_w,
                stride=layer.stride,
                channels=layer.filters,
                activation=layer.activation,
                alpha=layer.alpha,
                pad_h=layer.pad_h,
                pad_w=layer.pad_w,
            )
        elif layer.kind == "max_pool2d":
            layer_bytes += pack_pool_layer(
                pool_size=layer.pool_size,
                stride=layer.stride,
                pad_h=layer.pad_h,
                pad_w=layer.pad_w,
            )
        elif layer.kind == "avg_pool2d":
            layer_bytes += pack_avg_pool_layer(
                pool_size=layer.pool_size,
                stride=layer.stride,
                pad_h=layer.pad_h,
                pad_w=layer.pad_w,
            )
        elif layer.kind == "batch_norm2d":
            layer_bytes += pack_batch_norm_layer(channels=layer.channels)
        elif layer.kind == "layernorm2d":
            layer_bytes += pack_layernorm2d_layer(channels=layer.channels, eps=layer.eps)
        elif layer.kind == "convnextv2_block":
            layer_bytes += pack_convnextv2_block_layer(channels=layer.channels, eps=layer.eps)
        elif layer.kind == "mobilenetv4_uib":
            layer_bytes += pack_mobilenetv4_uib_layer(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                start_dw_kernel=layer.start_dw_kernel,
                middle_dw_kernel=layer.middle_dw_kernel,
                stride=layer.stride,
                expand_ratio=layer.expand_ratio,
                middle_dw_downsample=layer.middle_dw_downsample,
            )
        elif layer.kind == "resnet_basic_block":
            layer_bytes += pack_resnet_basic_block_layer(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                stride=layer.stride,
            )
        elif layer.kind == "flatten":
            layer_bytes += pack_flatten_layer()
        else:
            raise ValueError(f"unsupported layer kind: {layer.kind}")

    catalog = bytearray()
    for tensor in spec.weight_tensors:
        catalog += pack_tensor_desc(rank=tensor.ndim, dims=list(tensor.shape))
    for tensor in spec.bias_tensors:
        catalog += pack_tensor_desc(rank=tensor.ndim, dims=list(tensor.shape))

    body = header + layer_bytes + catalog + weights_blob + biases_blob
    if spec.tests and spec.tests.cases:
        body += pack_test_section(tolerance=spec.tests.tolerance, cases=spec.tests.cases)

    return body
