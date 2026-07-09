"""Write netkit .nk binary model files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .pad_encoding import encode_pad_extra, encode_pool_reserved
from .format import (
    FLAG_HAS_TESTS,
    FLAG_HAS_QUANT,
    FLAG_HAS_INT8_TESTS,
    Activation,
    DType,
    NetworkKind,
    VERSION,
    VERSION_QUANT,
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
    pack_yolox_decoupled_head_layer,
    pack_quant_section,
    payload_alignment_padding,
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
    pad_h_end: int = 0
    pad_w_end: int = 0
    pool_size: int = 2
    pool_w: int = 0
    channels: int = 0
    eps: float = 1e-6
    in_channels: int = 0
    out_channels: int = 0
    start_dw_kernel: int = 0
    middle_dw_kernel: int = 0
    middle_dw_downsample: int = 1
    expand_ratio: float = 1.0
    hidden_dim: int = 256
    num_classes: int = 80
    num_convs: int = 2


@dataclass
class QuantLayerParams:
    input_scale: float
    input_zero_point: int
    weight_scale: float
    weight_zero_point: int
    bias_scale: float
    bias_zero_point: int
    output_scale: float
    output_zero_point: int
    # Optional per-output-channel weight scales (TFLite-style). When set with
    # len > 1, pack_quant_section emits QUAN_FLAG_PER_CHANNEL_WEIGHTS. bias_scale
    # is then a fallback; runtime uses input_scale * weight_scales[c].
    weight_scales: list[float] | np.ndarray | None = None


@dataclass
class RegressionCase:
    name: str
    # Float models: float32. Quantized models: native int8 (prequantized in Python).
    input: list[float] | list[int] | np.ndarray
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
    weight_dtypes: list[DType] | None = None
    bias_dtypes: list[DType] | None = None
    quant_layers: list[QuantLayerParams] | None = None
    tests: RegressionSuite | None = None


def _dtype_to_numpy(dtype: DType):
    if dtype == DType.FLOAT32:
        return np.float32
    if dtype == DType.INT8:
        return np.int8
    if dtype == DType.INT32:
        return np.int32
    raise ValueError(f"unsupported dtype: {dtype}")


def write_nk(path: str | Path, spec: ModelSpec) -> None:
    Path(path).write_bytes(write_nk_bytes(spec))


def write_nk_bytes(spec: ModelSpec) -> bytes:
    network_kind = NetworkKind.MLP if spec.network == "mlp" else NetworkKind.CNN
    input_rank = len(spec.input_shape)
    quantized = spec.quant_layers is not None and len(spec.quant_layers) > 0

    weight_dtypes = spec.weight_dtypes or [DType.FLOAT32] * len(spec.weight_tensors)
    bias_dtypes = spec.bias_dtypes or [DType.FLOAT32] * len(spec.bias_tensors)

    weights_blob = b"".join(
        np.ascontiguousarray(w, dtype=_dtype_to_numpy(dt)).tobytes()
        for w, dt in zip(spec.weight_tensors, weight_dtypes)
    )
    biases_blob = b"".join(
        np.ascontiguousarray(b, dtype=_dtype_to_numpy(dt)).tobytes()
        for b, dt in zip(spec.bias_tensors, bias_dtypes)
    )

    flags = 0
    if spec.tests and spec.tests.cases:
        flags |= FLAG_HAS_TESTS
        if quantized:
            flags |= FLAG_HAS_INT8_TESTS
    if quantized:
        flags |= FLAG_HAS_QUANT

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
        version=VERSION_QUANT if quantized else VERSION,
    )

    layer_bytes = bytearray()
    for layer in spec.layers:
        if layer.kind == "dense":
            layer_bytes += pack_dense_layer(
                units=layer.units, activation=layer.activation, alpha=layer.alpha
            )
        elif layer.kind == "conv2d":
            pad_h_end = layer.pad_h_end or layer.pad_h
            pad_w_end = layer.pad_w_end or layer.pad_w
            pad_extra = encode_pad_extra(layer.pad_h, layer.pad_w, pad_h_end, pad_w_end)
            layer_bytes += pack_conv_layer(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                filters=layer.filters,
                activation=layer.activation,
                alpha=layer.alpha,
                pad_h=layer.pad_h,
                pad_w=layer.pad_w,
                pad_extra=pad_extra,
            )
        elif layer.kind == "depthwise_conv2d":
            kh = layer.kernel_h
            kw = layer.kernel_w
            pad_h_end = layer.pad_h_end or layer.pad_h
            pad_w_end = layer.pad_w_end or layer.pad_w
            kernel_w_byte = kw
            if kh != kw:
                if pad_h_end != layer.pad_h or pad_w_end != layer.pad_w:
                    kernel_w_byte = encode_pad_extra(
                        layer.pad_h, layer.pad_w, pad_h_end, pad_w_end
                    )
            elif kh == kw and (pad_h_end != layer.pad_h or pad_w_end != layer.pad_w):
                kernel_w_byte = encode_pad_extra(layer.pad_h, layer.pad_w, pad_h_end, pad_w_end)
            layer_bytes += pack_depthwise_conv_layer(
                kernel_h=kh,
                kernel_w=kernel_w_byte,
                stride=layer.stride,
                channels=layer.filters,
                activation=layer.activation,
                alpha=layer.alpha,
                pad_h=layer.pad_h,
                pad_w=layer.pad_w,
            )
        elif layer.kind == "max_pool2d":
            pool_w = layer.pool_w or layer.pool_size
            pad_h_end = layer.pad_h_end or layer.pad_h
            pad_w_end = layer.pad_w_end or layer.pad_w
            reserved = encode_pool_reserved(
                pool_h=layer.pool_size,
                pool_w=pool_w,
                top=layer.pad_h,
                left=layer.pad_w,
                bottom=pad_h_end,
                right=pad_w_end,
            )
            layer_bytes += pack_pool_layer(
                pool_size=layer.pool_size,
                stride=layer.stride,
                pad_h=layer.pad_h,
                pad_w=layer.pad_w,
                reserved=reserved,
            )
        elif layer.kind == "avg_pool2d":
            pool_w = layer.pool_w or layer.pool_size
            pad_h_end = layer.pad_h_end or layer.pad_h
            pad_w_end = layer.pad_w_end or layer.pad_w
            reserved = encode_pool_reserved(
                pool_h=layer.pool_size,
                pool_w=pool_w,
                top=layer.pad_h,
                left=layer.pad_w,
                bottom=pad_h_end,
                right=pad_w_end,
            )
            layer_bytes += pack_avg_pool_layer(
                pool_size=layer.pool_size,
                stride=layer.stride,
                pad_h=layer.pad_h,
                pad_w=layer.pad_w,
                reserved=reserved,
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
        elif layer.kind == "yolox_decoupled_head":
            layer_bytes += pack_yolox_decoupled_head_layer(
                in_channels=layer.in_channels,
                hidden_dim=layer.hidden_dim,
                num_classes=layer.num_classes,
                num_convs=layer.num_convs,
            )
        elif layer.kind == "flatten":
            layer_bytes += pack_flatten_layer()
        else:
            raise ValueError(f"unsupported layer kind: {layer.kind}")

    catalog = bytearray()
    for tensor, dtype in zip(spec.weight_tensors, weight_dtypes):
        catalog += pack_tensor_desc(rank=tensor.ndim, dims=list(tensor.shape), dtype=dtype)
    for tensor, dtype in zip(spec.bias_tensors, bias_dtypes):
        catalog += pack_tensor_desc(rank=tensor.ndim, dims=list(tensor.shape), dtype=dtype)

    quant_bytes = b""
    if quantized:
        quant_bytes = pack_quant_section(spec.quant_layers)

    meta = header + layer_bytes + catalog + quant_bytes
    align_pad = payload_alignment_padding(len(meta))
    body = meta + (b"\x00" * align_pad) + weights_blob + biases_blob
    if spec.tests and spec.tests.cases:
        body += pack_test_section(
            tolerance=spec.tests.tolerance,
            cases=spec.tests.cases,
            input_dtype=np.int8 if quantized else np.float32,
        )

    return body
