"""Shared constants for the netkit .nk binary format (must match include/nk_format.hpp)."""

from __future__ import annotations

import struct
from enum import IntEnum

import numpy as np

MAGIC = b"NKIT"
TEST_MAGIC = b"TCAS"
QUANT_MAGIC = b"QUAN"
VERSION = 3
VERSION_QUANT = 4
HEADER_BYTES = 48
TENSOR_DESC_BYTES = 24
PAYLOAD_ALIGN = 4
FLAG_HAS_TESTS = 0x0001
FLAG_HAS_QUANT = 0x0002
# TCAS inputs are native int8 bytes (not float literals in [-128, 127]).
FLAG_HAS_INT8_TESTS = 0x0004
# QUAN section reserved u16 (second half of <HH> after num_layers).
QUAN_FLAG_PER_CHANNEL_WEIGHTS = 0x0001
MLP_LAYER_QUANT_BYTES = 32
MAX_CASE_NAME_LEN = 127
MAX_LAYERS = 100
MAX_TENSOR_CATALOG = 128
MAX_CASE_FLOATS = 16384


class NetworkKind(IntEnum):
    MLP = 1
    CNN = 2


class LayerKind(IntEnum):
    DENSE = 1
    CONV2D = 2
    MAX_POOL2D = 3
    FLATTEN = 4
    AVG_POOL2D = 5
    BATCH_NORM2D = 6
    DEPTHWISE_CONV2D = 7
    CONVNEXTV2_BLOCK = 8
    MOBILENETV4_UIB = 9
    RESNET_BASIC_BLOCK = 10
    LAYERNORM2D = 11
    YOLOX_DECOUPLED_HEAD = 12


class DType(IntEnum):
    FLOAT32 = 1
    INT8 = 2
    INT32 = 3


class Activation(IntEnum):
    NONE = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3
    LEAKY_RELU = 4
    RELU6 = 5
    SOFTMAX = 6


ACTIVATION_FROM_NAME = {
    "none": Activation.NONE,
    "relu": Activation.RELU,
    "sigmoid": Activation.SIGMOID,
    "tanh": Activation.TANH,
    "leaky_relu": Activation.LEAKY_RELU,
    "relu6": Activation.RELU6,
    "softmax": Activation.SOFTMAX,
}


def activation_from_name(name: str) -> Activation:
    return ACTIVATION_FROM_NAME.get(name, Activation.NONE)


def weight_payload_bytes(header: dict) -> int:
    """Total weight + bias payload bytes in the .nk file (excluded from MCU arena when coefs stay in flash)."""
    return int(header["weights_bytes"]) + int(header["biases_bytes"])


def payload_alignment_padding(meta_bytes: int) -> int:
    """Zero bytes inserted before the weight payload so float data is 4-byte aligned."""
    return (-meta_bytes) % PAYLOAD_ALIGN


def skip_payload_alignment_padding(stream, meta_end: int) -> int:
    """Skip zero padding after catalog; return file offset where weight payload starts."""
    pad = payload_alignment_padding(meta_end)
    if pad == 0:
        return meta_end
    peek = stream.read(pad)
    if len(peek) == pad and all(b == 0 for b in peek):
        return meta_end + pad
    stream.seek(meta_end)
    return meta_end


def pack_header(
    *,
    network_kind: NetworkKind,
    input_rank: int,
    input_shape: list[int],
    num_layers: int,
    num_weight_tensors: int,
    num_bias_tensors: int,
    weights_bytes: int,
    biases_bytes: int,
    flags: int = 0,
    version: int = VERSION,
) -> bytes:
    shape = list(input_shape) + [0] * (4 - len(input_shape))
    return struct.pack(
        "<4sIBBH4IIIIII",
        MAGIC,
        version,
        int(network_kind),
        input_rank,
        flags,
        *shape[:4],
        num_layers,
        num_weight_tensors,
        num_bias_tensors,
        weights_bytes,
        biases_bytes,
    )


def pack_layer_kind(kind: LayerKind) -> bytes:
    return struct.pack("<BBBB", int(kind), 0, 0, 0)


def pack_dense_layer(*, units: int, activation: Activation, alpha: float) -> bytes:
    return pack_layer_kind(LayerKind.DENSE) + struct.pack(
        "<IB3xf", units, int(activation), alpha
    )


def pack_conv_layer(
    *,
    kernel_size: int,
    stride: int,
    filters: int,
    activation: Activation,
    alpha: float,
    pad_h: int = 0,
    pad_w: int = 0,
    pad_extra: int = 0,
) -> bytes:
    return pack_layer_kind(LayerKind.CONV2D) + struct.pack(
        "<III", kernel_size, stride, filters
    ) + struct.pack("<BBBBf", int(activation), pad_h, pad_w, pad_extra, alpha)


def pack_depthwise_conv_layer(
    *,
    kernel_h: int,
    kernel_w: int,
    stride: int,
    channels: int,
    activation: Activation,
    alpha: float,
    pad_h: int = 0,
    pad_w: int = 0,
) -> bytes:
    if kernel_w <= 0 or kernel_w > 255:
        raise ValueError(f"depthwise kernel_w must be in 1..255, got {kernel_w}")
    if kernel_h <= 0:
        raise ValueError(f"depthwise kernel_h must be positive, got {kernel_h}")
    return pack_layer_kind(LayerKind.DEPTHWISE_CONV2D) + struct.pack(
        "<III", kernel_h, stride, channels
    ) + struct.pack("<BBBBf", int(activation), pad_h, pad_w, kernel_w, alpha)


def pack_pool_layer(
    *,
    pool_size: int,
    stride: int,
    pad_h: int = 0,
    pad_w: int = 0,
    reserved: int = 0,
) -> bytes:
    return pack_layer_kind(LayerKind.MAX_POOL2D) + struct.pack(
        "<II", pool_size, stride
    ) + struct.pack("<BBH", pad_h, pad_w, reserved)


def pack_avg_pool_layer(
    *,
    pool_size: int,
    stride: int,
    pad_h: int = 0,
    pad_w: int = 0,
    reserved: int = 0,
) -> bytes:
    return pack_layer_kind(LayerKind.AVG_POOL2D) + struct.pack(
        "<II", pool_size, stride
    ) + struct.pack("<BBH", pad_h, pad_w, reserved)


def pack_batch_norm_layer(*, channels: int) -> bytes:
    return pack_layer_kind(LayerKind.BATCH_NORM2D) + struct.pack("<II", channels, 0)


def pack_convnextv2_block_layer(*, channels: int, eps: float = 1e-6) -> bytes:
    return pack_layer_kind(LayerKind.CONVNEXTV2_BLOCK) + struct.pack("<IIf", channels, 0, eps)


def pack_mobilenetv4_uib_layer(
    *,
    in_channels: int,
    out_channels: int,
    start_dw_kernel: int,
    middle_dw_kernel: int,
    stride: int,
    expand_ratio: float,
    middle_dw_downsample: int = 1,
) -> bytes:
    return pack_layer_kind(LayerKind.MOBILENETV4_UIB) + struct.pack(
        "<II4BfI",
        in_channels,
        out_channels,
        start_dw_kernel,
        middle_dw_kernel,
        stride,
        middle_dw_downsample,
        float(expand_ratio),
        0,
    )


def pack_resnet_basic_block_layer(
    *,
    in_channels: int,
    out_channels: int,
    stride: int,
) -> bytes:
    return pack_layer_kind(LayerKind.RESNET_BASIC_BLOCK) + struct.pack(
        "<IIB3x",
        in_channels,
        out_channels,
        stride,
    )


def pack_layernorm2d_layer(*, channels: int, eps: float = 1e-6) -> bytes:
    return pack_layer_kind(LayerKind.LAYERNORM2D) + struct.pack("<IIf", channels, 0, eps)


def pack_yolox_decoupled_head_layer(
    *,
    in_channels: int,
    hidden_dim: int,
    num_classes: int,
    num_convs: int = 2,
) -> bytes:
    return pack_layer_kind(LayerKind.YOLOX_DECOUPLED_HEAD) + struct.pack(
        "<III B3x", in_channels, hidden_dim, num_classes, num_convs
    )


def pack_flatten_layer() -> bytes:
    return pack_layer_kind(LayerKind.FLATTEN)


def pack_tensor_desc(*, rank: int, dims: list[int], dtype: DType = DType.FLOAT32) -> bytes:
    padded = list(dims) + [0] * (4 - len(dims))
    num_elements = 1
    for dim in dims:
        num_elements *= dim
    return struct.pack("<BBH", rank, int(dtype), 0) + struct.pack(
        "<4II", *padded[:4], num_elements
    )


def pack_mlp_layer_quant(
    *,
    input_scale: float,
    input_zero_point: int,
    weight_scale: float,
    weight_zero_point: int,
    bias_scale: float,
    bias_zero_point: int,
    output_scale: float,
    output_zero_point: int,
) -> bytes:
    return struct.pack(
        "<fi fi fi fi",
        float(input_scale),
        int(input_zero_point),
        float(weight_scale),
        int(weight_zero_point),
        float(bias_scale),
        int(bias_zero_point),
        float(output_scale),
        int(output_zero_point),
    )


def pack_quant_section(quant_layers: list) -> bytes:
    """Pack QUAN section. Optional per-channel weight scales follow descriptors.

    When any layer has ``weight_scales`` (len > 1), sets QUAN_FLAG_PER_CHANNEL_WEIGHTS
    and appends for each layer: u32 channel_count + float32[channel_count] scales.
    channel_count==0 (or omitted) means use the per-tensor weight_scale field.
    """
    has_pc = any(
        getattr(layer, "weight_scales", None) is not None
        and len(getattr(layer, "weight_scales")) > 1
        for layer in quant_layers
    )
    flags = QUAN_FLAG_PER_CHANNEL_WEIGHTS if has_pc else 0
    blob = bytearray()
    blob += QUANT_MAGIC
    blob += struct.pack("<HH", len(quant_layers), flags)
    for layer in quant_layers:
        blob += pack_mlp_layer_quant(
            input_scale=layer.input_scale,
            input_zero_point=layer.input_zero_point,
            weight_scale=layer.weight_scale,
            weight_zero_point=layer.weight_zero_point,
            bias_scale=layer.bias_scale,
            bias_zero_point=layer.bias_zero_point,
            output_scale=layer.output_scale,
            output_zero_point=layer.output_zero_point,
        )
    if has_pc:
        for layer in quant_layers:
            scales = getattr(layer, "weight_scales", None)
            if scales is None or len(scales) <= 1:
                blob += struct.pack("<I", 0)
            else:
                arr = np.asarray(scales, dtype=np.float32).reshape(-1)
                blob += struct.pack("<I", int(arr.size))
                blob += arr.tobytes()
    return bytes(blob)


def pack_test_section(
    *,
    tolerance: float,
    cases: list,
    input_dtype: type[np.generic] | np.dtype = np.float32,
) -> bytes:
    """Pack TCAS regression cases.

    ``input_dtype`` is ``np.int8`` for quantized models (prequantized in Python)
    or ``np.float32`` for float models. Expected outputs stay float32 reference
    values for float models; int8 models typically only use ``label`` in C++.
    """
    blob = bytearray()
    blob += TEST_MAGIC
    blob += struct.pack("<If", len(cases), float(tolerance))
    in_dt = np.dtype(input_dtype)
    for case in cases:
        name = case.name.encode("utf-8")[:MAX_CASE_NAME_LEN]
        blob += struct.pack("<B", len(name))
        blob += name
        pad = (4 - ((1 + len(name)) % 4)) % 4
        blob += b"\x00" * pad
        blob += struct.pack("<i", int(case.label))
        inp = np.asarray(case.input, dtype=in_dt).reshape(-1)
        out = np.asarray(case.expected, dtype=np.float32).reshape(-1)
        blob += struct.pack("<I", int(inp.size))
        blob += inp.tobytes()
        blob += struct.pack("<I", int(out.size))
        blob += out.tobytes()
    return bytes(blob)


def unpack_header(data: bytes) -> dict:
    if len(data) < HEADER_BYTES:
        raise ValueError("truncated .nk header")
    fields = struct.unpack("<4sIBBH4IIIIII", data[:HEADER_BYTES])
    magic = fields[0]
    if magic != MAGIC:
        raise ValueError("invalid .nk magic")
    return {
        "magic": magic,
        "version": fields[1],
        "network_kind": NetworkKind(fields[2]),
        "input_rank": fields[3],
        "flags": fields[4],
        "input_shape": list(fields[5:9]),
        "num_layers": fields[9],
        "num_weight_tensors": fields[10],
        "num_bias_tensors": fields[11],
        "weights_bytes": fields[12],
        "biases_bytes": fields[13],
    }
