"""Shared constants for the netkit .nk binary format (must match include/nk_format.hpp)."""

from __future__ import annotations

import struct
from enum import IntEnum

import numpy as np

MAGIC = b"NKIT"
TEST_MAGIC = b"TCAS"
VERSION = 2
HEADER_BYTES = 48
TENSOR_DESC_BYTES = 24
FLAG_HAS_TESTS = 0x0001
MAX_CASE_NAME_LEN = 127


class NetworkKind(IntEnum):
    MLP = 1
    CNN = 2


class LayerKind(IntEnum):
    DENSE = 1
    CONV2D = 2
    MAX_POOL2D = 3
    FLATTEN = 4


class DType(IntEnum):
    FLOAT32 = 1


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
) -> bytes:
    shape = list(input_shape) + [0] * (4 - len(input_shape))
    return struct.pack(
        "<4sIBBH4IIIIII",
        MAGIC,
        VERSION,
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
    *, kernel_size: int, stride: int, filters: int, activation: Activation, alpha: float
) -> bytes:
    return pack_layer_kind(LayerKind.CONV2D) + struct.pack(
        "<III", kernel_size, stride, filters
    ) + struct.pack("<B3xf", int(activation), alpha)


def pack_pool_layer(*, pool_size: int, stride: int) -> bytes:
    return pack_layer_kind(LayerKind.MAX_POOL2D) + struct.pack("<II", pool_size, stride)


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


def pack_test_section(*, tolerance: float, cases: list) -> bytes:
    blob = bytearray()
    blob += TEST_MAGIC
    blob += struct.pack("<If", len(cases), float(tolerance))
    for case in cases:
        name = case.name.encode("utf-8")[:MAX_CASE_NAME_LEN]
        blob += struct.pack("<B", len(name))
        blob += name
        pad = (4 - ((1 + len(name)) % 4)) % 4
        blob += b"\x00" * pad
        blob += struct.pack("<i", int(case.label))
        inp = np.asarray(case.input, dtype=np.float32).reshape(-1)
        out = np.asarray(case.expected, dtype=np.float32).reshape(-1)
        blob += struct.pack("<I", inp.size)
        blob += inp.tobytes()
        blob += struct.pack("<I", out.size)
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
