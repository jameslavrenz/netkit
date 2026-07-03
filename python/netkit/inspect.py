"""Inspect netkit .nk binary files."""

from __future__ import annotations

import io
import struct
from pathlib import Path

from .format import (
    HEADER_BYTES,
    TENSOR_DESC_BYTES,
    Activation,
    DType,
    LayerKind,
    unpack_header,
)


def _read_layer_body(stream: io.BytesIO, kind: int) -> dict:
    if kind == LayerKind.DENSE:
        units, activation, alpha = struct.unpack("<IB3xf", stream.read(12))
        return {
            "kind": "dense",
            "units": units,
            "activation": Activation(activation).name.lower(),
            "alpha": alpha,
        }
    if kind == LayerKind.CONV2D:
        kernel, stride, filters = struct.unpack("<III", stream.read(12))
        activation, pad_h, pad_w, _reserved, alpha = struct.unpack("<BBBBf", stream.read(8))
        return {
            "kind": "conv2d",
            "kernel_size": kernel,
            "stride": stride,
            "filters": filters,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "activation": Activation(activation).name.lower(),
            "alpha": alpha,
        }
    if kind == LayerKind.DEPTHWISE_CONV2D:
        kernel_h, stride, channels = struct.unpack("<III", stream.read(12))
        activation, pad_h, pad_w, kernel_w, alpha = struct.unpack("<BBBBf", stream.read(8))
        return {
            "kind": "depthwise_conv2d",
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "stride": stride,
            "filters": channels,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "activation": Activation(activation).name.lower(),
            "alpha": alpha,
        }
    if kind == LayerKind.MAX_POOL2D:
        pool, stride = struct.unpack("<II", stream.read(8))
        pad_h, pad_w, _reserved = struct.unpack("<BBH", stream.read(4))
        return {
            "kind": "max_pool2d",
            "pool_size": pool,
            "stride": stride,
            "pad_h": pad_h,
            "pad_w": pad_w,
        }
    if kind == LayerKind.AVG_POOL2D:
        pool, stride = struct.unpack("<II", stream.read(8))
        pad_h, pad_w, _reserved = struct.unpack("<BBH", stream.read(4))
        return {
            "kind": "avg_pool2d",
            "pool_size": pool,
            "stride": stride,
            "pad_h": pad_h,
            "pad_w": pad_w,
        }
    if kind == LayerKind.BATCH_NORM2D:
        channels, _reserved = struct.unpack("<II", stream.read(8))
        return {"kind": "batch_norm2d", "channels": channels}
    if kind == LayerKind.FLATTEN:
        return {"kind": "flatten"}
    raise ValueError(f"unsupported layer kind: {kind}")


def _layer_body_bytes(kind: int) -> int:
    if kind == LayerKind.DENSE:
        return 12
    if kind == LayerKind.CONV2D:
        return 20
    if kind == LayerKind.DEPTHWISE_CONV2D:
        return 20
    if kind in (LayerKind.MAX_POOL2D, LayerKind.AVG_POOL2D):
        return 12
    if kind == LayerKind.BATCH_NORM2D:
        return 8
    if kind == LayerKind.FLATTEN:
        return 0
    raise ValueError(f"unsupported layer kind: {kind}")


def _read_tensor_desc(stream: io.BytesIO) -> dict:
    rank, dtype, _pad = struct.unpack("<BBH", stream.read(4))
    dims = list(struct.unpack("<4I", stream.read(16)))
    num_elements = struct.unpack("<I", stream.read(4))[0]
    return {
        "rank": rank,
        "dtype": DType(dtype).name.lower(),
        "shape": dims[:rank],
        "num_elements": num_elements,
    }


def inspect_nk(path: str | Path) -> None:
    path = Path(path)
    stream = io.BytesIO(path.read_bytes())
    header = unpack_header(stream.read(HEADER_BYTES))

    print("netkit binary model (.nk)")
    print(f"  file:            {path}")
    print(f"  format version:  {header['version']}")
    print(f"  network:         {header['network_kind'].name.lower()}")
    print(f"  input rank:      {header['input_rank']}")
    shape = header["input_shape"][: header["input_rank"]]
    print(f"  input shape:     {shape}")
    print(f"  layers:          {header['num_layers']}")
    print(
        f"  weight tensors:  {header['num_weight_tensors']} ({header['weights_bytes']} bytes)"
    )
    print(f"  bias tensors:    {header['num_bias_tensors']} ({header['biases_bytes']} bytes)")

    print("\nLayer stack:")
    for index in range(header["num_layers"]):
        kind = struct.unpack("<B", stream.read(1))[0]
        stream.read(3)
        layer = _read_layer_body(stream, kind)
        print(f"  [{index}] {layer}")

    print("\nWeight tensor catalog:")
    for index in range(header["num_weight_tensors"]):
        desc = _read_tensor_desc(stream)
        print(f"  weight[{index}]: {desc}")

    print("\nBias tensor catalog:")
    for index in range(header["num_bias_tensors"]):
        desc = _read_tensor_desc(stream)
        print(f"  bias[{index}]: {desc}")
