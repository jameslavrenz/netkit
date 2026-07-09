"""Read quantized int8 .nk models (weights, biases, per-layer quant params)."""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .format import (
    DType,
    FLAG_HAS_QUANT,
    HEADER_BYTES,
    MLP_LAYER_QUANT_BYTES,
    QUANT_MAGIC,
    QUAN_FLAG_PER_CHANNEL_WEIGHTS,
    skip_payload_alignment_padding,
    unpack_header,
)
from .inspect import _read_layer_body, _read_tensor_desc
from .reader import _layers_to_arch
from .writer import QuantLayerParams


@dataclass(frozen=True)
class QuantNkBundle:
    arch: dict
    input_shape: list[int]
    layer_bodies: list[dict]
    weight_tensors: list[np.ndarray]
    bias_tensors: list[np.ndarray]
    quant_layers: list[QuantLayerParams]


def _read_quant_layer_params(stream: io.BytesIO) -> QuantLayerParams:
    chunk = stream.read(MLP_LAYER_QUANT_BYTES)
    fields = struct.unpack("<fi fi fi fi", chunk)
    return QuantLayerParams(
        input_scale=float(fields[0]),
        input_zero_point=int(fields[1]),
        weight_scale=float(fields[2]),
        weight_zero_point=int(fields[3]),
        bias_scale=float(fields[4]),
        bias_zero_point=int(fields[5]),
        output_scale=float(fields[6]),
        output_zero_point=int(fields[7]),
    )


def read_quant_nk(path: str | Path) -> QuantNkBundle:
    """Parse a quantized .nk file into int8/int32 tensors and quant metadata."""
    path = Path(path)
    stream = io.BytesIO(path.read_bytes())
    header = unpack_header(stream.read(HEADER_BYTES))
    if not (header.get("flags", 0) & FLAG_HAS_QUANT):
        raise ValueError(f"{path} is not a quantized .nk model")

    input_shape = header["input_shape"][: header["input_rank"]]
    layer_bodies: list[dict] = []
    for _ in range(header["num_layers"]):
        kind = struct.unpack("<B", stream.read(1))[0]
        stream.read(3)
        layer_bodies.append(_read_layer_body(stream, kind))

    weight_descs = [_read_tensor_desc(stream) for _ in range(header["num_weight_tensors"])]
    bias_descs = [_read_tensor_desc(stream) for _ in range(header["num_bias_tensors"])]

    magic = stream.read(4)
    if magic != QUANT_MAGIC:
        raise ValueError(f"invalid QUAN section in {path}")
    num_quant_layers, quan_flags = struct.unpack("<HH", stream.read(4))
    quant_layers = [_read_quant_layer_params(stream) for _ in range(num_quant_layers)]
    if quan_flags & QUAN_FLAG_PER_CHANNEL_WEIGHTS:
        for ql in quant_layers:
            (n_ch,) = struct.unpack("<I", stream.read(4))
            if n_ch == 0:
                continue
            scales = np.frombuffer(stream.read(n_ch * 4), dtype=np.float32).copy()
            if scales.size != n_ch:
                raise ValueError(f"truncated per-channel weight scales in {path}")
            ql.weight_scales = scales
            ql.weight_scale = float(scales[0])

    meta_end = stream.tell()
    skip_payload_alignment_padding(stream, meta_end)

    weights_blob = stream.read(header["weights_bytes"])
    biases_blob = stream.read(header["biases_bytes"])
    if len(weights_blob) != header["weights_bytes"] or len(biases_blob) != header["biases_bytes"]:
        raise ValueError(f"truncated weight payload in {path}")

    weight_tensors: list[np.ndarray] = []
    offset = 0
    for desc in weight_descs:
        nbytes = desc["num_elements"] * { "float32": 4, "int8": 1, "int32": 4 }[desc["dtype"]]
        chunk = weights_blob[offset : offset + nbytes]
        dtype = {"float32": np.float32, "int8": np.int8, "int32": np.int32}[desc["dtype"]]
        weight_tensors.append(np.frombuffer(chunk, dtype=dtype).copy())
        offset += nbytes

    bias_tensors: list[np.ndarray] = []
    offset = 0
    for desc in bias_descs:
        nbytes = desc["num_elements"] * { "float32": 4, "int8": 1, "int32": 4 }[desc["dtype"]]
        chunk = biases_blob[offset : offset + nbytes]
        dtype = {"float32": np.float32, "int8": np.int8, "int32": np.int32}[desc["dtype"]]
        bias_tensors.append(np.frombuffer(chunk, dtype=dtype).copy())
        offset += nbytes

    network = header["network_kind"].name.lower()
    arch = _layers_to_arch(network, input_shape, layer_bodies)
    return QuantNkBundle(
        arch=arch,
        input_shape=input_shape,
        layer_bodies=layer_bodies,
        weight_tensors=weight_tensors,
        bias_tensors=bias_tensors,
        quant_layers=quant_layers,
    )
