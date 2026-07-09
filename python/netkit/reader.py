"""Read netkit .nk binary model files."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np

from .cnn_layers import _layer_weight_tensor_count, conv2d_input_channels, depthwise_arch_entry, reconcile_depthwise_kernel
from .format import (
    HEADER_BYTES,
    Activation,
    FLAG_HAS_INT8_TESTS,
    FLAG_HAS_QUANT,
    FLAG_HAS_TESTS,
    LayerKind,
    MLP_LAYER_QUANT_BYTES,
    NetworkKind,
    QUANT_MAGIC,
    QUAN_FLAG_PER_CHANNEL_WEIGHTS,
    TEST_MAGIC,
    skip_payload_alignment_padding,
    unpack_header,
)


def _skip_quan_section(stream: io.BytesIO) -> None:
    """Advance past a QUAN block (descriptors + optional per-channel scales)."""
    magic = stream.read(4)
    if magic != QUANT_MAGIC:
        raise ValueError("invalid QUAN section in .nk file")
    num_quant_layers, quan_flags = struct.unpack("<HH", stream.read(4))
    stream.read(num_quant_layers * MLP_LAYER_QUANT_BYTES)
    if quan_flags & QUAN_FLAG_PER_CHANNEL_WEIGHTS:
        for _ in range(num_quant_layers):
            (n_ch,) = struct.unpack("<I", stream.read(4))
            if n_ch:
                stream.read(n_ch * 4)
from .inspect import _read_layer_body, _read_tensor_desc
from .writer import RegressionCase, RegressionSuite


def _layers_to_arch(network: str, input_shape: list[int], layers: list[dict]) -> dict:
    arch_layers: list[dict] = []
    for layer in layers:
        kind = layer["kind"]
        if kind == "dense":
            entry = {
                "type": "dense",
                "units": layer["units"],
                "activation": layer["activation"],
            }
            if layer.get("activation") == "leaky_relu":
                entry["alpha"] = float(layer.get("alpha", 0.01))
            arch_layers.append(entry)
        elif kind == "conv2d":
            entry = {
                "type": "conv2d",
                "kernel_size": layer["kernel_size"],
                "stride": layer["stride"],
                "filters": layer["filters"],
                "activation": layer["activation"],
            }
            if layer.get("pad_h", 0):
                entry["pad_h"] = layer["pad_h"]
            if layer.get("pad_w", 0):
                entry["pad_w"] = layer["pad_w"]
            if layer.get("pad_h_end", layer.get("pad_h", 0)) != layer.get("pad_h", 0):
                entry["pad_h_end"] = layer["pad_h_end"]
            if layer.get("pad_w_end", layer.get("pad_w", 0)) != layer.get("pad_w", 0):
                entry["pad_w_end"] = layer["pad_w_end"]
            if layer.get("activation") == "leaky_relu":
                entry["alpha"] = float(layer.get("alpha", 0.01))
            arch_layers.append(entry)
        elif kind == "depthwise_conv2d":
            arch_layers.append(depthwise_arch_entry(layer))
        elif kind == "max_pool2d":
            entry = {
                "type": "max_pool2d",
                "pool_size": layer["pool_size"],
                "stride": layer["stride"],
            }
            if layer.get("pad_h", 0):
                entry["pad_h"] = layer["pad_h"]
            if layer.get("pad_w", 0):
                entry["pad_w"] = layer["pad_w"]
            if layer.get("pool_w", layer["pool_size"]) != layer["pool_size"]:
                entry["pool_w"] = layer["pool_w"]
            if layer.get("pad_h_end", layer.get("pad_h", 0)) != layer.get("pad_h", 0):
                entry["pad_h_end"] = layer["pad_h_end"]
            if layer.get("pad_w_end", layer.get("pad_w", 0)) != layer.get("pad_w", 0):
                entry["pad_w_end"] = layer["pad_w_end"]
            arch_layers.append(entry)
        elif kind == "avg_pool2d":
            entry = {
                "type": "avg_pool2d",
                "pool_size": layer["pool_size"],
                "stride": layer["stride"],
            }
            if layer.get("pad_h", 0):
                entry["pad_h"] = layer["pad_h"]
            if layer.get("pad_w", 0):
                entry["pad_w"] = layer["pad_w"]
            if layer.get("pool_w", layer["pool_size"]) != layer["pool_size"]:
                entry["pool_w"] = layer["pool_w"]
            if layer.get("pad_h_end", layer.get("pad_h", 0)) != layer.get("pad_h", 0):
                entry["pad_h_end"] = layer["pad_h_end"]
            if layer.get("pad_w_end", layer.get("pad_w", 0)) != layer.get("pad_w", 0):
                entry["pad_w_end"] = layer["pad_w_end"]
            arch_layers.append(entry)
        elif kind == "batch_norm2d":
            arch_layers.append({"type": "batch_norm2d", "channels": layer["channels"]})
        elif kind == "layernorm2d":
            arch_layers.append(
                {
                    "type": "layernorm2d",
                    "channels": layer["channels"],
                    "eps": float(layer.get("eps", 1e-6)),
                }
            )
        elif kind == "convnextv2_block":
            arch_layers.append(
                {
                    "type": "convnextv2_block",
                    "channels": layer["channels"],
                    "eps": float(layer.get("eps", 1e-6)),
                }
            )
        elif kind == "mobilenetv4_uib":
            arch_layers.append(
                {
                    "type": "mobilenetv4_uib",
                    "in_channels": layer["in_channels"],
                    "out_channels": layer["out_channels"],
                    "start_dw_kernel": layer["start_dw_kernel"],
                    "middle_dw_kernel": layer["middle_dw_kernel"],
                    "stride": layer["stride"],
                    "middle_dw_downsample": layer.get("middle_dw_downsample", 1),
                    "expand_ratio": float(layer["expand_ratio"]),
                }
            )
        elif kind == "resnet_basic_block":
            arch_layers.append(
                {
                    "type": "resnet_basic_block",
                    "in_channels": layer["in_channels"],
                    "out_channels": layer["out_channels"],
                    "stride": layer["stride"],
                }
            )
        elif kind == "yolox_decoupled_head":
            arch_layers.append(
                {
                    "type": "yolox_decoupled_head",
                    "in_channels": layer["in_channels"],
                    "hidden_dim": layer["hidden_dim"],
                    "num_classes": layer["num_classes"],
                    "num_convs": layer["num_convs"],
                }
            )
        elif kind == "flatten":
            arch_layers.append({"type": "flatten"})
        else:
            raise ValueError(f"unsupported layer kind: {kind}")

    return {"network": network, "input": input_shape, "layers": arch_layers}



_DTYPE_ITEMSIZE: dict[str, tuple[str, int]] = {
    "float32": ("float32", 4),
    "int8": ("int8", 1),
    "int32": ("int32", 4),
}


def _read_payload_arrays(blob: bytes, descs: list[dict]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    offset = 0
    for desc in descs:
        dtype_name, itemsize = _DTYPE_ITEMSIZE[desc["dtype"]]
        nbytes = desc["num_elements"] * itemsize
        chunk = blob[offset : offset + nbytes]
        if len(chunk) != nbytes:
            raise ValueError("truncated tensor payload in .nk bytes")
        arrays.append(np.frombuffer(chunk, dtype=np.dtype(dtype_name)).copy())
        offset += nbytes
    if offset != len(blob):
        raise ValueError("weight payload size mismatch in .nk bytes")
    return arrays

def read_nk_bytes(data: bytes) -> tuple[dict, np.ndarray]:
    """Return architecture dict and interleaved flat weights from in-memory .nk bytes."""
    return _read_nk_stream(io.BytesIO(data))


def read_nk(path: str | Path) -> tuple[dict, np.ndarray]:
    """Return architecture dict and interleaved flat weights (W,B per layer)."""
    path = Path(path)
    return _read_nk_stream(io.BytesIO(path.read_bytes()))


def _read_nk_stream(stream: io.BytesIO) -> tuple[dict, np.ndarray]:
    header = unpack_header(stream.read(HEADER_BYTES))

    network = header["network_kind"].name.lower()
    input_shape = header["input_shape"][: header["input_rank"]]

    layers: list[dict] = []
    for _ in range(header["num_layers"]):
        kind = struct.unpack("<B", stream.read(1))[0]
        stream.read(3)
        layers.append(_read_layer_body(stream, kind))

    weight_descs = [_read_tensor_desc(stream) for _ in range(header["num_weight_tensors"])]
    bias_descs = [_read_tensor_desc(stream) for _ in range(header["num_bias_tensors"])]

    if header.get("flags", 0) & FLAG_HAS_QUANT:
        _skip_quan_section(stream)

    meta_end = stream.tell()
    payload_start = skip_payload_alignment_padding(stream, meta_end)

    weights_blob = stream.read(header["weights_bytes"])
    biases_blob = stream.read(header["biases_bytes"])
    if len(weights_blob) != header["weights_bytes"] or len(biases_blob) != header["biases_bytes"]:
        raise ValueError("truncated weight payload in .nk bytes")

    weight_arrays = _read_payload_arrays(weights_blob, weight_descs)
    bias_arrays = _read_payload_arrays(biases_blob, bias_descs)

    flat_parts: list[np.ndarray] = []
    for w, b in zip(weight_arrays, bias_arrays):
        flat_parts.append(w.reshape(-1))
        flat_parts.append(b.reshape(-1))
    flat_weights = (
        np.concatenate(flat_parts).astype(np.float32)
        if flat_parts
        else np.array([], dtype=np.float32)
    )

    arch = _layers_to_arch(network, input_shape, layers)
    weight_index = 0
    for layer_index, layer in enumerate(arch["layers"]):
        if layer["type"] == "conv2d":
            if weight_index >= len(weight_arrays):
                raise ValueError("missing conv weight tensor in .nk catalog")
            w_arr = weight_arrays[weight_index]
            layer["in_channels"] = conv2d_input_channels(
                layer, 0, weight_elems=int(w_arr.size)
            )
        elif layer["type"] == "depthwise_conv2d":
            if weight_index >= len(weight_arrays):
                raise ValueError("missing depthwise weight tensor in .nk catalog")
            parsed = layers[layer_index]
            kernel_w_byte = int(parsed.get("_kernel_w_byte", layer["kernel_w"]))
            kw, top, left, bottom, right = reconcile_depthwise_kernel(
                kernel_h=int(layer["kernel_h"]),
                kernel_w_byte=kernel_w_byte,
                pad_h=int(layer.get("pad_h", 0)),
                pad_w=int(layer.get("pad_w", 0)),
                channels=int(layer["filters"]),
                weight_elems=int(weight_arrays[weight_index].size),
            )
            layer["kernel_w"] = kw
            layer["pad_h"] = top
            layer["pad_w"] = left
            if bottom != top:
                layer["pad_h_end"] = bottom
            elif "pad_h_end" in layer:
                del layer["pad_h_end"]
            if right != left:
                layer["pad_w_end"] = right
            elif "pad_w_end" in layer:
                del layer["pad_w_end"]
        weight_index += _layer_weight_tensor_count(layer)
    return arch, flat_weights


def read_test_suite(path: str | Path) -> RegressionSuite | None:
    """Return embedded TCAS regression cases from a .nk file, or None if absent."""
    path = Path(path)
    stream = io.BytesIO(path.read_bytes())
    header = unpack_header(stream.read(HEADER_BYTES))
    if not (header.get("flags", 0) & FLAG_HAS_TESTS):
        return None

    for _ in range(header["num_layers"]):
        kind = struct.unpack("<B", stream.read(1))[0]
        stream.read(3)
        _read_layer_body(stream, kind)

    for _ in range(header["num_weight_tensors"] + header["num_bias_tensors"]):
        _read_tensor_desc(stream)

    if header.get("flags", 0) & FLAG_HAS_QUANT:
        _skip_quan_section(stream)

    meta_end = stream.tell()
    skip_payload_alignment_padding(stream, meta_end)
    stream.read(header["weights_bytes"])
    stream.read(header["biases_bytes"])

    magic = stream.read(4)
    if magic != TEST_MAGIC:
        raise ValueError(f"missing TCAS section in {path}")

    case_count, tolerance = struct.unpack("<If", stream.read(8))
    int8_inputs = bool(header["flags"] & FLAG_HAS_INT8_TESTS)
    cases: list[RegressionCase] = []
    for _ in range(case_count):
        name_len = struct.unpack("<B", stream.read(1))[0]
        name = stream.read(name_len).decode("utf-8")
        pad = (4 - ((1 + name_len) % 4)) % 4
        stream.read(pad)
        label = struct.unpack("<i", stream.read(4))[0]
        input_count = struct.unpack("<I", stream.read(4))[0]
        if int8_inputs:
            inp = np.frombuffer(stream.read(input_count), dtype=np.int8).copy()
        else:
            inp = np.frombuffer(stream.read(input_count * 4), dtype=np.float32).copy()
        output_count = struct.unpack("<I", stream.read(4))[0]
        expected = np.frombuffer(stream.read(output_count * 4), dtype=np.float32).copy()
        cases.append(
            RegressionCase(name=name, input=inp, expected=expected, label=label)
        )

    return RegressionSuite(tolerance=float(tolerance), cases=cases)
