"""Read netkit .nk binary model files."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np

from .cnn_layers import depthwise_arch_entry
from .format import HEADER_BYTES, Activation, LayerKind, NetworkKind, FLAG_HAS_TESTS, TEST_MAGIC, unpack_header
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
            arch_layers.append(entry)
        elif kind == "batch_norm2d":
            arch_layers.append({"type": "batch_norm2d", "channels": layer["channels"]})
        elif kind == "flatten":
            arch_layers.append({"type": "flatten"})
        else:
            raise ValueError(f"unsupported layer kind: {kind}")

    return {"network": network, "input": input_shape, "layers": arch_layers}


def read_nk(path: str | Path) -> tuple[dict, np.ndarray]:
    """Return architecture dict and interleaved flat weights (W,B per layer)."""
    path = Path(path)
    stream = io.BytesIO(path.read_bytes())
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

    weights_blob = stream.read(header["weights_bytes"])
    biases_blob = stream.read(header["biases_bytes"])
    if len(weights_blob) != header["weights_bytes"] or len(biases_blob) != header["biases_bytes"]:
        raise ValueError(f"truncated weight payload in {path}")

    weight_arrays: list[np.ndarray] = []
    offset = 0
    for desc in weight_descs:
        nbytes = desc["num_elements"] * 4
        weight_arrays.append(
            np.frombuffer(weights_blob, dtype=np.float32, count=desc["num_elements"], offset=offset).copy()
        )
        offset += nbytes

    bias_arrays: list[np.ndarray] = []
    offset = 0
    for desc in bias_descs:
        nbytes = desc["num_elements"] * 4
        bias_arrays.append(
            np.frombuffer(biases_blob, dtype=np.float32, count=desc["num_elements"], offset=offset).copy()
        )
        offset += nbytes

    flat_parts: list[np.ndarray] = []
    for w, b in zip(weight_arrays, bias_arrays):
        flat_parts.append(w.reshape(-1))
        flat_parts.append(b.reshape(-1))
    flat_weights = np.concatenate(flat_parts).astype(np.float32)

    arch = _layers_to_arch(network, input_shape, layers)
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

    stream.read(header["weights_bytes"])
    stream.read(header["biases_bytes"])

    magic = stream.read(4)
    if magic != TEST_MAGIC:
        raise ValueError(f"missing TCAS section in {path}")

    case_count, tolerance = struct.unpack("<If", stream.read(8))
    cases: list[RegressionCase] = []
    for _ in range(case_count):
        name_len = struct.unpack("<B", stream.read(1))[0]
        name = stream.read(name_len).decode("utf-8")
        pad = (4 - ((1 + name_len) % 4)) % 4
        stream.read(pad)
        label = struct.unpack("<i", stream.read(4))[0]
        input_count = struct.unpack("<I", stream.read(4))[0]
        inp = np.frombuffer(stream.read(input_count * 4), dtype=np.float32).copy()
        output_count = struct.unpack("<I", stream.read(4))[0]
        expected = np.frombuffer(stream.read(output_count * 4), dtype=np.float32).copy()
        cases.append(
            RegressionCase(name=name, input=inp, expected=expected, label=label)
        )

    return RegressionSuite(tolerance=float(tolerance), cases=cases)
