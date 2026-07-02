"""Convert supported ONNX graphs into netkit .nk models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import onnx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Requires onnx: pip install onnx numpy") from exc

from .format import Activation, activation_from_name
from .writer import LayerSpec, ModelSpec, write_nk


def _next_is_relu(nodes, index: int) -> bool:
    return index + 1 < len(nodes) and nodes[index + 1].op_type == "Relu"


def _next_is_softmax(nodes, index: int) -> bool:
    return index + 1 < len(nodes) and nodes[index + 1].op_type == "Softmax"


def _initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    from onnx import numpy_helper

    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _resolve_input_shape(model: onnx.ModelProto) -> tuple[str, list[int]]:
    if not model.graph.input:
        raise ValueError("ONNX graph has no inputs")

    dims = []
    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        dims.append(int(dim.dim_value))

    if len(dims) == 2:
        batch = dims[0] if dims[0] > 0 else 1
        return "mlp", [batch, dims[1]]
    if len(dims) == 4:
        _, channels, height, width = dims
        return "cnn", [height, width, channels]
    if len(dims) == 3:
        return "cnn", dims
    raise ValueError("unsupported ONNX input rank")


def _onnx_conv_to_netkit(weight: np.ndarray) -> np.ndarray:
    # ONNX [O, I, Kh, Kw] -> netkit [O, Kh, Kw, I]
    out_c, in_c, kh, kw = weight.shape
    w = weight.reshape(out_c, in_c, kh, kw)
    return np.transpose(w, (0, 2, 3, 1)).copy()


def _attr_int(node, name: str, default: int = 0) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _onnx_gemm_weight_to_netkit(weight: np.ndarray, *, trans_b: int) -> tuple[np.ndarray, int]:
    """Normalize ONNX Gemm B matrix to netkit dense layout [out, in]."""
    if trans_b:
        return weight.astype(np.float32), int(weight.shape[0])
    return weight.T.astype(np.float32), int(weight.shape[1])


def _attr_ints(node, name: str, default: list[int] | None = None) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return default or []


def onnx_to_spec(onnx_path: str | Path) -> ModelSpec:
    model = onnx.load(onnx_path)
    network, input_shape = _resolve_input_shape(model)
    initializers = _initializer_map(model)
    nodes = list(model.graph.node)

    layers: list[LayerSpec] = []
    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []

    if network == "mlp":
        in_features = input_shape[1]
        i = 0
        while i < len(nodes):
            node = nodes[i]
            if node.op_type in {"Relu", "Softmax"}:
                i += 1
                continue

            if node.op_type != "Gemm":
                raise ValueError(f"unsupported ONNX op for MLP: {node.op_type}")

            activation = Activation.NONE
            skip = 0
            if _next_is_relu(nodes, i):
                activation = Activation.RELU
                skip = 1
            elif _next_is_softmax(nodes, i):
                activation = Activation.SOFTMAX
                skip = 1

            weight = initializers[node.input[1]]
            bias = initializers[node.input[2]] if len(node.input) >= 3 else None
            trans_b = _attr_int(node, "transB", 0)
            packed_w, out_features = _onnx_gemm_weight_to_netkit(weight, trans_b=trans_b)
            layers.append(LayerSpec(kind="dense", units=out_features, activation=activation))
            weight_tensors.append(packed_w)
            bias_tensors.append(
                (bias.astype(np.float32) if bias is not None else np.zeros(out_features, dtype=np.float32))
            )
            in_features = out_features
            i += 1 + skip

        return ModelSpec(
            network="mlp",
            input_shape=input_shape,
            layers=layers,
            weight_tensors=weight_tensors,
            bias_tensors=bias_tensors,
        )

    # CNN
    channels = input_shape[2]
    spatial_h, spatial_w = input_shape[0], input_shape[1]
    dense_in = 0
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if node.op_type in {"Relu", "Softmax"}:
            i += 1
            continue

        activation = Activation.NONE
        skip = 0
        if _next_is_relu(nodes, i):
            activation = Activation.RELU
            skip = 1
        elif _next_is_softmax(nodes, i):
            activation = Activation.SOFTMAX
            skip = 1

        if node.op_type == "Conv":
            weight = initializers[node.input[1]]
            bias = initializers[node.input[2]] if len(node.input) >= 3 else None
            kernel_shape = _attr_ints(node, "kernel_shape")
            strides = _attr_ints(node, "strides")
            kernel = int(kernel_shape[0]) if kernel_shape else int(weight.shape[2])
            stride = int(strides[0]) if strides else kernel
            out_c = weight.shape[0]
            layers.append(
                LayerSpec(
                    kind="conv2d",
                    kernel_size=kernel,
                    stride=stride,
                    filters=out_c,
                    activation=activation,
                )
            )
            weight_tensors.append(_onnx_conv_to_netkit(weight.astype(np.float32)))
            bias_tensors.append(
                (bias.astype(np.float32) if bias is not None else np.zeros(out_c, dtype=np.float32))
            )
            spatial_h = (spatial_h - kernel) // stride + 1
            spatial_w = (spatial_w - kernel) // stride + 1
            channels = out_c
            i += 1 + skip
            continue

        if node.op_type == "MaxPool":
            kernel_shape = _attr_ints(node, "kernel_shape")
            strides = _attr_ints(node, "strides")
            kernel = int(kernel_shape[0]) if kernel_shape else 2
            stride = int(strides[0]) if strides else kernel
            layers.append(LayerSpec(kind="max_pool2d", pool_size=kernel, stride=stride))
            spatial_h = (spatial_h - kernel) // stride + 1
            spatial_w = (spatial_w - kernel) // stride + 1
            i += 1
            continue

        if node.op_type == "Flatten":
            layers.append(LayerSpec(kind="flatten"))
            dense_in = spatial_h * spatial_w * channels
            i += 1
            continue

        if node.op_type == "Gemm":
            weight = initializers[node.input[1]]
            bias = initializers[node.input[2]] if len(node.input) >= 3 else None
            trans_b = _attr_int(node, "transB", 0)
            packed_w, out_features = _onnx_gemm_weight_to_netkit(weight, trans_b=trans_b)
            layers.append(
                LayerSpec(kind="dense", units=out_features, activation=activation)
            )
            weight_tensors.append(packed_w)
            bias_tensors.append(
                (bias.astype(np.float32) if bias is not None else np.zeros(out_features, dtype=np.float32))
            )
            dense_in = out_features
            i += 1 + skip
            continue

        raise ValueError(f"unsupported ONNX op for CNN: {node.op_type}")

    return ModelSpec(
        network="cnn",
        input_shape=input_shape,
        layers=layers,
        weight_tensors=weight_tensors,
        bias_tensors=bias_tensors,
    )


def convert_onnx_to_nk(onnx_path: str | Path, output_path: str | Path | None = None) -> Path:
    onnx_path = Path(onnx_path)
    output_path = Path(output_path) if output_path else onnx_path.with_suffix(".nk")
    spec = onnx_to_spec(onnx_path)
    write_nk(output_path, spec)
    return output_path
