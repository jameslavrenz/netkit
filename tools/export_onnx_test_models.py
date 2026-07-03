#!/usr/bin/env python3
"""Export netkit .nk models to ONNX for Python parity tests.

Requires: pip install onnx numpy

Run from repo root:
    python3 tools/export_onnx_test_models.py
    make export-onnx-test
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Requires onnx: pip install onnx numpy") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from netkit import read_nk
from netkit.cnn_layers import depthwise_kernel_hw

MODELS = ROOT / "models"

MODEL_NKS = [
    "test_mlp.nk",
    "mlp_hand.nk",
    "test_cnn.nk",
    "cnn_4x4_single.nk",
    "cnn_hand.nk",
    "speech_kws.nk",
    "op_matrix_mlp.nk",
    "op_matrix_cnn.nk",
    "deep_mlp.nk",
    "mnist_mlp.nk",
    "mnist_cnn.nk",
    "fashion_mnist_mlp.nk",
    "fashion_mnist_cnn.nk",
]


def netkit_conv_to_onnx(
    w_flat: np.ndarray, out_c: int, in_c: int, kh: int, kw: int
) -> np.ndarray:
    """Netkit [O, Kh, Kw, I] -> ONNX [O, I, Kh, Kw]."""
    w = w_flat.reshape(out_c, kh, kw, in_c)
    return np.transpose(w, (0, 3, 1, 2)).copy()


def netkit_depthwise_conv_to_onnx(w_flat: np.ndarray, channels: int, kh: int, kw: int) -> np.ndarray:
    """Netkit [C, Kh, Kw] -> ONNX depthwise [C, 1, Kh, Kw]."""
    w = w_flat.reshape(channels, kh, kw)
    return w.reshape(channels, 1, kh, kw).astype(np.float32)


def _spatial_out(size: int, kernel: int, stride: int, pad: int = 0) -> int:
    return (size + 2 * pad - kernel) // stride + 1


def append_activation(
    nodes: list,
    tensor_in: str,
    activation: str,
    is_final: bool,
    *,
    alpha: float = 0.01,
    initializers: list | None = None,
) -> str:
    if activation == "relu":
        tensor_out = "output" if is_final else f"{tensor_in}_relu"
        nodes.append(helper.make_node("Relu", [tensor_in], [tensor_out]))
        return tensor_out
    if activation == "softmax":
        tensor_out = "output" if is_final else f"{tensor_in}_softmax"
        nodes.append(
            helper.make_node("Softmax", [tensor_in], [tensor_out], axis=1)
        )
        return tensor_out
    if activation == "sigmoid":
        tensor_out = "output" if is_final else f"{tensor_in}_sigmoid"
        nodes.append(helper.make_node("Sigmoid", [tensor_in], [tensor_out]))
        return tensor_out
    if activation == "tanh":
        tensor_out = "output" if is_final else f"{tensor_in}_tanh"
        nodes.append(helper.make_node("Tanh", [tensor_in], [tensor_out]))
        return tensor_out
    if activation == "leaky_relu":
        tensor_out = "output" if is_final else f"{tensor_in}_leaky"
        nodes.append(
            helper.make_node("LeakyRelu", [tensor_in], [tensor_out], alpha=float(alpha))
        )
        return tensor_out
    if activation == "relu6":
        tensor_out = "output" if is_final else f"{tensor_in}_relu6"
        relu_out = f"{tensor_in}_relu6_pre"
        min_name = f"{tensor_in}_relu6_min"
        max_name = f"{tensor_in}_relu6_max"
        if initializers is not None:
            initializers.append(numpy_helper.from_array(np.array(0.0, dtype=np.float32), min_name))
            initializers.append(numpy_helper.from_array(np.array(6.0, dtype=np.float32), max_name))
            nodes.append(helper.make_node("Relu", [tensor_in], [relu_out]))
            nodes.append(helper.make_node("Clip", [relu_out, min_name, max_name], [tensor_out]))
        else:
            nodes.append(helper.make_node("Relu", [tensor_in], [relu_out]))
            nodes.append(helper.make_node("Clip", [relu_out], [tensor_out], min=0.0, max=6.0))
        return tensor_out
    return "output" if is_final else tensor_in


def export_mlp(arch: dict, weights: np.ndarray, graph_name: str) -> onnx.ModelProto:
    batch, in_features = arch["input"]
    nodes: list = []
    initializers: list = []
    offset = 0
    tensor = "input"
    layers = arch["layers"]

    for idx, layer in enumerate(layers):
        out_features = layer["units"]
        w_size = in_features * out_features
        w = weights[offset : offset + w_size].reshape(out_features, in_features)
        offset += w_size
        b = weights[offset : offset + out_features]
        offset += out_features

        w_name = f"W{idx}"
        b_name = f"B{idx}"
        initializers.append(numpy_helper.from_array(w.astype(np.float32), w_name))
        initializers.append(numpy_helper.from_array(b.astype(np.float32), b_name))

        is_final = idx == len(layers) - 1
        gemm_out = "output" if is_final and layer.get("activation", "none") == "none" else f"h{idx}"
        nodes.append(
            helper.make_node(
                "Gemm",
                [tensor, w_name, b_name],
                [gemm_out],
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=1,
            )
        )
        tensor = append_activation(
            nodes, gemm_out, layer.get("activation", "none"), is_final,
            alpha=float(layer.get("alpha", 0.01)), initializers=initializers,
        )
        in_features = out_features

    graph = helper.make_graph(
        nodes,
        graph_name,
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [batch, arch["input"][1]]
            )
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, layers[-1]["units"]])],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def export_cnn(arch: dict, weights: np.ndarray, graph_name: str) -> onnx.ModelProto:
    input_h, input_w, channels = arch["input"]
    nodes: list = []
    initializers: list = []
    offset = 0
    tensor = "input"
    spatial_h, spatial_w = input_h, input_w
    dense_in = 0
    layer_idx = 0
    layers = arch["layers"]

    for idx, layer in enumerate(layers):
        layer_type = layer["type"]
        is_final = idx == len(layers) - 1

        if layer_type == "conv2d":
            kernel = layer["kernel_size"]
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            out_c = layer["filters"]
            in_c = channels
            kernel_elems = kernel * kernel * in_c
            w_flat = weights[offset : offset + kernel_elems * out_c]
            offset += kernel_elems * out_c
            b = weights[offset : offset + out_c]
            offset += out_c

            w_name = f"conv{layer_idx}_W"
            b_name = f"conv{layer_idx}_B"
            w_onnx = netkit_conv_to_onnx(w_flat, out_c, in_c, kernel, kernel)
            initializers.append(
                numpy_helper.from_array(w_onnx.astype(np.float32), w_name)
            )
            initializers.append(numpy_helper.from_array(b.astype(np.float32), b_name))

            gemm_out = "output" if is_final and layer.get("activation", "none") == "none" else f"conv{layer_idx}"
            nodes.append(
                helper.make_node(
                    "Conv",
                    [tensor, w_name, b_name],
                    [gemm_out],
                    kernel_shape=[kernel, kernel],
                    strides=[stride, stride],
                    pads=[pad_h, pad_w, pad_h, pad_w],
                )
            )
            tensor = append_activation(
                nodes,
                gemm_out,
                layer.get("activation", "none"),
                False,
                alpha=float(layer.get("alpha", 0.01)),
                initializers=initializers,
            )
            spatial_h = _spatial_out(spatial_h, kernel, stride, pad_h)
            spatial_w = _spatial_out(spatial_w, kernel, stride, pad_w)
            channels = out_c
            layer_idx += 1
            continue

        if layer_type == "depthwise_conv2d":
            kh, kw = depthwise_kernel_hw(layer)
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            ch = layer["filters"]
            kernel_elems = kh * kw * ch
            w_flat = weights[offset : offset + kernel_elems]
            offset += kernel_elems
            b = weights[offset : offset + ch]
            offset += ch

            w_name = f"dwconv{layer_idx}_W"
            b_name = f"dwconv{layer_idx}_B"
            w_onnx = netkit_depthwise_conv_to_onnx(w_flat, ch, kh, kw)
            initializers.append(numpy_helper.from_array(w_onnx, w_name))
            initializers.append(numpy_helper.from_array(b.astype(np.float32), b_name))

            gemm_out = "output" if is_final and layer.get("activation", "none") == "none" else f"dwconv{layer_idx}"
            nodes.append(
                helper.make_node(
                    "Conv",
                    [tensor, w_name, b_name],
                    [gemm_out],
                    kernel_shape=[kh, kw],
                    strides=[stride, stride],
                    pads=[pad_h, pad_w, pad_h, pad_w],
                    group=ch,
                )
            )
            tensor = append_activation(
                nodes,
                gemm_out,
                layer.get("activation", "none"),
                False,
                alpha=float(layer.get("alpha", 0.01)),
                initializers=initializers,
            )
            spatial_h = _spatial_out(spatial_h, kh, stride, pad_h)
            spatial_w = _spatial_out(spatial_w, kw, stride, pad_w)
            layer_idx += 1
            continue

        if layer_type == "max_pool2d":
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            pool_out = f"pool{layer_idx}"
            nodes.append(
                helper.make_node(
                    "MaxPool",
                    [tensor],
                    [pool_out],
                    kernel_shape=[pool, pool],
                    strides=[stride, stride],
                )
            )
            tensor = pool_out
            spatial_h = (spatial_h - pool) // stride + 1
            spatial_w = (spatial_w - pool) // stride + 1
            layer_idx += 1
            continue

        if layer_type == "flatten":
            # netkit flattens NHWC (spatial then channels); ONNX Conv tensors are NCHW.
            nhwc_out = f"nhwc{layer_idx}"
            nodes.append(
                helper.make_node("Transpose", [tensor], [nhwc_out], perm=[0, 2, 3, 1])
            )
            flat_out = f"flat{layer_idx}"
            nodes.append(helper.make_node("Flatten", [nhwc_out], [flat_out], axis=1))
            tensor = flat_out
            dense_in = spatial_h * spatial_w * channels
            layer_idx += 1
            continue

        if layer_type == "dense":
            out_features = layer["units"]
            in_features = dense_in
            w_size = in_features * out_features
            w = weights[offset : offset + w_size].reshape(out_features, in_features)
            offset += w_size
            b = weights[offset : offset + out_features]
            offset += out_features

            w_name = f"dense{layer_idx}_W"
            b_name = f"dense{layer_idx}_B"
            initializers.append(numpy_helper.from_array(w.astype(np.float32), w_name))
            initializers.append(numpy_helper.from_array(b.astype(np.float32), b_name))

            gemm_out = (
                "output"
                if is_final and layer.get("activation", "none") == "none"
                else f"dense{layer_idx}"
            )
            nodes.append(
                helper.make_node(
                    "Gemm",
                    [tensor, w_name, b_name],
                    [gemm_out],
                    alpha=1.0,
                    beta=1.0,
                    transA=0,
                    transB=1,
                )
            )
            tensor = append_activation(
                nodes,
                gemm_out,
                layer.get("activation", "none"),
                is_final,
                alpha=float(layer.get("alpha", 0.01)),
                initializers=initializers,
            )
            dense_in = out_features
            layer_idx += 1
            continue

        raise ValueError(f"Unsupported layer type: {layer_type}")

    if offset != len(weights):
        raise ValueError(
            f"{graph_name}: weight count mismatch (used {offset}, file has {len(weights)})"
        )

    if layers[-1]["type"] == "dense":
        out_dim = layers[-1]["units"]
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, out_dim]
        )
    else:
        out_dim = spatial_h * spatial_w * channels
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, out_dim]
        )

    graph = helper.make_graph(
        nodes,
        graph_name,
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [1, arch["input"][2], input_h, input_w]
            )
        ],
        [output_info],
        initializer=initializers,
    )

    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def expected_weight_floats(arch: dict) -> int:
    network = arch["network"]
    if network == "mlp":
        total = 0
        in_features = arch["input"][1]
        for layer in arch["layers"]:
            out_features = layer["units"]
            total += in_features * out_features + out_features
            in_features = out_features
        return total

    total = 0
    h, w, channels = arch["input"]
    dense_in = 0
    for layer in arch["layers"]:
        if layer["type"] == "conv2d":
            k = layer["kernel_size"]
            f = layer["filters"]
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            total += k * k * channels * f + f
            stride = layer.get("stride", 1)
            h = _spatial_out(h, k, stride, pad_h)
            w = _spatial_out(w, k, stride, pad_w)
            channels = f
        elif layer["type"] == "depthwise_conv2d":
            kh, kw = depthwise_kernel_hw(layer)
            ch = layer["filters"]
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            total += kh * kw * ch + ch
            stride = layer.get("stride", 1)
            h = _spatial_out(h, kh, stride, pad_h)
            w = _spatial_out(w, kw, stride, pad_w)
        elif layer["type"] == "max_pool2d":
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            h = (h - pool) // stride + 1
            w = (w - pool) // stride + 1
        elif layer["type"] == "flatten":
            dense_in = h * w * channels
        elif layer["type"] == "dense":
            out_features = layer["units"]
            total += dense_in * out_features + out_features
            dense_in = out_features
    return total


def export_netkit_nk(nk_path: Path) -> None:
    arch, weights = read_nk(nk_path)
    expected = expected_weight_floats(arch)
    if len(weights) != expected:
        raise ValueError(
            f"{nk_path.name}: expected {expected} floats, got {len(weights)}"
        )

    graph_name = nk_path.stem
    if arch["network"] == "mlp":
        model = export_mlp(arch, weights, graph_name)
    elif arch["network"] == "cnn":
        model = export_cnn(arch, weights, graph_name)
    else:
        raise ValueError(f"Unsupported network: {arch['network']}")

    out_path = nk_path.with_suffix(".onnx")
    onnx.save(model, out_path)
    print(f"Wrote {out_path} ({len(weights)} weights)")


def main() -> None:
    for name in MODEL_NKS:
        nk_path = MODELS / name
        if not nk_path.is_file():
            print(f"Skipping missing {nk_path.name}")
            continue
        export_netkit_nk(nk_path)


if __name__ == "__main__":
    main()
