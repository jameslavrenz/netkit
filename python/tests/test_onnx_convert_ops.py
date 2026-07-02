"""ONNX convert coverage for padding, avg pool, batch norm, and activation fusion."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from netkit.format import Activation, unpack_header
from netkit.onnx_convert import convert_onnx_to_nk, onnx_to_spec


def _build_padded_conv_relu_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])
    w = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), "w")
    b = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), "b")
    conv = helper.make_node("Conv", ["x", "w", "b"], ["c"], pads=[1, 1, 1, 1], strides=[1, 1])
    relu = helper.make_node("Relu", ["c"], ["y"])
    graph = helper.make_graph([conv, relu], "g", [x], [y], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def _build_avgpool_bn_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])
    avg = helper.make_node("AveragePool", ["x"], ["p"], kernel_shape=[2, 2], strides=[2, 2])
    scale = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), "scale")
    beta = numpy_helper.from_array(np.array([0.1, -0.2], dtype=np.float32), "beta")
    mean = numpy_helper.from_array(np.zeros(2, dtype=np.float32), "mean")
    var = numpy_helper.from_array(np.ones(2, dtype=np.float32), "var")
    bn = helper.make_node(
        "BatchNormalization",
        ["p", "scale", "beta", "mean", "var"],
        ["y"],
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [avg, bn],
        "g",
        [x],
        [y],
        [scale, beta, mean, var],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def _build_padded_maxpool_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])
    pool = helper.make_node(
        "MaxPool",
        ["x"],
        ["y"],
        kernel_shape=[2, 2],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph([pool], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def _build_global_avgpool_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 1, 1])
    gap = helper.make_node("GlobalAveragePool", ["x"], ["y"])
    graph = helper.make_graph([gap], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def test_conv_padding_and_relu_fusion():
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "pad_conv.onnx"
        _build_padded_conv_relu_onnx(onnx_path)
        spec = onnx_to_spec(onnx_path)
        assert spec.layers[0].kind == "conv2d"
        assert spec.layers[0].pad_h == 1
        assert spec.layers[0].pad_w == 1
        assert spec.layers[0].activation == Activation.RELU
        assert len(spec.layers) == 1

        nk_path = convert_onnx_to_nk(onnx_path)
        header = unpack_header(nk_path.read_bytes())
        assert header["num_layers"] == 1
        assert header["version"] == 3


def test_avgpool_and_batch_norm():
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "avg_bn.onnx"
        _build_avgpool_bn_onnx(onnx_path)
        spec = onnx_to_spec(onnx_path)
        assert [layer.kind for layer in spec.layers] == ["avg_pool2d", "batch_norm2d"]
        assert spec.layers[1].channels == 2
        assert len(spec.weight_tensors) == 1
        assert len(spec.bias_tensors) == 1


def test_maxpool_padding():
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "pad_pool.onnx"
        _build_padded_maxpool_onnx(onnx_path)
        spec = onnx_to_spec(onnx_path)
        assert spec.layers[0].kind == "max_pool2d"
        assert spec.layers[0].pad_h == 1
        assert spec.layers[0].pad_w == 1


def test_global_average_pool():
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "gap.onnx"
        _build_global_avgpool_onnx(onnx_path)
        spec = onnx_to_spec(onnx_path)
        assert spec.layers[0].kind == "avg_pool2d"
        assert spec.layers[0].pool_size == 4
        assert spec.layers[0].stride == 1
