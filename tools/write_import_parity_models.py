#!/usr/bin/env python3
"""Build ONNX import-extension fixtures with embedded tests and .nk sidecars.

Creates asymmetric conv, rectangular pool, MatMul MLP, CNN MatMul-head, exported
ResNet BasicBlock, and timm MobileNet UIB models for ONNX Runtime parity.

Run from repo root:
    python3 tools/write_import_parity_models.py
    make export-import-parity
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit import RegressionCase, RegressionSuite, read_nk, write_nk_from_arch
from netkit.onnx_convert import convert_onnx_to_nk
from netkit.reference_forward import forward_cnn, forward_mlp

MODELS = ROOT / "models"
OPSET = [helper.make_opsetid("", 13)]


def _save(model, path: Path) -> None:
    onnx.save(model, str(path))


def _case(name: str, inp: list[float], arch: dict, weights: np.ndarray) -> RegressionCase:
    if arch["network"] == "mlp":
        expected = forward_mlp(np.asarray(inp, dtype=np.float32), arch, weights)
    else:
        expected = forward_cnn(np.asarray(inp, dtype=np.float32), arch, weights)
    return RegressionCase(name=name, input=inp, expected=expected)


def build_asym_conv() -> tuple[Path, Path, RegressionSuite]:
    """Single conv with asymmetric ONNX pads [1,1,2,1] (top, left, bottom, right)."""
    onnx_path = MODELS / "import_asym_conv.onnx"
    nk_path = MODELS / "import_asym_conv.nk"

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 5, 4])
    w = numpy_helper.from_array(
        np.array([[[[1.0, 0.0, -1.0], [0.5, 0.0, -0.5], [1.0, 0.0, -1.0]]]], dtype=np.float32),
        "w",
    )
    b = numpy_helper.from_array(np.array([0.25], dtype=np.float32), "b")
    conv = helper.make_node(
        "Conv",
        ["x", "w", "b"],
        ["y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 2, 1],
    )
    graph = helper.make_graph([conv], "asym_conv", [x], [y], [w, b])
    _save(helper.make_model(graph, opset_imports=OPSET), onnx_path)
    convert_onnx_to_nk(onnx_path, nk_path, optimize=False, packager_fuse=False)

    arch, weights = read_nk(nk_path)
    inputs = [
        ("uniform", [1.0] * 16),
        ("corner impulse", [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]),
        ("checkerboard", [1.0 if (i + j) % 2 == 0 else -0.5 for i in range(4) for j in range(4)]),
    ]
    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[_case(name, list(inp), arch, weights) for name, inp in inputs],
    )
    return onnx_path, nk_path, suite


def build_rect_pool() -> tuple[Path, Path, RegressionSuite]:
    """MaxPool with kernel [2, 3] on 4×6 input."""
    onnx_path = MODELS / "import_rect_pool.onnx"
    nk_path = MODELS / "import_rect_pool.nk"

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3, 4])
    pool = helper.make_node(
        "MaxPool",
        ["x"],
        ["y"],
        kernel_shape=[2, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph([pool], "rect_pool", [x], [y])
    _save(helper.make_model(graph, opset_imports=OPSET), onnx_path)
    convert_onnx_to_nk(onnx_path, nk_path, optimize=False, packager_fuse=False)

    arch, weights = read_nk(nk_path)
    base = np.arange(1.0, 25.0, dtype=np.float32).reshape(4, 6)
    inputs = [
        ("ramp", base.reshape(-1).tolist()),
        ("uniform", [0.5] * 24),
        ("sparse peaks", [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]),
    ]
    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[_case(name, list(inp), arch, weights) for name, inp in inputs],
    )
    return onnx_path, nk_path, suite


def build_matmul_mlp() -> tuple[Path, Path, RegressionSuite]:
    """Two-layer MLP using MatMul + Add instead of Gemm."""
    onnx_path = MODELS / "import_matmul_mlp.onnx"
    nk_path = MODELS / "import_matmul_mlp.nk"

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    w1 = numpy_helper.from_array(
        np.array([[0.2, -0.1, 0.3], [0.5, 0.0, -0.4], [0.1, 0.2, 0.1]], dtype=np.float32),
        "w1",
    )
    b1 = numpy_helper.from_array(np.array([0.05, -0.1, 0.2], dtype=np.float32), "b1")
    w2 = numpy_helper.from_array(
        np.array([[0.6, -0.2], [-0.3, 0.4], [0.1, 0.2]], dtype=np.float32),
        "w2",
    )
    b2 = numpy_helper.from_array(np.array([0.1, -0.05], dtype=np.float32), "b2")
    mm1 = helper.make_node("MatMul", ["x", "w1"], ["h1"])
    add1 = helper.make_node("Add", ["h1", "b1"], ["a1"])
    relu = helper.make_node("Relu", ["a1"], ["r1"])
    mm2 = helper.make_node("MatMul", ["r1", "w2"], ["h2"])
    add2 = helper.make_node("Add", ["h2", "b2"], ["y"])
    graph = helper.make_graph([mm1, add1, relu, mm2, add2], "matmul_mlp", [x], [y], [w1, b1, w2, b2])
    _save(helper.make_model(graph, opset_imports=OPSET), onnx_path)
    convert_onnx_to_nk(onnx_path, nk_path, optimize=False, packager_fuse=False)

    arch, weights = read_nk(nk_path)
    inputs = [
        ("mixed", [1.0, -0.5, 2.0]),
        ("zero", [0.0, 0.0, 0.0]),
        ("relu gate", [-2.0, 0.5, 1.0]),
    ]
    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[_case(name, list(inp), arch, weights) for name, inp in inputs],
    )
    return onnx_path, nk_path, suite


def build_cnn_matmul_head() -> tuple[Path, Path, RegressionSuite]:
    """Conv → Flatten → Reshape → MatMul → Add (common classifier export pattern)."""
    onnx_path = MODELS / "import_cnn_matmul_head.onnx"
    nk_path = MODELS / "import_cnn_matmul_head.nk"

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    w_conv = numpy_helper.from_array(
        np.array(
            [
                [[[1.0, 0.0, -1.0], [0.0, 0.5, 0.0], [-1.0, 0.0, 1.0]]],
                [[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]],
            ],
            dtype=np.float32,
        ),
        "w_conv",
    )
    b_conv = numpy_helper.from_array(np.array([0.0, 0.1], dtype=np.float32), "b_conv")
    conv = helper.make_node(
        "Conv",
        ["x", "w_conv", "b_conv"],
        ["conv"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
    )
    nhwc = helper.make_node("Transpose", ["conv"], ["nhwc"], perm=[0, 2, 3, 1])
    flat = helper.make_node("Flatten", ["nhwc"], ["flat"], axis=1)
    shape = numpy_helper.from_array(np.array([1, 32], dtype=np.int64), "shape")
    reshape = helper.make_node("Reshape", ["flat", "shape"], ["rs"])
    w_head = numpy_helper.from_array(
        np.arange(64, dtype=np.float32).reshape(32, 2) * 0.01,
        "w_head",
    )
    b_head = numpy_helper.from_array(np.array([0.2, -0.1], dtype=np.float32), "b_head")
    mm = helper.make_node("MatMul", ["rs", "w_head"], ["mm"])
    add = helper.make_node("Add", ["mm", "b_head"], ["y"])
    graph = helper.make_graph(
        [conv, nhwc, flat, reshape, mm, add],
        "cnn_matmul_head",
        [x],
        [y],
        [w_conv, b_conv, shape, w_head, b_head],
    )
    _save(helper.make_model(graph, opset_imports=OPSET), onnx_path)
    convert_onnx_to_nk(onnx_path, nk_path, optimize=False, packager_fuse=False)

    arch, weights = read_nk(nk_path)
    inputs = [
        ("uniform", [1.0] * 16),
        ("gradient", [float(i) for i in range(16)]),
        ("center spike", [0.0] * 5 + [4.0] + [0.0] * 10),
    ]
    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[_case(name, list(inp), arch, weights) for name, inp in inputs],
    )
    return onnx_path, nk_path, suite


def _export_torch_onnx(model, dummy, path: Path) -> None:
    import torch

    export_kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    try:
        torch.onnx.export(model, dummy, str(path), **export_kwargs, dynamo=False)
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **export_kwargs)


def build_asym_depthwise_conv() -> tuple[Path, Path, RegressionSuite]:
    """Depthwise conv with asymmetric ONNX pads [1,1,2,1] on 3×3 kernel."""
    onnx_path = MODELS / "import_asym_depthwise_conv.onnx"
    nk_path = MODELS / "import_asym_depthwise_conv.nk"

    rng = np.random.default_rng(7)
    weight = rng.standard_normal((2, 1, 3, 3), dtype=np.float32) * 0.1
    bias = rng.standard_normal(2, dtype=np.float32) * 0.05
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 6, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 7, 6])
    w = numpy_helper.from_array(weight, "w")
    b = numpy_helper.from_array(bias, "b")
    conv = helper.make_node(
        "Conv",
        ["x", "w", "b"],
        ["y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 2, 1],
        group=2,
    )
    graph = helper.make_graph([conv], "asym_dw", [x], [y], [w, b])
    _save(helper.make_model(graph, opset_imports=OPSET), onnx_path)
    convert_onnx_to_nk(onnx_path, nk_path, optimize=False, packager_fuse=False)

    arch, weights = read_nk(nk_path)
    assert arch["layers"][0]["type"] == "depthwise_conv2d"
    h, w, c = arch["input"]
    inputs = [
        ("uniform", rng.standard_normal(h * w * c).astype(np.float32) * 0.2),
        ("ramp", np.linspace(-1.0, 1.0, h * w * c, dtype=np.float32)),
        ("corner", np.array([2.0] + [0.0] * (h * w * c - 1), dtype=np.float32)),
    ]
    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[_case(name, inp.tolist(), arch, weights) for name, inp in inputs],
    )
    return onnx_path, nk_path, suite


def build_import_mobilenet_uib_skip() -> tuple[Path, Path, RegressionSuite]:
    """Real timm MobileNetV4 UIB export with residual Add (has_skip)."""
    import torch
    import timm

    onnx_path = MODELS / "import_mobilenet_uib_skip.onnx"
    nk_path = MODELS / "import_mobilenet_uib_skip.nk"

    backbone = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=10)
    backbone.eval()
    target = None
    for stage in backbone.blocks:
        for blk in stage:
            if getattr(blk, "has_skip", False):
                target = blk
                break
        if target is not None:
            break
    assert target is not None

    class OneBlock(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.block = module

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    x = torch.randn(1, 3, 56, 56)
    x = torch.relu(backbone.bn1(backbone.conv_stem(x)))
    for stage_idx in range(len(backbone.blocks)):
        for blk in backbone.blocks[stage_idx]:
            if blk is target:
                dummy = torch.randn(*x.shape)
                break
            x = blk(x)
        else:
            continue
        break

    _export_torch_onnx(OneBlock(target), dummy, onnx_path)
    convert_onnx_to_nk(onnx_path, nk_path, fuse_composite=True, optimize=True, packager_fuse=True)

    arch, weights = read_nk(nk_path)
    assert arch["layers"][0]["type"] == "mobilenetv4_uib"
    rng = np.random.default_rng(42)
    h, w, c = arch["input"]
    inputs = [
        ("uniform", rng.standard_normal(h * w * c).astype(np.float32) * 0.2),
        ("ramp", np.linspace(-0.5, 0.5, h * w * c, dtype=np.float32)),
        ("sparse", np.array([0.0, 1.5, 0.0] + [0.0] * (h * w * c - 3), dtype=np.float32)),
    ]
    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[_case(name, inp.tolist(), arch, weights) for name, inp in inputs],
    )
    return onnx_path, nk_path, suite


def build_import_resnet_basic_block() -> tuple[Path, Path, RegressionSuite]:
    """Real timm ResNet-18 BasicBlock ONNX → primitive import → packager fuse."""
    import torch
    import timm

    onnx_path = MODELS / "import_resnet_basic_block.onnx"
    nk_path = MODELS / "import_resnet_basic_block.nk"

    backbone = timm.create_model("resnet18", pretrained=False, num_classes=10)
    backbone.eval()
    block = backbone.layer1[0]

    class OneBlock(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.block = module

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    x = torch.randn(1, 3, 56, 56)
    x = backbone.act1(backbone.bn1(backbone.conv1(x)))
    x = backbone.maxpool(x)
    dummy = torch.randn(*x.shape)
    _export_torch_onnx(OneBlock(block), dummy, onnx_path)
    convert_onnx_to_nk(
        onnx_path,
        nk_path,
        fuse_composite=False,
        optimize=True,
        packager_fuse=True,
    )

    arch, weights = read_nk(nk_path)
    assert arch["layers"][0]["type"] == "resnet_basic_block"
    rng = np.random.default_rng(42)
    h, w, c = arch["input"]
    inputs = [
        ("uniform", rng.standard_normal(h * w * c).astype(np.float32) * 0.2),
        ("ramp", np.linspace(-1.0, 1.0, h * w * c, dtype=np.float32)),
        ("corner", np.array([3.0] + [0.0] * (h * w * c - 2) + [2.0], dtype=np.float32)),
    ]
    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[_case(name, inp.tolist(), arch, weights) for name, inp in inputs],
    )
    return onnx_path, nk_path, suite


def build_import_mobilenet_uib() -> tuple[Path, Path, RegressionSuite]:
    """Real timm MobileNetV4 UIB export (stride-2, no residual Add)."""
    import torch
    import timm

    onnx_path = MODELS / "import_mobilenet_uib.onnx"
    nk_path = MODELS / "import_mobilenet_uib.nk"

    backbone = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=10)
    backbone.eval()
    target = backbone.blocks[2][0]
    assert not target.has_skip

    class OneBlock(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.block = module

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    x = torch.randn(1, 3, 56, 56)
    x = torch.relu(backbone.bn1(backbone.conv_stem(x)))
    for stage_idx in range(3):
        for blk in backbone.blocks[stage_idx]:
            if blk is target:
                dummy = torch.randn(*x.shape)
                break
            x = blk(x)
        else:
            continue
        break

    _export_torch_onnx(OneBlock(target), dummy, onnx_path)
    convert_onnx_to_nk(onnx_path, nk_path, fuse_composite=False, optimize=True, packager_fuse=True)

    arch, weights = read_nk(nk_path)
    rng = np.random.default_rng(42)
    h, w, c = arch["input"]
    inputs = [
        ("uniform", rng.standard_normal(h * w * c).astype(np.float32) * 0.2),
        ("ramp", np.linspace(-0.5, 0.5, h * w * c, dtype=np.float32)),
        ("sparse", np.array([0.0, 0.0, 1.5] + [0.0] * (h * w * c - 3), dtype=np.float32)),
    ]
    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[_case(name, inp.tolist(), arch, weights) for name, inp in inputs],
    )
    return onnx_path, nk_path, suite


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    builders = [
        build_asym_conv,
        build_asym_depthwise_conv,
        build_rect_pool,
        build_matmul_mlp,
        build_cnn_matmul_head,
        build_import_resnet_basic_block,
        build_import_mobilenet_uib,
        build_import_mobilenet_uib_skip,
    ]
    for builder in builders:
        _onnx, nk_path, suite = builder()
        write_nk_from_arch(*read_nk(nk_path), nk_path, suite)
        print(f"Wrote {nk_path.name} + sidecar ({len(suite.cases)} cases)")


if __name__ == "__main__":
    main()
