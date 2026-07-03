"""Depthwise conv ONNX import and reference parity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from netkit.arch_writer import write_nk_from_arch
from netkit.onnx_convert import convert_onnx_to_nk
from netkit.reader import read_nk
from netkit.reference_forward import forward_cnn


class DepthwiseConvOnnxTests(unittest.TestCase):
    def test_depthwise_conv_import_and_forward(self) -> None:
        weight = np.random.randn(4, 1, 3, 3).astype(np.float32) * 0.1
        bias = np.random.randn(4).astype(np.float32) * 0.05
        x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])
        y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])
        w = numpy_helper.from_array(weight, name="w")
        b = numpy_helper.from_array(bias, name="b")
        conv = helper.make_node(
            "Conv",
            inputs=["input", "w", "b"],
            outputs=["output"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],
            group=4,
        )
        graph = helper.make_graph([conv], "dw", [x], [y], [w, b])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "dw.onnx"
            nk_path = Path(tmp) / "dw.nk"
            onnx.save(model, onnx_path)
            convert_onnx_to_nk(onnx_path, nk_path)

            arch, weights = read_nk(nk_path)
            layer = arch["layers"][0]
            self.assertEqual(layer["type"], "depthwise_conv2d")
            self.assertEqual(layer["kernel_h"], 3)
            self.assertEqual(layer["kernel_w"], 3)
            inp = np.random.randn(8 * 8 * 4).astype(np.float32) * 0.2
            out = forward_cnn(inp, arch, weights)
            self.assertEqual(len(out), 8 * 8 * 4)

    def test_asymmetric_depthwise_kernel(self) -> None:
        arch = {
            "network": "cnn",
            "input": [32, 1, 4],
            "layers": [
                {
                    "type": "depthwise_conv2d",
                    "kernel_h": 5,
                    "kernel_w": 1,
                    "stride": 1,
                    "filters": 4,
                    "pad_h": 2,
                    "pad_w": 0,
                    "activation": "none",
                },
                {"type": "flatten"},
                {"type": "dense", "units": 2, "activation": "none"},
            ],
        }

        rng = np.random.default_rng(0)
        dw_w = rng.standard_normal((4, 5, 1), dtype=np.float32) * 0.1
        dw_b = rng.standard_normal(4, dtype=np.float32) * 0.05
        dense_w = rng.standard_normal((2, 32 * 4), dtype=np.float32) * 0.1
        dense_b = rng.standard_normal(2, dtype=np.float32) * 0.05
        flat_weights = np.concatenate(
            [dw_w.reshape(-1), dw_b, dense_w.reshape(-1), dense_b]
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            nk_path = Path(tmp) / "dw1d.nk"
            write_nk_from_arch(arch, flat_weights, nk_path)
            loaded_arch, loaded_weights = read_nk(nk_path)
            self.assertEqual(loaded_arch["layers"][0]["kernel_h"], 5)
            self.assertEqual(loaded_arch["layers"][0]["kernel_w"], 1)

            inp = rng.standard_normal(32 * 4, dtype=np.float32) * 0.2
            out = forward_cnn(inp, loaded_arch, loaded_weights)
            self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
