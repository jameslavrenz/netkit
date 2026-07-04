"""Packager-side composite fusion from linear .nk layer stacks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from netkit.nk_fuse import (
    FuseOptions,
    expand_convnextv2_block_to_linear,
    expand_mobilenetv4_uib_to_linear,
    expand_resnet_basic_block_to_linear,
    fuse_composite_blocks,
)
from netkit.nk_optimize import OptimizeOptions, optimize_nk
from netkit.reader import read_nk, read_test_suite
from netkit.reference_forward import forward_cnn


class TestNkFuse(unittest.TestCase):
    def test_uib_roundtrip_from_fixture(self) -> None:
        arch, weights = read_nk("models/mobilenetv4_small_uib.nk")
        self.assertEqual(arch["layers"][0]["type"], "mobilenetv4_uib")

        linear_arch, linear_weights = expand_mobilenetv4_uib_to_linear(arch, weights)
        self.assertGreater(len(linear_arch["layers"]), 1)
        self.assertTrue(all(layer["type"] != "mobilenetv4_uib" for layer in linear_arch["layers"]))

        suite = read_test_suite("models/mobilenetv4_small_uib.nk")
        assert suite is not None
        suite_inp = np.asarray(suite.cases[0].input, dtype=np.float32)
        expected = forward_cnn(suite_inp, arch, weights)
        fused = fuse_composite_blocks(linear_arch, linear_weights, verify_output=False)
        self.assertIn("mobilenetv4_uib", fused.applied)
        self.assertEqual(len(fused.arch["layers"]), 1)
        self.assertEqual(fused.arch["layers"][0]["type"], "mobilenetv4_uib")
        actual = forward_cnn(suite_inp, fused.arch, fused.weights)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)

    def test_optimize_can_fuse_uib(self) -> None:
        arch, weights = read_nk("models/mobilenetv4_small_uib.nk")
        linear_arch, linear_weights = expand_mobilenetv4_uib_to_linear(arch, weights)
        result = optimize_nk(
            linear_arch,
            linear_weights,
            options=OptimizeOptions(
                fold_conv_batch_norm=False,
                fold_batch_norm_into_dense=False,
                merge_linear_dense=False,
                remove_identity_batch_norm=False,
                fuse_composite=True,
            ),
        )
        self.assertIn("mobilenetv4_uib", result.applied)
        self.assertEqual(result.arch["layers"][0]["type"], "mobilenetv4_uib")

    def test_fuse_disabled_is_noop(self) -> None:
        arch, weights = read_nk("models/mobilenetv4_small_uib.nk")
        linear_arch, linear_weights = expand_mobilenetv4_uib_to_linear(arch, weights)
        result = fuse_composite_blocks(
            linear_arch,
            linear_weights,
            options=FuseOptions(mobilenetv4_uib=False),
        )
        self.assertEqual(result.applied, [])
        self.assertEqual(len(result.arch["layers"]), len(linear_arch["layers"]))

    def test_resnet_basic_block_roundtrip_from_fixture(self) -> None:
        arch, weights = read_nk("models/resnet18_basic_block.nk")
        self.assertEqual(arch["layers"][0]["type"], "resnet_basic_block")

        linear_arch, linear_weights = expand_resnet_basic_block_to_linear(arch, weights)
        self.assertEqual(len(linear_arch["layers"]), 4)
        self.assertTrue(all(layer["type"] != "resnet_basic_block" for layer in linear_arch["layers"]))

        suite = read_test_suite("models/resnet18_basic_block.nk")
        assert suite is not None
        suite_inp = np.asarray(suite.cases[0].input, dtype=np.float32)
        expected = forward_cnn(suite_inp, arch, weights)
        fused = fuse_composite_blocks(linear_arch, linear_weights, verify_output=False)
        self.assertIn("resnet_basic_block", fused.applied)
        self.assertEqual(len(fused.arch["layers"]), 1)
        self.assertEqual(fused.arch["layers"][0]["type"], "resnet_basic_block")
        actual = forward_cnn(suite_inp, fused.arch, fused.weights)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)

    def test_convnextv2_block_roundtrip_from_fixture(self) -> None:
        arch, weights = read_nk("models/convnextv2_atto_block.nk")
        self.assertEqual(arch["layers"][0]["type"], "convnextv2_block")

        linear_arch, linear_weights = expand_convnextv2_block_to_linear(arch, weights)
        self.assertEqual(len(linear_arch["layers"]), 5)
        self.assertTrue(all(layer["type"] != "convnextv2_block" for layer in linear_arch["layers"]))

        suite = read_test_suite("models/convnextv2_atto_block.nk")
        assert suite is not None
        suite_inp = np.asarray(suite.cases[0].input, dtype=np.float32)
        expected = forward_cnn(suite_inp, arch, weights)
        fused = fuse_composite_blocks(linear_arch, linear_weights, verify_output=False)
        self.assertIn("convnextv2_block", fused.applied)
        self.assertEqual(len(fused.arch["layers"]), 1)
        self.assertEqual(fused.arch["layers"][0]["type"], "convnextv2_block")
        actual = forward_cnn(suite_inp, fused.arch, fused.weights)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)

    def test_optimize_can_fuse_resnet_and_convnext(self) -> None:
        for path, block_type, expand in (
            ("models/resnet18_basic_block.nk", "resnet_basic_block", expand_resnet_basic_block_to_linear),
            ("models/convnextv2_atto_block.nk", "convnextv2_block", expand_convnextv2_block_to_linear),
        ):
            with self.subTest(path=path):
                arch, weights = read_nk(path)
                linear_arch, linear_weights = expand(arch, weights)
                result = optimize_nk(
                    linear_arch,
                    linear_weights,
                    options=OptimizeOptions(
                        fold_conv_batch_norm=False,
                        fold_batch_norm_into_dense=False,
                        merge_linear_dense=False,
                        remove_identity_batch_norm=False,
                        fuse_composite=True,
                    ),
                )
                self.assertIn(block_type, result.applied)
                self.assertEqual(result.arch["layers"][0]["type"], block_type)

    def test_onnx_resnet_primitive_import_then_packager_fuse(self) -> None:
        try:
            import onnx
            import timm
            import torch
        except ImportError:
            self.skipTest("torch/timm/onnx required")

        from netkit.onnx_convert import convert_onnx_to_nk, onnx_to_spec
        from netkit.reader import read_nk

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
        onnx_path = Path(tempfile.gettempdir()) / "netkit_resnet_packager_fuse.onnx"
        export_kwargs = dict(input_names=["input"], output_names=["output"], opset_version=17)
        try:
            torch.onnx.export(
                OneBlock(block), torch.randn(*x.shape), str(onnx_path), **export_kwargs, dynamo=False
            )
        except TypeError:
            torch.onnx.export(OneBlock(block), torch.randn(*x.shape), str(onnx_path), **export_kwargs)

        prim = onnx_to_spec(onnx_path, fuse_composite=False)
        self.assertEqual([layer.kind for layer in prim.layers], ["conv2d", "batch_norm2d", "conv2d", "batch_norm2d"])

        nk_path = Path(tempfile.gettempdir()) / "netkit_resnet_packager_fuse.nk"
        convert_onnx_to_nk(
            onnx_path,
            nk_path,
            fuse_composite=False,
            optimize=True,
            packager_fuse=True,
        )
        arch, _weights = read_nk(nk_path)
        self.assertEqual(arch["layers"][0]["type"], "resnet_basic_block")


if __name__ == "__main__":
    unittest.main()
