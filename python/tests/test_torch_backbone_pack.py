"""Tests for PyTorch backbone packing."""

from __future__ import annotations

import os
import unittest

import numpy as np

if os.environ.get("NETKIT_FAST_TESTS") == "1":
    raise unittest.SkipTest("skipped in default make test (use make test-full)")

try:
    import timm
    import torch
except ImportError:
    timm = None
    torch = None

from netkit.reference_forward import forward_cnn
from netkit.torch_backbone_pack import (
    backbone_torch_forward,
    load_backbone_model,
    pack_backbone_from_torch,
)
from netkit.torch_pack import assert_packed_matches_reference


@unittest.skipIf(timm is None or torch is None, 'timm required (pip install -e "python[train]")')
class TorchBackbonePackTests(unittest.TestCase):
    def test_resnet18_packed_matches_reference(self) -> None:
        model = load_backbone_model("resnet18", num_classes=10)
        arch, weights = pack_backbone_from_torch(
            "resnet18", model, height=56, width=56, num_classes=10
        )
        self.assertEqual(len(arch["layers"]), 13)
        self.assertEqual(sum(1 for layer in arch["layers"] if layer["type"] == "resnet_basic_block"), 8)

        def torch_forward(inp: np.ndarray) -> np.ndarray:
            return backbone_torch_forward(model, inp, height=56, width=56)

        assert_packed_matches_reference(arch, weights, torch_forward, seed=7, atol=1e-4, samples=3)
        inp = np.random.default_rng(1).standard_normal(56 * 56 * 3, dtype=np.float32)
        out = forward_cnn(inp, arch, weights)
        self.assertEqual(len(out), 10)

    def test_convnextv2_atto_packed_matches_reference(self) -> None:
        model = load_backbone_model("convnextv2_atto", num_classes=10)
        arch, weights = pack_backbone_from_torch(
            "convnextv2_atto", model, height=32, width=32, num_classes=10
        )
        self.assertEqual(len(arch["layers"]), 24)
        self.assertEqual(sum(1 for layer in arch["layers"] if layer["type"] == "convnextv2_block"), 12)

        def torch_forward(inp: np.ndarray) -> np.ndarray:
            return backbone_torch_forward(model, inp, height=32, width=32)

        assert_packed_matches_reference(arch, weights, torch_forward, seed=7, atol=1e-4, samples=2)
        inp = np.random.default_rng(2).standard_normal(32 * 32 * 3, dtype=np.float32)
        out = forward_cnn(inp, arch, weights)
        self.assertEqual(len(out), 10)

    def test_mobilenetv4_small_packed_matches_reference(self) -> None:
        model = load_backbone_model("mobilenetv4_small", num_classes=10)
        arch, weights = pack_backbone_from_torch(
            "mobilenetv4_small", model, height=56, width=56, num_classes=10
        )
        self.assertEqual(len(arch["layers"]), 22)
        self.assertEqual(sum(1 for layer in arch["layers"] if layer["type"] == "mobilenetv4_uib"), 12)

        def torch_forward(inp: np.ndarray) -> np.ndarray:
            return backbone_torch_forward(model, inp, height=56, width=56)

        assert_packed_matches_reference(arch, weights, torch_forward, seed=7, atol=1e-4, samples=3)
        inp = np.random.default_rng(3).standard_normal(56 * 56 * 3, dtype=np.float32)
        out = forward_cnn(inp, arch, weights)
        self.assertEqual(len(out), 10)


if __name__ == "__main__":
    unittest.main()
