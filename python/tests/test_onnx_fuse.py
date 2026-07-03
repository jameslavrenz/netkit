"""Tests for ONNX composite block fusion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import onnx
    import torch
    from torchvision.models import resnet18
except ImportError:
    onnx = None
    torch = None

from netkit.onnx_convert import onnx_to_spec


@unittest.skipIf(torch is None or onnx is None, "torch/onnx required")
class OnnxFuseTests(unittest.TestCase):
    def test_resnet18_onnx_fuses_basic_blocks(self) -> None:
        model = resnet18(weights=None)
        model.eval()
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        dummy = torch.randn(1, 3, 56, 56)
        onnx_path = Path(tempfile.gettempdir()) / "netkit_resnet18_fuse_test.onnx"
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
        )

        spec = onnx_to_spec(onnx_path, fuse_composite=True)
        block_layers = [layer for layer in spec.layers if layer.kind == "resnet_basic_block"]
        self.assertEqual(len(block_layers), 8)
        self.assertGreater(len(spec.layers), 0)


if __name__ == "__main__":
    unittest.main()
