"""Tests for ONNX composite block fusion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import onnx
    import timm
    import torch
except ImportError:
    onnx = None
    timm = None
    torch = None

from netkit.onnx_convert import onnx_to_spec


@unittest.skipIf(torch is None or onnx is None or timm is None, "torch/timm/onnx required")
class OnnxFuseTests(unittest.TestCase):
    def test_resnet18_onnx_fuses_basic_blocks(self) -> None:
        backbone = timm.create_model("resnet18", pretrained=False, num_classes=10)
        backbone.eval()

        class FeaturesOnly(torch.nn.Module):
            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.model.forward_features(x)

        model = FeaturesOnly(backbone)
        dummy = torch.randn(1, 3, 56, 56)
        onnx_path = Path(tempfile.gettempdir()) / "netkit_resnet18_fuse_test.onnx"
        export_kwargs = dict(
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
        )
        try:
            torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs, dynamo=False)
        except TypeError:
            torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)

        spec = onnx_to_spec(onnx_path, fuse_composite=True)
        block_layers = [layer for layer in spec.layers if layer.kind == "resnet_basic_block"]
        self.assertEqual(len(block_layers), 8)
        self.assertGreater(len(spec.layers), 0)


if __name__ == "__main__":
    unittest.main()
