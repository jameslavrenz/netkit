"""Tests for PyTorch backbone packing."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
    from torchvision.models import resnet18
except ImportError:
    torch = None

from netkit.reference_forward import forward_cnn
from netkit.torch_backbone_pack import pack_resnet18_from_torch
from netkit.torch_pack import assert_packed_matches_reference


@unittest.skipIf(torch is None, 'torch/torchvision required (pip install -e "python[train]" torchvision)')
class TorchBackbonePackTests(unittest.TestCase):
    def test_resnet18_packed_matches_reference(self) -> None:
        model = resnet18(weights=None)
        model.eval()
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        arch, weights = pack_resnet18_from_torch(model, height=56, width=56, num_classes=10)
        self.assertEqual(len(arch["layers"]), 13)
        self.assertEqual(sum(1 for layer in arch["layers"] if layer["type"] == "resnet_basic_block"), 8)

        def torch_forward(inp: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(inp.reshape(1, 56, 56, 3).transpose(0, 3, 1, 2).copy())
            with torch.no_grad():
                logits = model(x)
            return logits.cpu().numpy().reshape(-1)

        assert_packed_matches_reference(arch, weights, torch_forward, seed=7, atol=1e-4)
        inp = np.random.default_rng(1).standard_normal(56 * 56 * 3, dtype=np.float32)
        out = forward_cnn(inp, arch, weights)
        self.assertEqual(len(out), 10)


if __name__ == "__main__":
    unittest.main()
