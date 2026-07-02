"""Tests that PyTorch weight packing matches netkit reference forward."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from netkit.torch_models import TutorialCnn28, TutorialMlp784
from netkit.torch_pack import (
    assert_packed_matches_reference,
    forward_cnn_netkit,
    forward_mlp_netkit,
    pack_conv2d,
    pack_dense,
    pack_tutorial_cnn,
    pack_tutorial_mlp,
)


@unittest.skipIf(torch is None, "torch required (pip install -e \"python[train]\")")
class TestTorchPack(unittest.TestCase):
    def test_dense_layout_matches_matmul(self) -> None:
        linear = torch.nn.Linear(4, 3)
        torch.manual_seed(1)
        linear.weight.data = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        linear.bias.data = torch.tensor([1.0, 2.0, 3.0])

        w, b = pack_dense(linear)
        x = np.array([1.0, 0.5, -1.0, 2.0], dtype=np.float32)
        expected = x @ w.T + b
        actual = linear(torch.from_numpy(x)).detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_conv_layout_matches_reference_indexing(self) -> None:
        conv = torch.nn.Conv2d(2, 1, kernel_size=2, bias=True)
        torch.manual_seed(2)
        conv.weight.data = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
        conv.bias.data = torch.tensor([0.5])

        w_nk, b = pack_conv2d(conv)
        self.assertEqual(w_nk.shape, (1, 2, 2, 2))
        self.assertEqual(tuple(w_nk[0, 0, 0, :]), tuple(conv.weight[0, :, 0, 0].detach().numpy()))
        self.assertEqual(float(b[0]), 0.5)

    def test_tutorial_mlp_pack_matches_reference(self) -> None:
        torch.manual_seed(3)
        model = TutorialMlp784()
        arch = {
            "network": "mlp",
            "input": [1, 784],
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 10, "activation": "softmax"},
            ],
        }
        weights = pack_tutorial_mlp(model)
        assert_packed_matches_reference(
            arch,
            weights,
            lambda inp: forward_mlp_netkit(model, inp),
            samples=4,
            seed=4,
        )

    def test_tutorial_cnn_pack_matches_reference(self) -> None:
        torch.manual_seed(5)
        model = TutorialCnn28()
        arch = {
            "network": "cnn",
            "input": [28, 28, 1],
            "layers": [
                {"type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 32, "activation": "relu"},
                {"type": "max_pool2d", "pool_size": 2, "stride": 2},
                {"type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 64, "activation": "relu"},
                {"type": "max_pool2d", "pool_size": 2, "stride": 2},
                {"type": "flatten"},
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 10, "activation": "softmax"},
            ],
        }
        weights = pack_tutorial_cnn(model)
        assert_packed_matches_reference(
            arch,
            weights,
            lambda inp: forward_cnn_netkit(model, inp),
            samples=2,
            seed=6,
        )


if __name__ == "__main__":
    unittest.main()
