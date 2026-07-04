"""Pack timm backbones to .nk and compare C++ runtime vs PyTorch + NumPy reference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import timm  # noqa: F401
except ImportError:
    torch = None

from netkit.arch_writer import write_nk_from_arch
from netkit.reference_forward import forward_cnn
from netkit.runtime_infer import nk_infer_bin, run_nk_infer
from netkit.torch_backbone_pack import (
    PACK_ARCH_DEFAULTS,
    backbone_torch_forward,
    load_backbone_model,
    pack_backbone_from_torch,
)

ROOT = Path(__file__).resolve().parents[2]

@unittest.skipIf(torch is None, 'torch/timm required (pip install -e "python[train]")')
class TorchBackboneRuntimeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not nk_infer_bin(ROOT).is_file():
            raise unittest.SkipTest("tools/nk_infer missing — run make tools/nk_infer")

    def _assert_runtime_parity(self, arch_name: str, *, samples: int, atol: float) -> None:
        height, width = PACK_ARCH_DEFAULTS[arch_name]
        torch.manual_seed(42)
        model = load_backbone_model(arch_name, num_classes=10)
        arch, weights = pack_backbone_from_torch(
            arch_name, model, height=height, width=width, num_classes=10
        )

        rng = np.random.default_rng(7)
        flat_size = height * width * 3

        with tempfile.TemporaryDirectory(prefix="netkit_backbone_") as tmp:
            nk_path = Path(tmp) / f"{arch_name}.nk"
            write_nk_from_arch(arch, weights, nk_path, tests=None)

            for sample_idx in range(samples):
                inp = rng.uniform(-0.3, 0.3, flat_size).astype(np.float32)
                torch_out = backbone_torch_forward(model, inp, height=height, width=width)
                ref_out = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
                runtime_out = run_nk_infer(nk_path, inp, root=ROOT)

                np.testing.assert_allclose(
                    ref_out,
                    torch_out,
                    rtol=0.0,
                    atol=atol,
                    err_msg=f"{arch_name} sample {sample_idx}: numpy ref vs timm",
                )
                np.testing.assert_allclose(
                    runtime_out,
                    torch_out,
                    rtol=0.0,
                    atol=atol,
                    err_msg=f"{arch_name} sample {sample_idx}: nk_infer vs timm",
                )

    def test_resnet18_runtime_matches_timm(self) -> None:
        self._assert_runtime_parity("resnet18", samples=3, atol=1e-4)

    def test_convnextv2_atto_runtime_matches_timm(self) -> None:
        self._assert_runtime_parity("convnextv2_atto", samples=2, atol=1e-4)

    def test_mobilenetv4_small_runtime_matches_timm(self) -> None:
        self._assert_runtime_parity("mobilenetv4_small", samples=2, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
