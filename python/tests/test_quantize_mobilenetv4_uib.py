"""Quantized MobileNetV4 UIB round-trip tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from netkit.reader import read_nk
from netkit.quantize import forward_quantized_cnn, quantize_cnn, quantized_cnn_to_spec
from netkit.reference_forward import forward_cnn
from netkit.writer import RegressionCase, RegressionSuite, write_nk

ROOT = Path(__file__).resolve().parents[2]
UIB_FIXTURE = ROOT / "models" / "mobilenetv4_small_uib.nk"


class QuantizeMobilenetV4UibTests(unittest.TestCase):
    def test_uib_round_trip_nk(self) -> None:
        arch, weights = read_nk(UIB_FIXTURE)
        cal = np.random.default_rng(0).standard_normal(
            (4, arch["input"][0] * arch["input"][1] * arch["input"][2]), dtype=np.float32
        ) * 0.1
        pack = quantize_cnn(arch, weights, cal)
        spec = quantized_cnn_to_spec(arch, pack)

        inp = cal[0]
        ref = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
        q_out = forward_quantized_cnn(inp, arch, pack, output_float=True)
        self.assertEqual(q_out.shape, ref.shape)
        np.testing.assert_allclose(q_out, ref, atol=0.5, rtol=0.0)

        out_path = ROOT / "models" / "_tmp_mobilenetv4_uib_int8.nk"
        try:
            spec.tests = None
            write_nk(out_path, spec)
            self.assertTrue(out_path.exists())
        finally:
            if out_path.exists():
                out_path.unlink()


if __name__ == "__main__":
    unittest.main()
