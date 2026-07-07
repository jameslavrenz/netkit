"""Tests for int8 CNN quantization."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from netkit.format import FLAG_HAS_QUANT, HEADER_BYTES, unpack_header
from netkit.quantize import forward_quantized_cnn, quantize_cnn, quantized_cnn_to_spec
from netkit.writer import RegressionCase, RegressionSuite, write_nk_bytes

ARCH = {
    "network": "cnn",
    "input": [4, 4, 1],
    "layers": [
        {"type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 2, "activation": "relu"},
        {"type": "max_pool2d", "pool_size": 2, "stride": 2},
        {"type": "flatten"},
        {"type": "dense", "units": 3, "activation": "softmax"},
    ],
}


def _tiny_cnn_weights() -> np.ndarray:
    conv_w = np.linspace(-0.5, 0.5, 18, dtype=np.float32).reshape(2, 3, 3, 1)
    conv_b = np.array([0.01, -0.02], dtype=np.float32)
    fc_w = np.array([[0.3, -0.1], [-0.2, 0.4], [0.1, -0.1]], dtype=np.float32)
    fc_b = np.array([0.0, 0.01, -0.01], dtype=np.float32)
    return np.concatenate([conv_w.reshape(-1), conv_b, fc_w.reshape(-1), fc_b]).astype(np.float32)


class TestQuantizeCnn(unittest.TestCase):
    def test_conv_weight_ohwi_cmsis_layout(self) -> None:
        """Conv weights are stored [O,Kh,Kw,I] — same linear order CMSIS-NN expects."""
        weights = _tiny_cnn_weights()
        cal = np.random.default_rng(1).uniform(0.0, 1.0, (8, 16)).astype(np.float32)
        pack = quantize_cnn(ARCH, weights, cal)

        packed = pack.weight_tensors[0].reshape(2, 3, 3, 1)
        flat = pack.weight_tensors[0].reshape(-1)
        for o in range(2):
            for kh in range(3):
                for kw in range(3):
                    for ic in range(1):
                        cmsis_index = ((o * 3 + kh) * 3 + kw) * 1 + ic
                        self.assertEqual(int(flat[cmsis_index]), int(packed[o, kh, kw, ic]))

    def test_round_trip_nk_v4(self) -> None:
        weights = _tiny_cnn_weights()
        cal = np.random.default_rng(0).uniform(0.0, 1.0, (16, 16)).astype(np.float32)
        pack = quantize_cnn(ARCH, weights, cal)
        spec = quantized_cnn_to_spec(ARCH, pack)
        spec.tests = RegressionSuite(
            tolerance=0.08,
            cases=[
                RegressionCase(
                    name="tiny cnn quant",
                    input=cal[0],
                    expected=forward_quantized_cnn(cal[0], ARCH, pack, output_float=True),
                )
            ],
        )
        blob = write_nk_bytes(spec)

        header = unpack_header(blob[:HEADER_BYTES])
        self.assertEqual(header["version"], 4)
        self.assertTrue(header["flags"] & FLAG_HAS_QUANT)

        if os.environ.get("NETKIT_FAST_TESTS") == "1":
            # Host desktop builds lack CMSIS-NN quant runtime (see docs/DATATYPES.md).
            # Python forward_quantized_cnn above validates pack math; MCU / test-full covers runtime.
            return

        with tempfile.TemporaryDirectory() as tmp:
            nk_path = Path(tmp) / "tiny_cnn_int8.nk"
            nk_path.write_bytes(blob)
            proc = subprocess.run(
                ["./netkit", "test", str(nk_path.resolve())],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
