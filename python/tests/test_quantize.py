"""Tests for int8 MLP quantization and .nk v4 export."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from netkit.quantize import forward_quantized_mlp, quantize_mlp, quantized_mlp_to_spec
from netkit.format import FLAG_HAS_QUANT, unpack_header, HEADER_BYTES
from netkit.writer import RegressionCase, RegressionSuite, write_nk_bytes

ARCH = {
    "network": "mlp",
    "input": [1, 4],
    "layers": [
        {"type": "dense", "units": 3, "activation": "relu"},
        {"type": "dense", "units": 2, "activation": "softmax"},
    ],
}


def _tiny_float_weights() -> np.ndarray:
    w0 = np.array(
        [
            [0.5, -0.25, 0.1, 0.0],
            [-0.2, 0.3, 0.15, -0.05],
            [0.05, 0.1, -0.3, 0.2],
        ],
        dtype=np.float32,
    )
    b0 = np.array([0.01, -0.02, 0.03], dtype=np.float32)
    w1 = np.array(
        [
            [0.4, -0.1, 0.2],
            [-0.3, 0.25, 0.05],
        ],
        dtype=np.float32,
    )
    b1 = np.array([0.0, 0.01], dtype=np.float32)
    return np.concatenate([w0.reshape(-1), b0, w1.reshape(-1), b1]).astype(np.float32)


class TestQuantizeMlp(unittest.TestCase):
    def test_round_trip_nk_v4(self) -> None:
        weights = _tiny_float_weights()
        cal = np.random.default_rng(0).uniform(-0.5, 0.5, (32, 4)).astype(np.float32)
        pack = quantize_mlp(ARCH, weights, cal)
        spec = quantized_mlp_to_spec(ARCH, pack)
        spec.tests = RegressionSuite(
            tolerance=0.05,
            cases=[
                RegressionCase(
                    name="tiny quant",
                    input=cal[0],
                    expected=forward_quantized_mlp(cal[0], ARCH, pack, output_float=True),
                )
            ],
        )
        blob = write_nk_bytes(spec)

        header = unpack_header(blob[:HEADER_BYTES])
        self.assertEqual(header["version"], 4)
        self.assertTrue(header["flags"] & FLAG_HAS_QUANT)

        if os.environ.get("NETKIT_FAST_TESTS") == "1":
            # Host desktop builds lack CMSIS-NN quant runtime (see docs/DATATYPES.md).
            # Python forward_quantized_mlp above validates pack math; MCU / test-full covers runtime.
            return

        with tempfile.TemporaryDirectory() as tmp:
            nk_path = Path(tmp) / "tiny_int8.nk"
            nk_path.write_bytes(blob)
            proc = subprocess.run(
                ["./netkit", "test", str(nk_path.resolve())],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

    def test_forward_is_finite(self) -> None:
        weights = _tiny_float_weights()
        cal = np.random.default_rng(1).uniform(-0.5, 0.5, (16, 4)).astype(np.float32)
        pack = quantize_mlp(ARCH, weights, cal)
        inp = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
        out = forward_quantized_mlp(inp, ARCH, pack, output_float=True)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertAlmostEqual(float(out.sum()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
