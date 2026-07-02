"""Compare .nk runtime output against ONNX Runtime on matching .onnx sidecars."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

# mnist_cnn / fashion_mnist_cnn ONNX sidecars do not match .nk via ONNX Runtime today;
# .nk correctness is covered by C++ embedded TCAS tests.
PARITY_PAIRS = [
    ("test_mlp.nk", "test_mlp.onnx"),
    ("mlp_hand.nk", "mlp_hand.onnx"),
    ("test_cnn.nk", "test_cnn.onnx"),
    ("cnn_4x4_single.nk", "cnn_4x4_single.onnx"),
    ("cnn_hand.nk", "cnn_hand.onnx"),
    ("op_matrix_mlp.nk", "op_matrix_mlp.onnx"),
    ("op_matrix_cnn.nk", "op_matrix_cnn.onnx"),
    ("deep_mlp.nk", "deep_mlp.onnx"),
    ("mnist_mlp.nk", "mnist_mlp.onnx"),
    ("fashion_mnist_mlp.nk", "fashion_mnist_mlp.onnx"),
]

# Full suite when every pair is present (CNN tutorial models excluded above).
EXPECTED_CASES = 49


def _nk_infer_bin() -> Path:
    env = os.environ.get("NK_INFER_BIN")
    if env:
        return Path(env)
    return ROOT / "tools" / "nk_infer"


def _run_nk(nk_path: Path, flat_input: np.ndarray) -> np.ndarray:
    args = [str(_nk_infer_bin()), str(nk_path)] + [str(float(v)) for v in flat_input.reshape(-1)]
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"nk_infer failed ({proc.returncode}):\n{proc.stderr}")
    body = proc.stdout.strip()
    if not body:
        return np.array([], dtype=np.float32)
    return np.array([float(v) for v in body.split(",")], dtype=np.float32)


def _align_onnx_output(raw: np.ndarray) -> np.ndarray:
    out = np.asarray(raw, dtype=np.float32)
    if out.ndim == 4:
        # ONNX NCHW -> netkit NHWC flatten order
        out = np.transpose(out, (0, 2, 3, 1))
    return out.reshape(-1)


def _run_onnx(onnx_path: Path, network: str, input_shape: list[int], flat_input: np.ndarray) -> np.ndarray:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest("onnx and onnxruntime required for parity tests") from exc

    model = onnx.load(onnx_path)
    if model.ir_version > 10:
        model.ir_version = 10

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    if network == "mlp":
        batch, features = input_shape
        tensor = flat_input.reshape(batch, features).astype(np.float32)
    else:
        height, width, channels = input_shape
        nhwc = flat_input.reshape(height, width, channels)
        nchw = np.transpose(nhwc, (2, 0, 1))
        tensor = nchw.reshape(1, channels, height, width).astype(np.float32)

    outputs = session.run(None, {input_name: tensor})
    return _align_onnx_output(outputs[0])


class TestOnnxParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _nk_infer_bin().is_file():
            raise unittest.SkipTest(f"nk_infer not found: {_nk_infer_bin()} (run make tools/nk_infer)")
        try:
            import onnx  # noqa: F401
            import onnxruntime as ort
        except ImportError as exc:
            raise unittest.SkipTest("onnx and onnxruntime required (pip install onnx onnxruntime)") from exc
        # Synthetic CNN sidecars trigger benign MergeShapeInfo warnings at default verbosity.
        ort.set_default_logger_severity(3)

    def test_nk_matches_onnx_runtime(self) -> None:
        from netkit import read_nk, read_test_suite

        pairs = [
            (nk_name, onnx_name)
            for nk_name, onnx_name in PARITY_PAIRS
            if (MODELS / nk_name).is_file() and (MODELS / onnx_name).is_file()
        ]
        if not pairs:
            self.skipTest("no .nk/.onnx parity pairs found in models/")

        expected = 0
        for nk_name, _onnx_name in pairs:
            suite = read_test_suite(MODELS / nk_name)
            if suite is not None:
                expected += len(suite.cases)

        passed = 0
        for nk_name, onnx_name in pairs:
            nk_path = MODELS / nk_name
            onnx_path = MODELS / onnx_name
            suite = read_test_suite(nk_path)
            self.assertIsNotNone(suite, f"no embedded tests in {nk_path}")
            assert suite is not None

            arch, _weights = read_nk(nk_path)
            network = arch["network"]
            input_shape = arch["input"]

            for case in suite.cases:
                with self.subTest(model=nk_name, case=case.name):
                    flat = np.asarray(case.input, dtype=np.float32)
                    nk_out = _run_nk(nk_path, flat)
                    onnx_out = _run_onnx(onnx_path, network, input_shape, flat)
                    self.assertEqual(nk_out.size, onnx_out.size, "output length mismatch")
                    np.testing.assert_allclose(
                        nk_out,
                        onnx_out,
                        rtol=0.0,
                        atol=suite.tolerance,
                        err_msg=f"{nk_name}/{case.name}: .nk vs ONNX Runtime",
                    )
                    passed += 1

        self.assertEqual(passed, expected)


if __name__ == "__main__":
    unittest.main()
