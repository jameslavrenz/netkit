"""Tests for YOLOX Nano PAFPN + feature taps."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from netkit.arch_writer import pack_random_cnn_weights, write_nk_from_arch
from netkit.cnn_layers import _layer_weight_tensor_count
from netkit.mobilenetv4_small import build_mobilenetv4_small_arch
from netkit.reader import read_nk
from netkit.reference_forward import forward_cnn
from netkit.writer import RegressionCase, RegressionSuite
from netkit.yolox_decode import decode_yolox_output
from netkit.yolox_detector import build_yolox_mnv4_small_detector
from netkit.yolox_pafpn import (
    MNV4_SMALL_C3_BLOCK_INDEX,
    MNV4_SMALL_C4_BLOCK_INDEX,
    MNV4_SMALL_C5_BLOCK_INDEX,
    YOLOX_PAFPN_STRIDES,
    pack_yolox_pafpn_weights_flat,
    pafpn_output_elements,
    verify_mnv4_small_tap_blocks,
    yolox_pafpn_forward_from_offset,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
FULL_FIXTURE = MODELS / "yolox_mnv4_small.nk"
TAPS_FIXTURE = MODELS / "yolox_pafpn_taps.nk"


class YoloxPafpnArchTests(unittest.TestCase):
    def test_tap_block_indices(self) -> None:
        verify_mnv4_small_tap_blocks()
        self.assertEqual(MNV4_SMALL_C3_BLOCK_INDEX, 4)
        self.assertEqual(MNV4_SMALL_C4_BLOCK_INDEX, 10)
        self.assertEqual(MNV4_SMALL_C5_BLOCK_INDEX, 17)

    def test_detector_inserts_taps_and_pafpn(self) -> None:
        arch = build_yolox_mnv4_small_detector(height=64, width=64, num_classes=10, hidden_dim=64)
        types = [layer["type"] for layer in arch["layers"]]
        self.assertIn("feature_tap", types)
        self.assertEqual(types.count("feature_tap"), 2)
        self.assertEqual(types[-1], "yolox_pafpn_multiscale")
        backbone = build_mobilenetv4_small_arch(height=64, width=64, include_head=False)
        # Backbone layers preserved with two taps inserted.
        self.assertEqual(len(arch["layers"]), len(backbone["layers"]) + 2 + 1)

    def test_pafpn_weight_tensor_count(self) -> None:
        layer = {
            "type": "yolox_pafpn_multiscale",
            "c3_channels": 64,
            "c4_channels": 96,
            "c5_channels": 960,
            "hidden_dim": 64,
            "num_classes": 10,
            "num_convs": 2,
        }
        self.assertEqual(_layer_weight_tensor_count(layer), 11 + 3 * (3 + 2 * 2))

    def test_output_size_64(self) -> None:
        self.assertEqual(pafpn_output_elements(input_height=64, input_width=64, num_classes=10), 1260)
        arch = build_yolox_mnv4_small_detector(height=64, width=64, num_classes=10, hidden_dim=32, num_convs=1)
        rng = np.random.default_rng(3)
        weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        inp = rng.standard_normal(64 * 64 * 3, dtype=np.float32) * 0.05
        out = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
        self.assertEqual(out.size, pafpn_output_elements(input_height=64, input_width=64, num_classes=10))


class YoloxPafpnReferenceTests(unittest.TestCase):
    def test_numpy_self_consistency(self) -> None:
        rng = np.random.default_rng(5)
        H, C3, C4, C5, nc, ncvs = 32, 64, 96, 960, 5, 1
        c3 = rng.standard_normal((8, 8, C3), dtype=np.float32) * 0.05
        c4 = rng.standard_normal((4, 4, C4), dtype=np.float32) * 0.05
        c5 = rng.standard_normal((2, 2, C5), dtype=np.float32) * 0.05
        parts = pack_yolox_pafpn_weights_flat(
            rng,
            c3_channels=C3,
            c4_channels=C4,
            c5_channels=C5,
            hidden_dim=H,
            num_classes=nc,
            num_convs=ncvs,
            scale=0.02,
        )
        weights = np.concatenate(parts)
        out1, off1 = yolox_pafpn_forward_from_offset(
            c3, c4, c5,
            c3_channels=C3, c4_channels=C4, c5_channels=C5,
            hidden_dim=H, num_classes=nc, num_convs=ncvs,
            weights=weights, offset=0,
        )
        out2, off2 = yolox_pafpn_forward_from_offset(
            c3, c4, c5,
            c3_channels=C3, c4_channels=C4, c5_channels=C5,
            hidden_dim=H, num_classes=nc, num_convs=ncvs,
            weights=weights, offset=0,
        )
        self.assertEqual(off1, len(weights))
        self.assertEqual(off1, off2)
        np.testing.assert_allclose(out1, out2)

    def test_decode_multiscale(self) -> None:
        nc = 3
        out_c = 4 + 1 + nc
        # Build fake flat concat with one high-score cell on P3
        p3 = np.zeros((8, 8, out_c), dtype=np.float32)
        # log(0.5) so exp-decoded LTRB stays modest on the P3 stride-8 grid
        p3[2, 3, 0:4] = np.log([0.5, 0.5, 0.5, 0.5]).astype(np.float32)
        p3[2, 3, 4] = 10.0
        p3[2, 3, 5] = 10.0
        flat = np.concatenate(
            [
                p3.reshape(-1),
                np.zeros(4 * 4 * out_c, dtype=np.float32),
                np.zeros(2 * 2 * out_c, dtype=np.float32),
            ]
        )
        dets = decode_yolox_output(
            flat, num_classes=nc, score_threshold=0.5, input_height=64, input_width=64,
            strides=YOLOX_PAFPN_STRIDES,
        )
        self.assertGreaterEqual(len(dets), 1)
        self.assertEqual(dets[0].class_id, 0)


class YoloxPafpnFixtureTests(unittest.TestCase):
    def test_full_fixture_roundtrip(self) -> None:
        if not FULL_FIXTURE.is_file():
            raise unittest.SkipTest("run tools/write_yolox_mnv4_detector_fixture.py")
        arch, _weights = read_nk(FULL_FIXTURE)
        self.assertEqual(arch["layers"][-1]["type"], "yolox_pafpn_multiscale")
        taps = [layer for layer in arch["layers"] if layer["type"] == "feature_tap"]
        self.assertEqual(len(taps), 2)

    def test_taps_fixture_roundtrip(self) -> None:
        if not TAPS_FIXTURE.is_file():
            raise unittest.SkipTest("run tools/write_yolox_head_only_fixture.py")
        arch, _weights = read_nk(TAPS_FIXTURE)
        self.assertEqual(arch["layers"][-1]["type"], "yolox_pafpn_multiscale")

    def test_temp_write_roundtrip(self) -> None:
        arch = {
            "network": "cnn",
            "input": [8, 8, 64],
            "layers": [
                {"type": "feature_tap", "channels": 64, "tap_id": 0},
                {
                    "type": "conv2d",
                    "kernel_size": 3,
                    "stride": 2,
                    "filters": 96,
                    "pad_h": 1,
                    "pad_w": 1,
                    "activation": "none",
                },
                {"type": "feature_tap", "channels": 96, "tap_id": 1},
                {
                    "type": "conv2d",
                    "kernel_size": 3,
                    "stride": 2,
                    "filters": 960,
                    "pad_h": 1,
                    "pad_w": 1,
                    "activation": "none",
                },
                {
                    "type": "yolox_pafpn_multiscale",
                    "c3_channels": 64,
                    "c4_channels": 96,
                    "c5_channels": 960,
                    "hidden_dim": 32,
                    "num_classes": 3,
                    "num_convs": 1,
                },
            ],
        }
        rng = np.random.default_rng(1)
        weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        inp = rng.standard_normal(8 * 8 * 64, dtype=np.float32) * 0.05
        expected = forward_cnn(inp, arch, weights)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pafpn.nk"
            write_nk_from_arch(
                arch,
                weights,
                path,
                RegressionSuite(
                    tolerance=1e-4,
                    cases=[RegressionCase(name="tmp", input=inp, expected=expected)],
                ),
            )
            loaded = read_nk(path)
            self.assertEqual(loaded[0]["layers"][-1]["type"], "yolox_pafpn_multiscale")


if __name__ == "__main__":
    unittest.main()
