"""Tests for YOLOX MobileNetV4-Small multi-scale PAFPN detector."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from netkit.arch_writer import count_packed_cnn_weight_floats, pack_random_cnn_weights, write_nk_from_arch
from netkit.cnn_layers import _layer_weight_tensor_count
from netkit.mobilenetv4_small import build_mobilenetv4_small_arch
from netkit.reader import read_nk, read_test_suite
from netkit.reference_forward import forward_cnn
from netkit.runtime_infer import nk_infer_bin, run_nk_infer
from netkit.writer import RegressionCase, RegressionSuite
from netkit.yolox_decode import Detection, decode_yolox_output, nms_detections
from netkit.yolox_detector import (
    MNv4_SMALL_BACKBONE_OUT_CHANNELS,
    build_yolox_mnv4_small_detector,
    forward_yolox_backbone,
    head_weight_offset,
    pack_yolox_head_weights_flat,
    yolox_decoupled_head_forward_nhwc,
    yolox_head_output_channels,
)
from netkit.yolox_pafpn import pafpn_output_elements

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
LIB = ROOT / "libnetkit.a"
FULL_FIXTURE = MODELS / "yolox_mnv4_small.nk"
TAPS_FIXTURE = MODELS / "yolox_pafpn_taps.nk"


def _require_nk(path: Path, *, tool: str) -> None:
    if not path.is_file():
        raise unittest.SkipTest(f"{path.name} missing — run {tool}")


class YoloxArchTests(unittest.TestCase):
    def test_backbone_matches_mnv4_small_without_classifier(self) -> None:
        yolox = build_yolox_mnv4_small_detector(height=64, width=64, num_classes=10, hidden_dim=64)
        mnv4 = build_mobilenetv4_small_arch(height=64, width=64, include_head=False)
        self.assertEqual(yolox["input"], mnv4["input"])
        backbone_layers = [
            layer for layer in yolox["layers"]
            if layer["type"] not in ("feature_tap", "yolox_pafpn_multiscale")
        ]
        self.assertEqual(backbone_layers, mnv4["layers"])
        self.assertEqual(yolox["layers"][-1]["c5_channels"], MNv4_SMALL_BACKBONE_OUT_CHANNELS)

    def test_output_grid_multiscale(self) -> None:
        for height, width in ((64, 64), (128, 128)):
            arch = build_yolox_mnv4_small_detector(
                height=height, width=width, num_classes=3, hidden_dim=16, num_convs=1
            )
            rng = np.random.default_rng(11)
            weights = pack_random_cnn_weights(arch, rng, scale=0.02)
            inp = rng.standard_normal(height * width * 3, dtype=np.float32) * 0.05
            out = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
            self.assertEqual(
                out.size, pafpn_output_elements(input_height=height, input_width=width, num_classes=3)
            )

    def test_head_weight_tensor_count(self) -> None:
        head = {
            "type": "yolox_decoupled_head",
            "in_channels": 960,
            "hidden_dim": 64,
            "num_classes": 10,
            "num_convs": 2,
        }
        self.assertEqual(_layer_weight_tensor_count(head), 3 + 2 * 2)
        flat_parts = pack_yolox_head_weights_flat(
            np.random.default_rng(0),
            in_channels=960,
            hidden_dim=64,
            num_classes=10,
            num_convs=2,
        )
        self.assertEqual(
            sum(part.size for part in flat_parts),
            count_packed_cnn_weight_floats(
                {"network": "cnn", "input": [2, 2, 960], "layers": [head]}
            ),
        )


class YoloxChainTests(unittest.TestCase):
    def test_backbone_weights_are_prefix_of_full_detector(self) -> None:
        arch = build_yolox_mnv4_small_detector(height=64, width=64, num_classes=10, hidden_dim=64)
        mnv4 = build_mobilenetv4_small_arch(height=64, width=64, include_head=False)
        rng = np.random.default_rng(123)
        full_weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        rng2 = np.random.default_rng(123)
        backbone_weights = pack_random_cnn_weights(mnv4, rng2, scale=0.02)
        offset = head_weight_offset(arch)
        # Taps add no weights; neck starts after backbone (+ taps).
        self.assertEqual(offset, len(backbone_weights))
        np.testing.assert_array_equal(full_weights[:offset], backbone_weights)

    def test_backbone_forward_matches_mnv4_without_head(self) -> None:
        arch = build_yolox_mnv4_small_detector(height=64, width=64, num_classes=10, hidden_dim=64)
        mnv4 = build_mobilenetv4_small_arch(height=64, width=64, include_head=False)
        rng = np.random.default_rng(9)
        weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        inp = rng.standard_normal(64 * 64 * 3, dtype=np.float32) * 0.05

        via_yolox = np.asarray(forward_yolox_backbone(inp, arch, weights), dtype=np.float32)
        via_mnv4 = np.asarray(
            forward_cnn(inp, mnv4, weights[: head_weight_offset(arch)]), dtype=np.float32
        )
        np.testing.assert_allclose(via_yolox, via_mnv4, rtol=0, atol=1e-6)
        self.assertEqual(via_yolox.size, 2 * 2 * MNv4_SMALL_BACKBONE_OUT_CHANNELS)

    def test_backbone_output_shape_before_neck(self) -> None:
        arch = build_yolox_mnv4_small_detector(height=64, width=64, num_classes=10, hidden_dim=64)
        rng = np.random.default_rng(4)
        weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        inp = rng.standard_normal(64 * 64 * 3, dtype=np.float32) * 0.05
        backbone = np.asarray(forward_yolox_backbone(inp, arch, weights), dtype=np.float32)
        self.assertEqual(backbone.shape, (2 * 2 * MNv4_SMALL_BACKBONE_OUT_CHANNELS,))


class YoloxDecodeTests(unittest.TestCase):
    def test_decode_single_cell_box_geometry(self) -> None:
        num_classes = 2
        out_c = yolox_head_output_channels(num_classes)
        raw = np.zeros((1, 1, out_c), dtype=np.float32)
        # Head emits log-distances; decode uses exp → LTRB = (1,2,3,4).
        raw[0, 0, 0:4] = np.log([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
        raw[0, 0, 4] = 10.0
        raw[0, 0, 5] = -10.0
        raw[0, 0, 6] = 10.0

        dets = decode_yolox_output(
            raw,
            num_classes=num_classes,
            score_threshold=0.5,
            input_height=56,
            input_width=56,
        )
        self.assertEqual(len(dets), 1)
        det = dets[0]
        self.assertEqual(det.class_id, 1)
        self.assertGreater(det.score, 0.5)
        stride = 56.0
        cx, cy = 0.5 * stride, 0.5 * stride
        self.assertAlmostEqual(det.x1, max(0.0, cx - 1.0 * stride))
        self.assertAlmostEqual(det.y1, max(0.0, cy - 2.0 * stride))
        self.assertAlmostEqual(det.x2, min(56.0, cx + 3.0 * stride))
        self.assertAlmostEqual(det.y2, min(56.0, cy + 4.0 * stride))

    def test_decode_respects_score_threshold(self) -> None:
        num_classes = 1
        out_c = yolox_head_output_channels(num_classes)
        raw = np.zeros((1, 1, out_c), dtype=np.float32)
        raw[0, 0, 4] = -20.0
        raw[0, 0, 5] = -20.0
        self.assertEqual(
            decode_yolox_output(raw, num_classes=1, score_threshold=0.1, input_height=8, input_width=8),
            [],
        )

    def test_nms_suppresses_overlapping_same_class(self) -> None:
        dets = [
            Detection(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            Detection(1.0, 1.0, 11.0, 11.0, 0.8, 0),  # high IoU with first
            Detection(50.0, 50.0, 60.0, 60.0, 0.7, 0),  # far away
            Detection(0.0, 0.0, 10.0, 10.0, 0.95, 1),  # different class, keep
        ]
        kept = nms_detections(dets, iou_threshold=0.5)
        self.assertEqual(len(kept), 3)
        self.assertEqual({(d.class_id, round(d.score, 2)) for d in kept}, {(0, 0.9), (0, 0.7), (1, 0.95)})

    def test_decode_nms_optional(self) -> None:
        num_classes = 1
        out_c = yolox_head_output_channels(num_classes)
        # Two neighboring cells with high scores → overlapping boxes after decode.
        raw = np.zeros((1, 2, out_c), dtype=np.float32)
        raw[0, 0, 0:4] = np.log([1.0, 1.0, 1.0, 1.0])
        raw[0, 0, 4] = 10.0
        raw[0, 0, 5] = 10.0
        raw[0, 1, 0:4] = np.log([1.0, 1.0, 1.0, 1.0])
        raw[0, 1, 4] = 9.0
        raw[0, 1, 5] = 9.0
        without = decode_yolox_output(
            raw, num_classes=1, score_threshold=0.5, input_height=8, input_width=16
        )
        with_nms = decode_yolox_output(
            raw,
            num_classes=1,
            score_threshold=0.5,
            input_height=8,
            input_width=16,
            nms_iou_threshold=0.5,
        )
        self.assertGreaterEqual(len(without), 2)
        self.assertEqual(len(with_nms), 1)

    def test_decode_accepts_flat_output(self) -> None:
        num_classes = 3
        out_c = yolox_head_output_channels(num_classes)
        raw = np.zeros((2, 2, out_c), dtype=np.float32)
        raw[1, 1, 4] = 8.0
        raw[1, 1, 5] = 8.0
        flat = raw.ravel()
        dets = decode_yolox_output(
            flat, num_classes=num_classes, score_threshold=0.0, input_height=64, input_width=64
        )
        self.assertTrue(all(isinstance(d, Detection) for d in dets))


class YoloxFixtureTests(unittest.TestCase):
    def test_full_fixture_roundtrip(self) -> None:
        _require_nk(FULL_FIXTURE, tool="tools/write_yolox_mnv4_detector_fixture.py")
        arch, weights = read_nk(FULL_FIXTURE)
        suite = read_test_suite(FULL_FIXTURE)
        self.assertEqual(len(arch["layers"]), 21)  # 18 backbone + 2 taps + PAFPN
        self.assertEqual(arch["layers"][-1]["type"], "yolox_pafpn_multiscale")
        self.assertEqual(len(suite.cases), 1)

        case = suite.cases[0]
        actual = np.asarray(
            forward_cnn(np.asarray(case.input, dtype=np.float32), arch, weights), dtype=np.float32
        )
        expected = np.asarray(case.expected, dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=suite.tolerance)

    def test_taps_fixture_roundtrip(self) -> None:
        _require_nk(TAPS_FIXTURE, tool="tools/write_yolox_head_only_fixture.py")
        arch, weights = read_nk(TAPS_FIXTURE)
        suite = read_test_suite(TAPS_FIXTURE)
        self.assertEqual(arch["layers"][-1]["type"], "yolox_pafpn_multiscale")
        case = suite.cases[0]
        actual = np.asarray(
            forward_cnn(np.asarray(case.input, dtype=np.float32), arch, weights), dtype=np.float32
        )
        np.testing.assert_allclose(actual, case.expected, rtol=0, atol=suite.tolerance)


class YoloxRuntimeTests(unittest.TestCase):
    def test_cpp_full_detector_matches_reference(self) -> None:
        if not LIB.is_file() or not nk_infer_bin(ROOT).is_file():
            self.skipTest("build required — run `make lib tools/nk_infer`")
        _require_nk(FULL_FIXTURE, tool="tools/write_yolox_mnv4_detector_fixture.py")

        arch, weights = read_nk(FULL_FIXTURE)
        case = read_test_suite(FULL_FIXTURE).cases[0]
        inp = np.asarray(case.input, dtype=np.float32)
        expected = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
        actual = run_nk_infer(FULL_FIXTURE, inp, root=ROOT)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)

    def test_cpp_taps_fixture_matches_reference(self) -> None:
        if not LIB.is_file() or not nk_infer_bin(ROOT).is_file():
            self.skipTest("build required — run `make lib tools/nk_infer`")
        _require_nk(TAPS_FIXTURE, tool="tools/write_yolox_head_only_fixture.py")

        arch, weights = read_nk(TAPS_FIXTURE)
        case = read_test_suite(TAPS_FIXTURE).cases[0]
        inp = np.asarray(case.input, dtype=np.float32)
        expected = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
        actual = run_nk_infer(TAPS_FIXTURE, inp, root=ROOT)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)

    def test_temp_nk_roundtrip_through_writer(self) -> None:
        import tempfile

        arch = build_yolox_mnv4_small_detector(
            height=64, width=64, num_classes=3, hidden_dim=16, num_convs=1
        )
        rng = np.random.default_rng(55)
        weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        inp = rng.standard_normal(64 * 64 * 3, dtype=np.float32) * 0.05
        expected = forward_cnn(inp, arch, weights)
        with tempfile.TemporaryDirectory() as tmp:
            nk_path = Path(tmp) / "yolox_tmp.nk"
            write_nk_from_arch(
                arch,
                weights,
                nk_path,
                RegressionSuite(
                    tolerance=1e-4,
                    cases=[RegressionCase(name="tmp", input=inp, expected=expected)],
                ),
            )
            loaded_arch, loaded_weights = read_nk(nk_path)
            self.assertEqual(loaded_arch["layers"][-1]["type"], "yolox_pafpn_multiscale")
            out = np.asarray(forward_cnn(inp, loaded_arch, loaded_weights), dtype=np.float32)
            np.testing.assert_allclose(out, expected, rtol=0, atol=1e-4)
            if nk_infer_bin(ROOT).is_file():
                runtime = run_nk_infer(nk_path, inp, root=ROOT)
                np.testing.assert_allclose(runtime, expected, rtol=0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
