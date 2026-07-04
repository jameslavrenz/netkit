"""Tests for YOLOX MobileNetV4-Small single-scale detector."""

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
from netkit.yolox_decode import Detection, decode_yolox_output
from netkit.yolox_detector import (
    MNv4_SMALL_BACKBONE_OUT_CHANNELS,
    build_yolox_mnv4_small_detector,
    forward_yolox_backbone,
    forward_yolox_head_nhwc,
    head_weight_offset,
    pack_yolox_head_weights_flat,
    yolox_decoupled_head_forward_nhwc,
    yolox_head_output_channels,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
LIB = ROOT / "libnetkit.a"
FULL_FIXTURE = MODELS / "yolox_mnv4_small.nk"
HEAD_FIXTURE = MODELS / "yolox_head_only.nk"


def _require_nk(path: Path, *, tool: str) -> None:
    if not path.is_file():
        raise unittest.SkipTest(f"{path.name} missing — run {tool}")


class YoloxArchTests(unittest.TestCase):
    def test_backbone_matches_mnv4_small_without_classifier(self) -> None:
        yolox = build_yolox_mnv4_small_detector(height=56, width=56, num_classes=10, hidden_dim=64)
        mnv4 = build_mobilenetv4_small_arch(height=56, width=56, include_head=False)
        self.assertEqual(yolox["input"], mnv4["input"])
        self.assertEqual(yolox["layers"][:-1], mnv4["layers"])
        self.assertEqual(yolox["layers"][-1]["in_channels"], MNv4_SMALL_BACKBONE_OUT_CHANNELS)

    def test_output_grid_stride_is_32(self) -> None:
        for height, width, grid in ((56, 56, 2), (64, 64, 2), (32, 32, 1)):
            arch = build_yolox_mnv4_small_detector(
                height=height, width=width, num_classes=3, hidden_dim=16, num_convs=1
            )
            rng = np.random.default_rng(11)
            weights = pack_random_cnn_weights(arch, rng, scale=0.02)
            inp = rng.standard_normal(height * width * 3, dtype=np.float32) * 0.05
            out = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
            out_c = yolox_head_output_channels(3)
            self.assertEqual(out.size, grid * grid * out_c)

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
        self.assertEqual(sum(part.size for part in flat_parts), count_packed_cnn_weight_floats(
            {"network": "cnn", "input": [2, 2, 960], "layers": [head]}
        ))


class YoloxChainTests(unittest.TestCase):
    def test_backbone_weights_are_prefix_of_full_detector(self) -> None:
        arch = build_yolox_mnv4_small_detector(height=56, width=56, num_classes=10, hidden_dim=64)
        mnv4 = build_mobilenetv4_small_arch(height=56, width=56, include_head=False)
        rng = np.random.default_rng(123)
        full_weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        rng2 = np.random.default_rng(123)
        backbone_weights = pack_random_cnn_weights(mnv4, rng2, scale=0.02)
        offset = head_weight_offset(arch)
        self.assertEqual(offset, len(backbone_weights))
        np.testing.assert_array_equal(full_weights[:offset], backbone_weights)

    def test_backbone_forward_matches_mnv4_without_head(self) -> None:
        arch = build_yolox_mnv4_small_detector(height=56, width=56, num_classes=10, hidden_dim=64)
        mnv4 = build_mobilenetv4_small_arch(height=56, width=56, include_head=False)
        rng = np.random.default_rng(9)
        weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        inp = rng.standard_normal(56 * 56 * 3, dtype=np.float32) * 0.05

        via_yolox = np.asarray(forward_yolox_backbone(inp, arch, weights), dtype=np.float32)
        via_mnv4 = np.asarray(forward_cnn(inp, mnv4, weights[: head_weight_offset(arch)]), dtype=np.float32)
        np.testing.assert_allclose(via_yolox, via_mnv4, rtol=0, atol=1e-6)
        self.assertEqual(via_yolox.size, 2 * 2 * MNv4_SMALL_BACKBONE_OUT_CHANNELS)

    def test_head_on_backbone_features_matches_full_forward(self) -> None:
        arch = build_yolox_mnv4_small_detector(height=56, width=56, num_classes=10, hidden_dim=64)
        rng = np.random.default_rng(17)
        weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        inp = rng.standard_normal(56 * 56 * 3, dtype=np.float32) * 0.05

        full_out = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
        backbone_flat = forward_yolox_backbone(inp, arch, weights)
        features = np.asarray(backbone_flat, dtype=np.float32).reshape(2, 2, MNv4_SMALL_BACKBONE_OUT_CHANNELS)
        head_layer = arch["layers"][-1]
        head_out = forward_yolox_head_nhwc(
            features, head_layer, weights, offset=head_weight_offset(arch)
        )
        np.testing.assert_allclose(head_out.ravel(), full_out, rtol=0, atol=1e-5)

    def test_backbone_output_shape_before_head(self) -> None:
        arch = build_yolox_mnv4_small_detector(height=56, width=56, num_classes=10, hidden_dim=64)
        rng = np.random.default_rng(4)
        weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        inp = rng.standard_normal(56 * 56 * 3, dtype=np.float32) * 0.05
        backbone = np.asarray(forward_yolox_backbone(inp, arch, weights), dtype=np.float32)
        self.assertEqual(backbone.shape, (2 * 2 * MNv4_SMALL_BACKBONE_OUT_CHANNELS,))


class YoloxDecodeTests(unittest.TestCase):
    def test_decode_single_cell_box_geometry(self) -> None:
        """Hand-built 1×1 output: verify grid decode math with known ltrb logits."""
        num_classes = 2
        out_c = yolox_head_output_channels(num_classes)
        raw = np.zeros((1, 1, out_c), dtype=np.float32)
        raw[0, 0, 0:4] = [1.0, 2.0, 3.0, 4.0]
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

    def test_decode_accepts_flat_output(self) -> None:
        num_classes = 3
        out_c = yolox_head_output_channels(num_classes)
        raw = np.zeros((2, 2, out_c), dtype=np.float32)
        raw[1, 1, 4] = 8.0
        raw[1, 1, 5] = 8.0
        flat = raw.ravel()
        dets = decode_yolox_output(flat, num_classes=num_classes, score_threshold=0.0, input_height=64, input_width=64)
        self.assertTrue(all(isinstance(d, Detection) for d in dets))


class YoloxFixtureTests(unittest.TestCase):
    def test_full_fixture_roundtrip(self) -> None:
        _require_nk(FULL_FIXTURE, tool="tools/write_yolox_mnv4_detector_fixture.py")
        arch, weights = read_nk(FULL_FIXTURE)
        suite = read_test_suite(FULL_FIXTURE)
        self.assertEqual(len(arch["layers"]), 19)
        self.assertEqual(arch["layers"][-1]["type"], "yolox_decoupled_head")
        self.assertEqual(len(suite.cases), 1)

        case = suite.cases[0]
        actual = np.asarray(forward_cnn(np.asarray(case.input, dtype=np.float32), arch, weights), dtype=np.float32)
        expected = np.asarray(case.expected, dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=suite.tolerance)

    def test_full_fixture_backbone_head_chain(self) -> None:
        _require_nk(FULL_FIXTURE, tool="tools/write_yolox_mnv4_detector_fixture.py")
        arch, weights = read_nk(FULL_FIXTURE)
        suite = read_test_suite(FULL_FIXTURE)
        case = suite.cases[0]
        inp = np.asarray(case.input, dtype=np.float32)

        full_out = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
        backbone = np.asarray(forward_yolox_backbone(inp, arch, weights), dtype=np.float32).reshape(
            2, 2, MNv4_SMALL_BACKBONE_OUT_CHANNELS
        )
        head_out = forward_yolox_head_nhwc(
            backbone, arch["layers"][-1], weights, offset=head_weight_offset(arch)
        )
        np.testing.assert_allclose(head_out.ravel(), full_out, rtol=0, atol=suite.tolerance)
        np.testing.assert_allclose(full_out, case.expected, rtol=0, atol=suite.tolerance)

    def test_head_only_fixture_roundtrip(self) -> None:
        _require_nk(HEAD_FIXTURE, tool="tools/write_yolox_head_only_fixture.py")
        arch, weights = read_nk(HEAD_FIXTURE)
        suite = read_test_suite(HEAD_FIXTURE)
        self.assertEqual(len(arch["layers"]), 1)
        self.assertEqual(arch["input"], [2, 2, MNv4_SMALL_BACKBONE_OUT_CHANNELS])
        case = suite.cases[0]
        actual = np.asarray(forward_cnn(np.asarray(case.input, dtype=np.float32), arch, weights), dtype=np.float32)
        np.testing.assert_allclose(actual, case.expected, rtol=0, atol=suite.tolerance)

    def test_head_only_reference_matches_isolated_forward(self) -> None:
        _require_nk(HEAD_FIXTURE, tool="tools/write_yolox_head_only_fixture.py")
        arch, weights = read_nk(HEAD_FIXTURE)
        case = read_test_suite(HEAD_FIXTURE).cases[0]
        inp = np.asarray(case.input, dtype=np.float32).reshape(2, 2, MNv4_SMALL_BACKBONE_OUT_CHANNELS)
        head = arch["layers"][0]
        isolated, _ = yolox_decoupled_head_forward_nhwc(
            inp,
            in_channels=head["in_channels"],
            hidden_dim=head["hidden_dim"],
            num_classes=head["num_classes"],
            num_convs=head["num_convs"],
            weights=weights,
            offset=0,
        )
        np.testing.assert_allclose(isolated.ravel(), case.expected, rtol=0, atol=1e-4)


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

    def test_cpp_head_only_matches_reference(self) -> None:
        if not LIB.is_file() or not nk_infer_bin(ROOT).is_file():
            self.skipTest("build required — run `make lib tools/nk_infer`")
        _require_nk(HEAD_FIXTURE, tool="tools/write_yolox_head_only_fixture.py")

        arch, weights = read_nk(HEAD_FIXTURE)
        case = read_test_suite(HEAD_FIXTURE).cases[0]
        inp = np.asarray(case.input, dtype=np.float32)
        expected = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
        actual = run_nk_infer(HEAD_FIXTURE, inp, root=ROOT)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)

    def test_temp_nk_roundtrip_through_writer(self) -> None:
        import tempfile

        arch = build_yolox_mnv4_small_detector(height=32, width=32, num_classes=3, hidden_dim=16, num_convs=1)
        rng = np.random.default_rng(55)
        weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        inp = rng.standard_normal(32 * 32 * 3, dtype=np.float32) * 0.05
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
            self.assertEqual(loaded_arch["layers"][-1]["type"], "yolox_decoupled_head")
            out = np.asarray(forward_cnn(inp, loaded_arch, loaded_weights), dtype=np.float32)
            np.testing.assert_allclose(out, expected, rtol=0, atol=1e-4)
            if nk_infer_bin(ROOT).is_file():
                runtime = run_nk_infer(nk_path, inp, root=ROOT)
                np.testing.assert_allclose(runtime, expected, rtol=0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
