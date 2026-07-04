"""Hand regression inputs and auto-generated expected outputs."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from netkit import read_nk, read_test_suite
from netkit.regression_data import HAND_CASE_INPUTS, build_hand_suite

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


class TestHandRegressionData(unittest.TestCase):
    def test_build_hand_suite_matches_bundled_tc_as(self) -> None:
        for nk_name in HAND_CASE_INPUTS:
            path = MODELS / nk_name
            arch, weights = read_nk(path)
            built = build_hand_suite(nk_name, arch, weights)
            embedded = read_test_suite(path)
            self.assertIsNotNone(embedded, f"{nk_name} missing TCAS section")
            assert embedded is not None

            self.assertAlmostEqual(built.tolerance, embedded.tolerance)
            self.assertEqual(len(built.cases), len(embedded.cases))
            for built_case, embedded_case in zip(built.cases, embedded.cases):
                self.assertEqual(built_case.name, embedded_case.name)
                np.testing.assert_allclose(
                    built_case.input,
                    embedded_case.input,
                    rtol=0,
                    atol=0,
                )
                np.testing.assert_allclose(
                    built_case.expected,
                    embedded_case.expected,
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
