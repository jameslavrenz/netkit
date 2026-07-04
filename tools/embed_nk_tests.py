#!/usr/bin/env python3
"""Embed regression test cases into hand-check .nk models."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit import read_nk, write_nk_from_arch
from netkit.regression_data import HAND_CASE_INPUTS, build_hand_suite

MODELS = ROOT / "models"


def main() -> None:
    for nk_name in HAND_CASE_INPUTS:
        nk_path = MODELS / nk_name
        arch, weights = read_nk(nk_path)
        suite = build_hand_suite(nk_name, arch, weights)
        write_nk_from_arch(arch, weights, nk_path, suite)
        print(f"Embedded {len(suite.cases)} cases into {nk_path}")


if __name__ == "__main__":
    main()
