"""Embedded regression suites for bundled hand-check models.

Hand-chosen inputs live here; expected outputs are computed from each model's
weights via the NumPy reference forward pass when embedding (``make embed-tests``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .reference_forward import forward_cnn, forward_mlp
from .writer import RegressionCase, RegressionSuite


@dataclass(frozen=True)
class HandCaseInput:
    name: str
    input: list[float]


@dataclass(frozen=True)
class HandModelCases:
    tolerance: float
    cases: tuple[HandCaseInput, ...]


HAND_CASE_INPUTS: dict[str, HandModelCases] = {
    "test_mlp.nk": HandModelCases(
        tolerance=1e-5,
        cases=(
            HandCaseInput("2-layer forward", [1, 2]),
            HandCaseInput("zero input", [0, 0]),
            HandCaseInput("relu clamps negative hidden", [-1, 3]),
        ),
    ),
    "mlp_hand.nk": HandModelCases(
        tolerance=1e-5,
        cases=(
            HandCaseInput("positive features", [1, 2, 3]),
            HandCaseInput("zero input (bias only)", [0, 0, 0]),
            HandCaseInput("relu zeros negative hidden unit", [-1, 2, 1]),
            HandCaseInput("scaled positive features", [4, 5, 6]),
            HandCaseInput("fractional input", [0.5, 1.5, 2.5]),
            HandCaseInput("sparse activation", [2, 0, 1]),
        ),
    ),
    "test_cnn.nk": HandModelCases(
        tolerance=1e-5,
        cases=(
            HandCaseInput(
                "channel stacking",
                [1, 10, 2, 20, 3, 30, 4, 40],
            ),
            HandCaseInput(
                "mixed channel weights",
                [2, 4, 6, 8, 1, 3, 5, 7],
            ),
        ),
    ),
    "cnn_4x4_single.nk": HandModelCases(
        tolerance=1e-5,
        cases=(
            HandCaseInput("3x3 single-layer spatial conv", [1] * 16),
            HandCaseInput(
                "corner impulses",
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            ),
        ),
    ),
    "cnn_hand.nk": HandModelCases(
        tolerance=1e-5,
        cases=(
            HandCaseInput(
                "graded 2-channel spatial input",
                [1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60, 7, 70, 8, 80, 9, 90],
            ),
            HandCaseInput("uniform channels", [1, 2] * 9),
            HandCaseInput(
                "checkerboard pattern",
                [9, 1, 8, 2, 7, 3, 6, 4, 5, 5, 4, 6, 3, 7, 2, 8, 1, 9],
            ),
        ),
    ),
}


def build_hand_suite(
    nk_name: str,
    arch: dict[str, Any],
    weights: np.ndarray,
) -> RegressionSuite:
    """Build a regression suite with reference-forward expected outputs."""
    spec = HAND_CASE_INPUTS[nk_name]
    network = arch["network"]
    if network == "mlp":
        run_forward = forward_mlp
    elif network == "cnn":
        run_forward = forward_cnn
    else:
        raise ValueError(f"{nk_name}: unsupported network {network!r}")

    cases = [
        RegressionCase(
            name=case.name,
            input=case.input,
            expected=run_forward(np.asarray(case.input, dtype=np.float32), arch, weights),
        )
        for case in spec.cases
    ]
    return RegressionSuite(tolerance=spec.tolerance, cases=cases)


def build_hand_suites(models_dir: Path) -> dict[str, RegressionSuite]:
    """Read bundled hand models and build suites from ``HAND_CASE_INPUTS``."""
    from .reader import read_nk

    suites: dict[str, RegressionSuite] = {}
    for nk_name in HAND_CASE_INPUTS:
        arch, weights = read_nk(models_dir / nk_name)
        suites[nk_name] = build_hand_suite(nk_name, arch, weights)
    return suites
