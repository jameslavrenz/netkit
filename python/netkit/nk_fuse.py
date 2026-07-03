"""Packager-side fusion of primitive CNN layers into composite blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FuseOptions:
    """Composite block fusion toggles (packager-side)."""

    resnet_basic_block: bool = True
    mobilenetv4_uib: bool = True


@dataclass
class FuseArchResult:
    arch: dict[str, Any]
    weights: np.ndarray
    applied: list[str] = field(default_factory=list)


def fuse_composite_blocks(
    arch: dict[str, Any],
    weights: np.ndarray,
    *,
    options: FuseOptions | None = None,
) -> FuseArchResult:
    """Collapse primitive layer sequences into composite blocks where patterns match.

    ResNet BasicBlock fusion with skip connections is handled during ONNX import
    (see ``onnx_convert.onnx_to_spec(..., fuse_composite=True)``). This pass targets
    already-linear ``.nk`` arch dicts; today it is a no-op unless future UIB matchers
    are added.
    """
    _ = options or FuseOptions()
    return FuseArchResult(arch=arch, weights=weights, applied=[])
