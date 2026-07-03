"""Shared depthwise conv helpers for arch dicts and .nk packing."""

from __future__ import annotations

from typing import Any


def depthwise_kernel_hw(layer: dict[str, Any]) -> tuple[int, int]:
    """Return (kernel_h, kernel_w) for depthwise_conv2d arch layers."""
    try:
        return int(layer["kernel_h"]), int(layer["kernel_w"])
    except KeyError as exc:
        raise ValueError("depthwise_conv2d requires kernel_h and kernel_w") from exc


def depthwise_arch_entry(layer: dict[str, Any]) -> dict[str, Any]:
    """Build arch dict entry from a parsed depthwise layer descriptor."""
    entry: dict[str, Any] = {
        "type": "depthwise_conv2d",
        "kernel_h": layer["kernel_h"],
        "kernel_w": layer["kernel_w"],
        "stride": layer["stride"],
        "filters": layer["filters"],
        "activation": layer.get("activation", "none"),
    }
    if layer.get("pad_h", 0):
        entry["pad_h"] = layer["pad_h"]
    if layer.get("pad_w", 0):
        entry["pad_w"] = layer["pad_w"]
    if layer.get("activation") == "leaky_relu":
        entry["alpha"] = float(layer.get("alpha", 0.01))
    return entry
