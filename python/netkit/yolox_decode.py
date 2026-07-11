"""Host-side decode helpers for YOLOX multi-scale (and legacy single-scale) outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .yolox_detector import yolox_head_output_channels
from .yolox_pafpn import YOLOX_PAFPN_STRIDES


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


def _decode_grid(
    arr: np.ndarray,
    *,
    num_classes: int,
    score_threshold: float,
    stride: float,
    img_h: float,
    img_w: float,
) -> list[Detection]:
    h, w, c = arr.shape
    expected = yolox_head_output_channels(num_classes)
    if c != expected:
        raise ValueError(f"expected {expected} channels, got {c}")

    reg = arr[..., 0:4]
    obj = 1.0 / (1.0 + np.exp(-arr[..., 4:5]))
    cls = 1.0 / (1.0 + np.exp(-arr[..., 5:]))

    detections: list[Detection] = []
    for y in range(h):
        for x in range(w):
            class_scores = cls[y, x] * obj[y, x, 0]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < score_threshold:
                continue

            cx = (float(x) + 0.5) * stride
            cy = (float(y) + 0.5) * stride
            # YOLOX-style: head emits log-distances; decode with exp → positive LTRB.
            l, t, r, b = (float(np.exp(float(v))) for v in reg[y, x])
            x1 = cx - l * stride
            y1 = cy - t * stride
            x2 = cx + r * stride
            y2 = cy + b * stride
            detections.append(
                Detection(
                    x1=max(0.0, x1),
                    y1=max(0.0, y1),
                    x2=min(img_w, x2),
                    y2=min(img_h, y2),
                    score=score,
                    class_id=class_id,
                )
            )
    return detections


def decode_yolox_output(
    output: np.ndarray,
    *,
    num_classes: int,
    score_threshold: float = 0.25,
    input_height: int | None = None,
    input_width: int | None = None,
    strides: tuple[int, ...] | list[int] | None = None,
) -> list[Detection]:
    """Decode raw YOLOX output into axis-aligned boxes.

    Multi-scale (default when ``strides`` is set or flat size matches PAFPN):
    flat concat [P3|P4|P5] with strides [8,16,32].

    Legacy single-scale: HxWxC or square flat grid.
    """
    arr = np.asarray(output, dtype=np.float32)
    out_c = yolox_head_output_channels(num_classes)
    use_strides = list(strides) if strides is not None else list(YOLOX_PAFPN_STRIDES)

    if arr.ndim == 1 and input_height is not None and input_width is not None:
        # Try multi-scale PAFPN layout first.
        grids = []
        total = 0
        ok = True
        for s in use_strides:
            gh = input_height // s
            gw = input_width // s
            if gh < 1 or gw < 1:
                ok = False
                break
            grids.append((gh, gw))
            total += gh * gw * out_c
        if ok and total == arr.size:
            detections: list[Detection] = []
            offset = 0
            img_h = float(input_height)
            img_w = float(input_width)
            for (gh, gw), stride in zip(grids, use_strides):
                n = gh * gw * out_c
                grid = arr[offset : offset + n].reshape(gh, gw, out_c)
                offset += n
                detections.extend(
                    _decode_grid(
                        grid,
                        num_classes=num_classes,
                        score_threshold=score_threshold,
                        stride=float(stride),
                        img_h=img_h,
                        img_w=img_w,
                    )
                )
            detections.sort(key=lambda det: det.score, reverse=True)
            return detections

    if arr.ndim == 1:
        side = int(round(np.sqrt(arr.size / out_c)))
        if side * side * out_c != arr.size:
            raise ValueError(f"cannot infer square grid from flat output size {arr.size}")
        arr = arr.reshape(side, side, out_c)
    if arr.ndim != 3:
        raise ValueError(f"expected HxWxC output, got shape {arr.shape}")

    h, w, _c = arr.shape
    img_h = float(input_height if input_height is not None else h)
    img_w = float(input_width if input_width is not None else w)
    stride_h = img_h / float(h)
    stride_w = img_w / float(w)
    # Use mean stride for isotropic decode when single-scale.
    stride = 0.5 * (stride_h + stride_w)
    detections = _decode_grid(
        arr,
        num_classes=num_classes,
        score_threshold=score_threshold,
        stride=stride,
        img_h=img_h,
        img_w=img_w,
    )
    detections.sort(key=lambda det: det.score, reverse=True)
    return detections
