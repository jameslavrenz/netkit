#!/usr/bin/env python3
"""Re-score YOLOX MiniDetector on a val hold-out with optional NMS.

Backends:
  torch  — MiniDetector checkpoint (default)
  tflite — exported yolox_mnv4_pafpn_320.tflite (LiteRT)
  both   — run torch then tflite on the same hold-out and print both

Compares rough greedy mAP@0.5 vs COCO-style AP@0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch
from PIL import Image

from netkit.yolox_decode import decode_yolox_output
from train_yolox_mnv4_pafpn_mini import (
    MiniDetector,
    NUM_CLASSES,
    boxes_to_xyxy_pixels,
    coco_ap_at_iou,
    ensure_coco_val,
    letterbox,
    list_coco_val_samples,
    load_yolo_label,
    preds_to_flat,
    rough_map50,
)

DEFAULT_TFLITE = ROOT / "benchmark" / "tflm" / "generated" / "yolox_mnv4_pafpn_320.tflite"
MIN_BOX_SIDE = 4.0


def _load_holdout(data_root: Path, holdout: int):
    val_root = ensure_coco_val(data_root)
    return list_coco_val_samples(val_root, holdout)


def _score_flat(
    flat: np.ndarray,
    *,
    size: int,
    nms_iou: float | None,
    gt,
) -> tuple[list, list, int, int]:
    dets = decode_yolox_output(
        flat,
        num_classes=NUM_CLASSES,
        score_threshold=0.05,
        input_height=size,
        input_width=size,
        nms_iou_threshold=nms_iou,
    )
    pred_tuples = [
        (d.class_id, float(d.score), float(d.x1), float(d.y1), float(d.x2), float(d.y2))
        for d in dets
        if (d.x2 - d.x1) >= MIN_BOX_SIDE and (d.y2 - d.y1) >= MIN_BOX_SIDE
    ]
    rough_preds = [p for p in pred_tuples if p[1] >= 0.1]
    tp, n_gt = rough_map50(rough_preds, gt, iou_thr=0.5)
    return pred_tuples, gt, tp, n_gt


def _finalize(preds_by_img, gts_by_img, map_tp, map_gt, meta: dict) -> dict:
    rough = map_tp / max(1, map_gt)
    coco_map, coco_pooled, _ = coco_ap_at_iou(
        preds_by_img, gts_by_img, iou_thr=0.5, score_thr=0.05
    )
    out = dict(meta)
    out.update(
        {
            "rough_map50": rough,
            "rough_tp_gt": f"{map_tp}/{map_gt}",
            "coco_map50": coco_map,
            "coco_ap_pooled": coco_pooled,
            "pack_worthy": coco_map >= 0.15,
        }
    )
    return out


def eval_ckpt(
    ckpt_path: Path,
    *,
    data_root: Path,
    holdout: int,
    size: int,
    hidden: int,
    nms_iou: float | None,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = int(ckpt.get("hidden", hidden))
    size = int(ckpt.get("size", size))

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    model = MiniDetector(hidden=hidden, freeze_backbone=True, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    pairs = _load_holdout(data_root, holdout)
    map_tp = 0
    map_gt = 0
    preds_by_img: list = []
    gts_by_img: list = []
    n_dets_total = 0

    for img_path, lab_path in pairs:
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        oh, ow = rgb.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(rgb, size)
        tensor = (
            torch.from_numpy(np.array(canvas, copy=True))
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(device)
            / 255.0
        )
        with torch.no_grad():
            preds = model(tensor)
        flat = preds_to_flat(preds)
        gt = boxes_to_xyxy_pixels(
            load_yolo_label(lab_path),
            size=size,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            orig_w=ow,
            orig_h=oh,
        )
        pred_tuples, gt, tp, n_gt = _score_flat(flat, size=size, nms_iou=nms_iou, gt=gt)
        n_dets_total += len(pred_tuples)
        map_tp += tp
        map_gt += n_gt
        preds_by_img.append(pred_tuples)
        gts_by_img.append(gt)

    return _finalize(
        preds_by_img,
        gts_by_img,
        map_tp,
        map_gt,
        {
            "backend": "torch",
            "ckpt": str(ckpt_path),
            "size": size,
            "hidden": hidden,
            "holdout": len(pairs),
            "nms_iou": nms_iou,
            "n_dets_total": n_dets_total,
        },
    )


def eval_tflite(
    model_path: Path,
    *,
    data_root: Path,
    holdout: int,
    size: int,
    nms_iou: float | None,
    num_threads: int = 1,
) -> dict:
    from ai_edge_litert.interpreter import Interpreter

    if not model_path.is_file():
        raise SystemExit(
            f"missing {model_path} — run: make -C benchmark/tflm export-yolox"
        )

    interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    in_idx = in_detail["index"]
    out_idx = out_detail["index"]
    in_shape = tuple(int(x) for x in in_detail["shape"])
    # Accept NHWC [1,H,W,3] or NCHW [1,3,H,W]
    nhwc = len(in_shape) == 4 and in_shape[-1] == 3

    pairs = _load_holdout(data_root, holdout)
    map_tp = 0
    map_gt = 0
    preds_by_img: list = []
    gts_by_img: list = []
    n_dets_total = 0

    for img_path, lab_path in pairs:
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        oh, ow = rgb.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(rgb, size)
        nhwc_f32 = (canvas.astype(np.float32) / 255.0)[None, ...]
        if nhwc:
            feed = nhwc_f32
        else:
            feed = np.transpose(nhwc_f32, (0, 3, 1, 2))
        interp.set_tensor(in_idx, feed.astype(np.float32, copy=False))
        interp.invoke()
        flat = np.asarray(interp.get_tensor(out_idx), dtype=np.float32).reshape(-1)
        gt = boxes_to_xyxy_pixels(
            load_yolo_label(lab_path),
            size=size,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            orig_w=ow,
            orig_h=oh,
        )
        pred_tuples, gt, tp, n_gt = _score_flat(flat, size=size, nms_iou=nms_iou, gt=gt)
        n_dets_total += len(pred_tuples)
        map_tp += tp
        map_gt += n_gt
        preds_by_img.append(pred_tuples)
        gts_by_img.append(gt)

    return _finalize(
        preds_by_img,
        gts_by_img,
        map_tp,
        map_gt,
        {
            "backend": "tflite",
            "model": str(model_path),
            "input_shape": list(in_shape),
            "size": size,
            "holdout": len(pairs),
            "nms_iou": nms_iou,
            "n_dets_total": n_dets_total,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("torch", "tflite", "both"),
        default="torch",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "yolox_mnv4_pafpn_coco_train_50k_ema.pt",
    )
    parser.add_argument("--tflite", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--holdout", type=int, default=200)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--nms-iou", type=float, default=0.65)
    parser.add_argument("--no-nms", action="store_true")
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()

    nms = None if args.no_nms else args.nms_iou
    results = []
    if args.backend in ("torch", "both"):
        results.append(
            eval_ckpt(
                args.ckpt,
                data_root=args.data,
                holdout=args.holdout,
                size=args.size,
                hidden=args.hidden,
                nms_iou=nms,
            )
        )
    if args.backend in ("tflite", "both"):
        results.append(
            eval_tflite(
                args.tflite,
                data_root=args.data,
                holdout=args.holdout,
                size=args.size,
                nms_iou=nms,
                num_threads=args.num_threads,
            )
        )

    if len(results) == 1:
        print(json.dumps(results[0], indent=2))
    else:
        payload = {"results": results}
        t = next(r for r in results if r["backend"] == "torch")
        f = next(r for r in results if r["backend"] == "tflite")
        payload["delta_coco_map50"] = float(f["coco_map50"]) - float(t["coco_map50"])
        payload["delta_rough_map50"] = float(f["rough_map50"]) - float(t["rough_map50"])
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
