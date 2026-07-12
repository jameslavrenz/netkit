#!/usr/bin/env python3
"""Re-score a YOLOX MiniDetector checkpoint on a val hold-out with optional NMS.

Compares rough greedy mAP@0.5 vs COCO-style AP@0.5, with and without NMS.
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


def eval_ckpt(
    ckpt_path: Path,
    *,
    data_root: Path,
    holdout: int,
    size: int,
    hidden: int,
    nms_iou: float | None,
    score_thr: float,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta_hidden = int(ckpt.get("hidden", hidden))
    meta_size = int(ckpt.get("size", size))
    hidden = meta_hidden
    size = meta_size

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    model = MiniDetector(hidden=hidden, freeze_backbone=True).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    val_root = ensure_coco_val(data_root)
    pairs = list_coco_val_samples(val_root, holdout)

    map_tp = 0
    map_gt = 0
    preds_by_img: list = []
    gts_by_img: list = []
    n_dets_total = 0
    min_box_side = 4.0

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
        dets = decode_yolox_output(
            flat,
            num_classes=NUM_CLASSES,
            score_threshold=0.05,
            input_height=size,
            input_width=size,
            nms_iou_threshold=nms_iou,
        )
        gt = boxes_to_xyxy_pixels(
            load_yolo_label(lab_path),
            size=size,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            orig_w=ow,
            orig_h=oh,
        )
        pred_tuples = [
            (d.class_id, float(d.score), float(d.x1), float(d.y1), float(d.x2), float(d.y2))
            for d in dets
            if (d.x2 - d.x1) >= min_box_side and (d.y2 - d.y1) >= min_box_side
        ]
        n_dets_total += len(pred_tuples)
        rough_preds = [p for p in pred_tuples if p[1] >= 0.1]
        tp, n_gt = rough_map50(rough_preds, gt, iou_thr=0.5)
        map_tp += tp
        map_gt += n_gt
        preds_by_img.append(pred_tuples)
        gts_by_img.append(gt)

    rough = map_tp / max(1, map_gt)
    coco_map, coco_pooled, _ = coco_ap_at_iou(
        preds_by_img, gts_by_img, iou_thr=0.5, score_thr=0.05
    )
    return {
        "ckpt": str(ckpt_path),
        "size": size,
        "hidden": hidden,
        "holdout": len(pairs),
        "nms_iou": nms_iou,
        "n_dets_total": n_dets_total,
        "rough_map50": rough,
        "rough_tp_gt": f"{map_tp}/{map_gt}",
        "coco_map50": coco_map,
        "coco_ap_pooled": coco_pooled,
        "pack_worthy": coco_map >= 0.15,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--holdout", type=int, default=200)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--nms-iou", type=float, default=0.65)
    parser.add_argument("--no-nms", action="store_true")
    parser.add_argument("--score-thr", type=float, default=0.05)
    args = parser.parse_args()

    nms = None if args.no_nms else args.nms_iou
    result = eval_ckpt(
        args.ckpt,
        data_root=args.data,
        holdout=args.holdout,
        size=args.size,
        hidden=args.hidden,
        nms_iou=nms,
        score_thr=args.score_thr,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
