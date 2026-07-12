#!/usr/bin/env python3
"""Tiny real-data dry run: MNv4-Small + Nano PAFPN on a mini COCO set.

Downloads Ultralytics coco128 (~128 images with boxes) unless --data already exists.
Frozen ImageNet backbone, 320^2, short Adam train, then demos boxes on a held-out image.

Example:
  python tools/train_yolox_mnv4_pafpn_mini.py --steps 400 --batch 4
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torch is required") from exc

from PIL import Image, ImageDraw

from netkit.yolox_decode import decode_yolox_output

COCO128_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
)
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_TRAIN_IMAGE_URL = "http://images.cocodataset.org/train2017/{name}"
STRIDES = (8, 16, 32)
NUM_CLASSES = 80


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class DwPw(nn.Module):
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, 3, stride=stride, padding=1, groups=channels, bias=True
        )
        self.pw = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _silu(self.pw(self.dw(x)))


class DecoupledHead(nn.Module):
    def __init__(self, hidden: int, num_classes: int, num_convs: int = 2):
        super().__init__()
        self.stem = nn.Conv2d(hidden, hidden, 1, bias=True)
        self.cls_convs = nn.ModuleList(
            [nn.Conv2d(hidden, hidden, 3, padding=1, bias=True) for _ in range(num_convs)]
        )
        self.reg_convs = nn.ModuleList(
            [nn.Conv2d(hidden, hidden, 3, padding=1, bias=True) for _ in range(num_convs)]
        )
        self.cls_pred = nn.Conv2d(hidden, num_classes, 1, bias=True)
        self.reg_pred = nn.Conv2d(hidden, 4, 1, bias=True)
        self.obj_pred = nn.Conv2d(hidden, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = _silu(self.stem(x))
        cls = stem
        for conv in self.cls_convs:
            cls = _silu(conv(cls))
        reg = stem
        for conv in self.reg_convs:
            reg = _silu(conv(reg))
        return torch.cat([self.reg_pred(reg), self.obj_pred(reg), self.cls_pred(cls)], dim=1)


class NanoPafpn(nn.Module):
    def __init__(
        self, c3: int, c4: int, c5: int, hidden: int, num_classes: int, num_convs: int = 2
    ):
        super().__init__()
        self.lat3 = nn.Conv2d(c3, hidden, 1, bias=True)
        self.lat4 = nn.Conv2d(c4, hidden, 1, bias=True)
        self.lat5 = nn.Conv2d(c5, hidden, 1, bias=True)
        self.td_p4 = DwPw(hidden, stride=1)
        self.td_p3 = DwPw(hidden, stride=1)
        self.bu_n4 = DwPw(hidden, stride=2)
        self.bu_n5 = DwPw(hidden, stride=2)
        self.heads = nn.ModuleList(
            [DecoupledHead(hidden, num_classes, num_convs) for _ in range(3)]
        )

    def forward(
        self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor
    ) -> list[torch.Tensor]:
        l3 = self.lat3(c3)
        l4 = self.lat4(c4)
        l5 = self.lat5(c5)
        p5 = l5
        p4 = self.td_p4(l4 + F.interpolate(p5, scale_factor=2, mode="nearest"))
        p3 = self.td_p3(l3 + F.interpolate(p4, scale_factor=2, mode="nearest"))
        n3 = p3
        n4 = self.bu_n4(n3) + p4
        n5 = self.bu_n5(n4) + p5
        return [head(feat) for feat, head in zip((n3, n4, n5), self.heads)]


class MiniDetector(nn.Module):
    def __init__(self, hidden: int = 64, freeze_backbone: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "mobilenetv4_conv_small",
            pretrained=True,
            features_only=True,
            out_indices=(2, 3, 4),
        )
        self._backbone_frozen = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        with torch.no_grad():
            feats = self.backbone(torch.zeros(1, 3, 64, 64))
        c3, c4, c5 = (f.shape[1] for f in feats)
        self.neck = NanoPafpn(c3, c4, c5, hidden, NUM_CLASSES)

    def set_backbone_frozen(self, frozen: bool) -> None:
        self._backbone_frozen = frozen
        for p in self.backbone.parameters():
            p.requires_grad = not frozen

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c3, c4, c5 = self.backbone(x)
        return self.neck(c3, c4, c5)


def ensure_coco128(data_root: Path) -> Path:
    """Return path to coco128 root containing images/ and labels/."""
    root = data_root / "coco128"
    images = root / "images" / "train2017"
    labels = root / "labels" / "train2017"
    if images.is_dir() and labels.is_dir() and any(images.glob("*.jpg")):
        print(f"using existing {root}")
        return root

    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "coco128.zip"
    if not zip_path.is_file():
        print(f"downloading {COCO128_URL} ...")
        urllib.request.urlretrieve(COCO128_URL, zip_path)
    print(f"extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)
    if not images.is_dir():
        raise SystemExit(f"expected {images} after extract")
    return root


def load_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO txt → list of (cls, cx, cy, w, h) normalized 0–1."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = parts
        out.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return out


def list_samples(coco128: Path, max_images: int) -> list[tuple[Path, Path]]:
    images = sorted((coco128 / "images" / "train2017").glob("*.jpg"))
    pairs = []
    for img in images:
        lab = coco128 / "labels" / "train2017" / f"{img.stem}.txt"
        pairs.append((img, lab))
        if len(pairs) >= max_images:
            break
    return pairs


def ensure_coco_val(data_root: Path) -> Path:
    """Download COCO val2017 images + annotations; emit YOLO-format labels."""
    root = data_root / "coco_val2017"
    img_dir = root / "images"
    lab_dir = root / "labels"
    ann_file = root / "annotations" / "instances_val2017.json"
    if (
        img_dir.is_dir()
        and lab_dir.is_dir()
        and ann_file.is_file()
        and len(list(img_dir.glob("*.jpg"))) > 1000
        and len(list(lab_dir.glob("*.txt"))) > 1000
    ):
        print(f"using existing {root}")
        return root

    root.mkdir(parents=True, exist_ok=True)
    ann_zip = data_root / "annotations_trainval2017.zip"
    img_zip = data_root / "val2017.zip"
    if not ann_zip.is_file():
        print(f"downloading {COCO_ANNOTATIONS_URL} ...")
        urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, ann_zip)
    if not img_zip.is_file():
        print(f"downloading {COCO_VAL_IMAGES_URL} (~1GB) ...")
        urllib.request.urlretrieve(COCO_VAL_IMAGES_URL, img_zip)

    print("extracting annotations ...")
    with zipfile.ZipFile(ann_zip, "r") as zf:
        zf.extract("annotations/instances_val2017.json", root)
    print("extracting val2017 images ...")
    with zipfile.ZipFile(img_zip, "r") as zf:
        zf.extractall(root)
    # zip contains val2017/*.jpg
    src_imgs = root / "val2017"
    img_dir.mkdir(parents=True, exist_ok=True)
    if src_imgs.is_dir():
        for p in src_imgs.glob("*.jpg"):
            dest = img_dir / p.name
            if not dest.exists():
                p.replace(dest)
        # leave empty dir if any remain
        try:
            src_imgs.rmdir()
        except OSError:
            pass

    ann_path = root / "annotations" / "instances_val2017.json"
    print(f"writing YOLO labels from {ann_path} ...")
    data = json.loads(ann_path.read_text())
    cats = sorted(c["id"] for c in data["categories"])
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    if len(cat_to_idx) != NUM_CLASSES:
        print(f"warning: expected {NUM_CLASSES} cats, got {len(cat_to_idx)}")

    images = {im["id"]: im for im in data["images"]}
    by_img: dict[int, list[str]] = {i: [] for i in images}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        im = images.get(ann["image_id"])
        if im is None:
            continue
        cls = cat_to_idx.get(ann["category_id"])
        if cls is None:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue
        cx = (x + w / 2.0) / im["width"]
        cy = (y + h / 2.0) / im["height"]
        bw = w / im["width"]
        bh = h / im["height"]
        by_img[ann["image_id"]].append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    lab_dir.mkdir(parents=True, exist_ok=True)
    for im_id, im in images.items():
        stem = Path(im["file_name"]).stem
        (lab_dir / f"{stem}.txt").write_text("\n".join(by_img[im_id]) + ("\n" if by_img[im_id] else ""))
    print(f"coco val ready: images={len(list(img_dir.glob('*.jpg')))} labels={len(list(lab_dir.glob('*.txt')))}")
    return root


def list_coco_val_samples(root: Path, max_images: int) -> list[tuple[Path, Path]]:
    images = sorted((root / "images").glob("*.jpg"))
    pairs = []
    for img in images:
        lab = root / "labels" / f"{img.stem}.txt"
        if not lab.is_file():
            continue
        # Prefer images that have at least one box for training signal.
        if lab.stat().st_size == 0:
            continue
        pairs.append((img, lab))
        if len(pairs) >= max_images:
            break
    return pairs


def _write_yolo_labels_from_coco_json(
    ann_path: Path, lab_dir: Path, *, image_ids: set[int] | None = None
) -> dict[int, dict]:
    """Write YOLO txt labels; return image-id → image dict for selected images."""
    data = json.loads(ann_path.read_text())
    cats = sorted(c["id"] for c in data["categories"])
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    images = {im["id"]: im for im in data["images"]}
    if image_ids is not None:
        images = {i: images[i] for i in image_ids if i in images}
    by_img: dict[int, list[str]] = {i: [] for i in images}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann["image_id"] not in images:
            continue
        cls = cat_to_idx.get(ann["category_id"])
        if cls is None:
            continue
        im = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue
        cx = (x + w / 2.0) / im["width"]
        cy = (y + h / 2.0) / im["height"]
        bw = w / im["width"]
        bh = h / im["height"]
        by_img[ann["image_id"]].append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    lab_dir.mkdir(parents=True, exist_ok=True)
    for im_id, im in images.items():
        stem = Path(im["file_name"]).stem
        lines = by_img[im_id]
        (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return images


def ensure_coco_train_subset(data_root: Path, max_images: int) -> Path:
    """Download a boxed train2017 image subset (no full 18GB zip).

    Reuses annotations_trainval2017.zip; fetches images one-by-one from the COCO CDN.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    root = data_root / "coco_train_subset"
    img_dir = root / "images"
    lab_dir = root / "labels"
    ann_file = root / "annotations" / "instances_train2017.json"
    manifest = root / f"subset_{max_images}.json"

    if (
        manifest.is_file()
        and img_dir.is_dir()
        and lab_dir.is_dir()
        and len(list(img_dir.glob("*.jpg"))) >= max_images
        and len(list(lab_dir.glob("*.txt"))) >= max_images
    ):
        print(f"using existing {root} ({max_images} target)")
        return root

    root.mkdir(parents=True, exist_ok=True)
    ann_zip = data_root / "annotations_trainval2017.zip"
    if not ann_zip.is_file():
        print(f"downloading {COCO_ANNOTATIONS_URL} ...")
        urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, ann_zip)
    if not ann_file.is_file():
        print("extracting instances_train2017.json ...")
        with zipfile.ZipFile(ann_zip, "r") as zf:
            zf.extract("annotations/instances_train2017.json", root)

    print(f"selecting {max_images} boxed train2017 images ...")
    data = json.loads(ann_file.read_text())
    images = {im["id"]: im for im in data["images"]}
    boxed: set[int] = set()
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if w > 1 and h > 1 and ann["image_id"] in images:
            boxed.add(ann["image_id"])
    # Stable order by file name
    ordered = sorted(boxed, key=lambda i: images[i]["file_name"])
    selected_ids = ordered[:max_images]
    selected = [images[i] for i in selected_ids]

    img_dir.mkdir(parents=True, exist_ok=True)

    def _fetch(im: dict) -> tuple[str, bool, str]:
        name = im["file_name"]
        dest = img_dir / name
        if dest.is_file() and dest.stat().st_size > 1000:
            return name, True, "exists"
        url = COCO_TRAIN_IMAGE_URL.format(name=name)
        try:
            urllib.request.urlretrieve(url, dest)
            return name, True, "ok"
        except Exception as exc:  # noqa: BLE001
            return name, False, str(exc)

    missing = [im for im in selected if not (img_dir / im["file_name"]).is_file()]
    print(f"downloading {len(missing)} / {len(selected)} train images ...")
    fails = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(_fetch, im) for im in missing]
        done = 0
        for fut in as_completed(futs):
            name, ok, msg = fut.result()
            done += 1
            if not ok:
                fails += 1
                print(f"  fail {name}: {msg}")
            if done % 200 == 0 or done == len(futs):
                print(f"  fetched {done}/{len(futs)}", flush=True)
    if fails:
        print(f"warning: {fails} image downloads failed")

    print("writing YOLO labels for subset ...")
    _write_yolo_labels_from_coco_json(ann_file, lab_dir, image_ids=set(selected_ids))
    manifest.write_text(
        json.dumps(
            {
                "max_images": max_images,
                "selected": [im["file_name"] for im in selected],
                "count": len(selected),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"coco train subset ready: images={len(list(img_dir.glob('*.jpg')))} "
        f"labels={len(list(lab_dir.glob('*.txt')))}"
    )
    return root


def list_coco_subset_samples(root: Path, max_images: int) -> list[tuple[Path, Path]]:
    images = sorted((root / "images").glob("*.jpg"))
    pairs = []
    for img in images:
        lab = root / "labels" / f"{img.stem}.txt"
        if not lab.is_file() or lab.stat().st_size == 0:
            continue
        pairs.append((img, lab))
        if len(pairs) >= max_images:
            break
    return pairs


def letterbox(
    rgb: np.ndarray, size: int
) -> tuple[np.ndarray, float, int, int]:
    """Resize with aspect preserved; pad to size×size. Returns image, scale, pad_x, pad_y."""
    h, w = rgb.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = np.asarray(
        Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR), dtype=np.uint8
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y = (size - nh) // 2
    pad_x = (size - nw) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def color_jitter(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light brightness / contrast jitter in float, return uint8."""
    x = rgb.astype(np.float32)
    # brightness
    x = x * float(rng.uniform(0.7, 1.3))
    # contrast around per-image mean
    mean = float(x.mean())
    x = (x - mean) * float(rng.uniform(0.7, 1.3)) + mean
    return np.clip(x, 0, 255).astype(np.uint8)


def flip_lr_boxes(
    boxes: list[tuple[int, float, float, float, float]], size: int
) -> list[tuple[int, float, float, float, float]]:
    out = []
    for cls, x1, y1, x2, y2 in boxes:
        out.append((cls, size - 1 - x2, y1, size - 1 - x1, y2))
    return out


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between sets of boxes [N,4] and [M,4]."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-7
    return (inter / union).astype(np.float32)


def rough_map50(
    pred_boxes: list[tuple[int, float, float, float, float, float]],
    gt_boxes: list[tuple[int, float, float, float, float]],
    *,
    iou_thr: float = 0.5,
) -> tuple[int, int]:
    """Greedy class-aware matching. Returns (tp, n_gt) for one image.

    pred_boxes: (cls, score, x1, y1, x2, y2) sorted by score desc.
    """
    if not gt_boxes:
        return 0, 0
    gt = list(gt_boxes)
    used = [False] * len(gt)
    tp = 0
    for cls, _score, x1, y1, x2, y2 in pred_boxes:
        best_j = -1
        best_iou = 0.0
        for j, (gcls, gx1, gy1, gx2, gy2) in enumerate(gt):
            if used[j] or gcls != cls:
                continue
            iou = float(
                box_iou_xyxy(
                    np.array([[x1, y1, x2, y2]], dtype=np.float32),
                    np.array([[gx1, gy1, gx2, gy2]], dtype=np.float32),
                )[0, 0]
            )
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            used[best_j] = True
            tp += 1
    return tp, len(gt)


def coco_ap_at_iou(
    preds_by_img: list[list[tuple[int, float, float, float, float, float]]],
    gts_by_img: list[list[tuple[int, float, float, float, float]]],
    *,
    iou_thr: float = 0.5,
    score_thr: float = 0.05,
) -> tuple[float, float, dict[int, float]]:
    """COCO-style AP from precision–recall (all-point) at a fixed IoU.

    Returns (mAP over classes with GT, overall AP pooling all classes, per-class AP).
    preds: (cls, score, x1, y1, x2, y2); gts: (cls, x1, y1, x2, y2).
    """
    classes = sorted({g[0] for gts in gts_by_img for g in gts})
    if not classes:
        return 0.0, 0.0, {}

    def _ap_for_class(cid: int) -> float:
        scores: list[float] = []
        matches: list[int] = []  # 1 TP, 0 FP
        n_gt = 0
        for preds, gts in zip(preds_by_img, gts_by_img):
            gt = [(i, g) for i, g in enumerate(gts) if g[0] == cid]
            n_gt += len(gt)
            used = set()
            ranked = sorted(
                [p for p in preds if p[0] == cid and p[1] >= score_thr],
                key=lambda p: p[1],
                reverse=True,
            )
            for cls, score, x1, y1, x2, y2 in ranked:
                del cls
                best_iou = 0.0
                best_j = -1
                for j, g in gt:
                    if j in used:
                        continue
                    iou = float(
                        box_iou_xyxy(
                            np.array([[x1, y1, x2, y2]], dtype=np.float32),
                            np.array([[g[1], g[2], g[3], g[4]]], dtype=np.float32),
                        )[0, 0]
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                scores.append(float(score))
                if best_j >= 0 and best_iou >= iou_thr:
                    used.add(best_j)
                    matches.append(1)
                else:
                    matches.append(0)
        if n_gt == 0:
            return float("nan")
        if not scores:
            return 0.0
        order = np.argsort(-np.asarray(scores))
        matches_a = np.asarray(matches, dtype=np.float64)[order]
        tp = np.cumsum(matches_a)
        fp = np.cumsum(1.0 - matches_a)
        recall = tp / float(n_gt)
        precision = tp / np.maximum(tp + fp, 1e-9)
        # COCO-style all-point interpolation
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    per_cls: dict[int, float] = {}
    vals = []
    for cid in classes:
        ap = _ap_for_class(cid)
        if not np.isnan(ap):
            per_cls[cid] = ap
            vals.append(ap)
    map_cls = float(np.mean(vals)) if vals else 0.0
    # Overall: treat as one mega-class by pooling (also useful single number)
    overall = _ap_for_class_pooled(preds_by_img, gts_by_img, iou_thr=iou_thr, score_thr=score_thr)
    return map_cls, overall, per_cls


def _ap_for_class_pooled(
    preds_by_img: list[list[tuple[int, float, float, float, float, float]]],
    gts_by_img: list[list[tuple[int, float, float, float, float]]],
    *,
    iou_thr: float,
    score_thr: float,
) -> float:
    """Single PR curve with class-aware matching (all classes together)."""
    scores: list[float] = []
    matches: list[int] = []
    n_gt = sum(len(g) for g in gts_by_img)
    if n_gt == 0:
        return 0.0
    for preds, gts in zip(preds_by_img, gts_by_img):
        used = set()
        ranked = sorted(
            [p for p in preds if p[1] >= score_thr],
            key=lambda p: p[1],
            reverse=True,
        )
        for cls, score, x1, y1, x2, y2 in ranked:
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gts):
                if j in used or g[0] != cls:
                    continue
                iou = float(
                    box_iou_xyxy(
                        np.array([[x1, y1, x2, y2]], dtype=np.float32),
                        np.array([[g[1], g[2], g[3], g[4]]], dtype=np.float32),
                    )[0, 0]
                )
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            scores.append(float(score))
            if best_j >= 0 and best_iou >= iou_thr:
                used.add(best_j)
                matches.append(1)
            else:
                matches.append(0)
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    matches_a = np.asarray(matches, dtype=np.float64)[order]
    tp = np.cumsum(matches_a)
    fp = np.cumsum(1.0 - matches_a)
    recall = tp / float(n_gt)
    precision = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def boxes_to_xyxy_pixels(
    labels: list[tuple[int, float, float, float, float]],
    *,
    size: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_w: int,
    orig_h: int,
) -> list[tuple[int, float, float, float, float]]:
    """Normalized YOLO → pixel xyxy in letterboxed canvas."""
    out = []
    for cls, cx, cy, bw, bh in labels:
        x1 = (cx - bw / 2.0) * orig_w * scale + pad_x
        y1 = (cy - bh / 2.0) * orig_h * scale + pad_y
        x2 = (cx + bw / 2.0) * orig_w * scale + pad_x
        y2 = (cy + bh / 2.0) * orig_h * scale + pad_y
        x1 = float(np.clip(x1, 0, size - 1))
        y1 = float(np.clip(y1, 0, size - 1))
        x2 = float(np.clip(x2, 0, size - 1))
        y2 = float(np.clip(y2, 0, size - 1))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append((cls, x1, y1, x2, y2))
    return out


def _empty_targets(
    *, size: int, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    out = []
    for stride in STRIDES:
        gh = size // stride
        gw = size // stride
        out.append(
            (
                torch.zeros(4, gh, gw, device=device),
                torch.zeros(1, gh, gw, device=device),
                torch.zeros(NUM_CLASSES, gh, gw, device=device),
            )
        )
    return out


def assign_targets(
    boxes: list[tuple[int, float, float, float, float]],
    *,
    size: int,
    device: torch.device,
    center_radius: float = 2.5,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Center-radius multi-positive assign (YOLOX-lite; not full SimOTA).

    For each GT box, mark grid cells whose centers lie inside the box and within
    ``center_radius * stride`` of the box center, on the primary FPN level and
    its immediate neighbor levels. Smaller boxes win conflicts (overwrite).

    Returns per-scale (reg, obj, cls) on ``device``. Allocates on CPU first to
    avoid MPS graph-cache blowups.
    """
    targets = []
    for stride in STRIDES:
        gh = size // stride
        gw = size // stride
        reg = torch.zeros(4, gh, gw)
        obj = torch.zeros(1, gh, gw)
        cls = torch.zeros(NUM_CLASSES, gh, gw)
        # Track assigned box area so smaller GTs can overwrite larger ones.
        area = torch.full((gh, gw), float("inf"))
        targets.append((reg, obj, cls, area, gh, gw, stride))

    # Sort large→small so smaller boxes overwrite later.
    ordered = sorted(boxes, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]), reverse=True)

    for cls_id, x1, y1, x2, y2 in ordered:
        bw = x2 - x1
        bh = y2 - y1
        area_box = max(bw * bh, 1.0)
        side = max(bw, bh)
        if side < 32:
            primary = 0
        elif side < 96:
            primary = 1
        else:
            primary = 2
        levels = {primary}
        if primary > 0:
            levels.add(primary - 1)
        if primary < 2:
            levels.add(primary + 1)

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        for level in levels:
            reg, obj, cls, area_map, gh, gw, stride = targets[level]
            radius = center_radius * float(stride)
            gx0 = max(0, int((cx - radius) / stride))
            gy0 = max(0, int((cy - radius) / stride))
            gx1 = min(gw - 1, int((cx + radius) / stride))
            gy1 = min(gh - 1, int((cy + radius) / stride))
            eps = 1e-4
            for gy in range(gy0, gy1 + 1):
                for gx in range(gx0, gx1 + 1):
                    cell_cx = (gx + 0.5) * stride
                    cell_cy = (gy + 0.5) * stride
                    if not (x1 <= cell_cx <= x2 and y1 <= cell_cy <= y2):
                        continue
                    if (cell_cx - cx) ** 2 + (cell_cy - cy) ** 2 > radius * radius:
                        continue
                    if float(area_map[gy, gx]) < area_box:
                        continue  # already claimed by a smaller box
                    area_map[gy, gx] = area_box
                    reg[0, gy, gx] = max(eps, (cell_cx - x1) / stride)
                    reg[1, gy, gx] = max(eps, (cell_cy - y1) / stride)
                    reg[2, gy, gx] = max(eps, (x2 - cell_cx) / stride)
                    reg[3, gy, gx] = max(eps, (y2 - cell_cy) / stride)
                    obj[0, gy, gx] = 1.0
                    cls[:, gy, gx] = 0.0
                    if 0 <= cls_id < NUM_CLASSES:
                        cls[cls_id, gy, gx] = 1.0

    return [
        (reg.to(device), obj.to(device), cls.to(device))
        for reg, obj, cls, *_ in targets
    ]


def _flatten_pred_anchors(
    pred_levels: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack one-image preds into flat arrays for SimOTA.

    Returns (cx, cy, strides, ltrb_exp, obj_prob, cls_prob) with shape [N] / [N,4] / [N,C].
    pred_levels: list of [C,H,W] float tensors (CPU).
    """
    cxs: list[np.ndarray] = []
    cys: list[np.ndarray] = []
    strides: list[np.ndarray] = []
    ltrbs: list[np.ndarray] = []
    objs: list[np.ndarray] = []
    clss: list[np.ndarray] = []
    for level, pred in enumerate(pred_levels):
        stride = float(STRIDES[level])
        _c, gh, gw = pred.shape
        ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        cx = (xs.astype(np.float32) + 0.5) * stride
        cy = (ys.astype(np.float32) + 0.5) * stride
        reg = pred[0:4].detach().float().cpu().numpy()
        ltrb = np.exp(np.clip(reg, -4.0, 8.0))  # stride units
        obj_logits = np.clip(pred[4].detach().float().cpu().numpy(), -60.0, 60.0)
        cls_logits = np.clip(pred[5:].detach().float().cpu().numpy(), -60.0, 60.0)
        obj = 1.0 / (1.0 + np.exp(-obj_logits))
        cls = 1.0 / (1.0 + np.exp(-cls_logits))
        cxs.append(cx.reshape(-1))
        cys.append(cy.reshape(-1))
        strides.append(np.full(gh * gw, stride, dtype=np.float32))
        ltrbs.append(ltrb.reshape(4, -1).T)
        objs.append(obj.reshape(-1))
        clss.append(cls.reshape(NUM_CLASSES, -1).T)
    return (
        np.concatenate(cxs),
        np.concatenate(cys),
        np.concatenate(strides),
        np.concatenate(ltrbs, axis=0),
        np.concatenate(objs),
        np.concatenate(clss, axis=0),
    )


def assign_targets_simota(
    pred_levels: list[torch.Tensor],
    boxes: list[tuple[int, float, float, float, float]],
    *,
    size: int,
    device: torch.device,
    center_radius: float = 2.5,
    candidate_topk: int = 10,
    iou_weight: float = 3.0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """SimOTA dynamic label assign (YOLOX-style) using current predictions.

    Candidate anchors: cell center inside the GT box **or** within
    ``center_radius * stride`` of the GT center. Cost = BCE(cls*obj, one-hot)
    + ``iou_weight * (-log(IoU))``. Dynamic-k from top-``candidate_topk`` IoUs.
    Conflicts: one anchor → lowest-cost GT.
    """
    if not boxes:
        return _empty_targets(size=size, device=device)

    cx, cy, strides, ltrb, obj_p, cls_p = _flatten_pred_anchors(pred_levels)
    n_anch = cx.shape[0]
    gt = np.asarray(boxes, dtype=np.float32)  # [G,5] cls,x1,y1,x2,y2
    n_gt = gt.shape[0]
    gt_cls = gt[:, 0].astype(np.int64)
    gt_xyxy = gt[:, 1:5]

    # Candidate mask [G, N]
    in_box = (
        (cx[None, :] >= gt_xyxy[:, 0:1])
        & (cx[None, :] <= gt_xyxy[:, 2:3])
        & (cy[None, :] >= gt_xyxy[:, 1:2])
        & (cy[None, :] <= gt_xyxy[:, 3:4])
    )
    gt_cx = 0.5 * (gt_xyxy[:, 0] + gt_xyxy[:, 2])
    gt_cy = 0.5 * (gt_xyxy[:, 1] + gt_xyxy[:, 3])
    radius = center_radius * strides[None, :]
    in_center = (cx[None, :] - gt_cx[:, None]) ** 2 + (cy[None, :] - gt_cy[:, None]) ** 2 <= (
        radius**2
    )
    candidate = in_box | in_center
    # Keep anchors that are candidates for at least one GT (fg pool).
    fg = candidate.any(axis=0)
    if not np.any(fg):
        return _empty_targets(size=size, device=device)

    fg_idx = np.flatnonzero(fg)
    cand = candidate[:, fg_idx]  # [G, F]
    cx_f = cx[fg_idx]
    cy_f = cy[fg_idx]
    stride_f = strides[fg_idx]
    ltrb_f = ltrb[fg_idx]
    # Decode pred boxes for candidates
    px1 = cx_f - ltrb_f[:, 0] * stride_f
    py1 = cy_f - ltrb_f[:, 1] * stride_f
    px2 = cx_f + ltrb_f[:, 2] * stride_f
    py2 = cy_f + ltrb_f[:, 3] * stride_f
    pred_xyxy = np.stack([px1, py1, px2, py2], axis=1)

    ious = box_iou_xyxy(gt_xyxy, pred_xyxy)  # [G, F]
    # Pairwise classification cost (sum BCE over classes), soft targets.
    # pred_scores ≈ σ(cls) * σ(obj) as in YOLOX.
    pred_scores = cls_p[fg_idx] * obj_p[fg_idx, None]  # [F, C]
    pred_scores = np.clip(pred_scores, 1e-6, 1.0 - 1e-6)
    one_hot = np.zeros((n_gt, NUM_CLASSES), dtype=np.float32)
    for i, c in enumerate(gt_cls):
        if 0 <= c < NUM_CLASSES:
            one_hot[i, c] = 1.0
    # BCE(pred, gt) broadcast: [G,F,C]
    ps = pred_scores[None, :, :]
    gt_oh = one_hot[:, None, :]
    bce = -(gt_oh * np.log(ps) + (1.0 - gt_oh) * np.log(1.0 - ps)).sum(axis=2)
    iou_loss = -np.log(ious + 1e-8)
    cost = bce + iou_weight * iou_loss
    # Prefer in-box∩in-center over soft candidates (YOLOX large prior).
    soft_only = cand & ~(in_box[:, fg_idx] & in_center[:, fg_idx])
    cost = cost + soft_only.astype(np.float32) * 1.0e5
    cost = np.where(cand, cost, np.inf)

    # Dynamic-k matching
    matching = np.zeros((n_gt, fg_idx.shape[0]), dtype=np.uint8)
    for g in range(n_gt):
        valid = np.flatnonzero(cand[g])
        if valid.size == 0:
            continue
        k_use = min(candidate_topk, int(valid.size))
        top = np.partition(ious[g, valid], -k_use)[-k_use:]
        dyn_k = max(1, min(k_use, int(top.sum())))
        order = np.argsort(cost[g])
        picked = 0
        for j in order:
            if not cand[g, j] or not np.isfinite(cost[g, j]):
                continue
            matching[g, j] = 1
            picked += 1
            if picked >= dyn_k:
                break

    # Resolve anchors claimed by multiple GTs → keep lowest cost.
    multi = matching.sum(axis=0) > 1
    if np.any(multi):
        for j in np.flatnonzero(multi):
            gs = np.flatnonzero(matching[:, j])
            best = gs[np.argmin(cost[gs, j])]
            matching[:, j] = 0
            matching[best, j] = 1

    # Materialize targets on CPU grids
    grids: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    offset = 0
    level_offsets: list[tuple[int, int, int, int]] = []  # start, gh, gw, stride
    for stride in STRIDES:
        gh = size // stride
        gw = size // stride
        level_offsets.append((offset, gh, gw, stride))
        offset += gh * gw
        grids.append(
            (
                torch.zeros(4, gh, gw),
                torch.zeros(1, gh, gw),
                torch.zeros(NUM_CLASSES, gh, gw),
                stride,
            )
        )

    eps = 1e-4
    for g in range(n_gt):
        cls_id = int(gt_cls[g])
        x1, y1, x2, y2 = (float(v) for v in gt_xyxy[g])
        for local_j in np.flatnonzero(matching[g]):
            anch = int(fg_idx[local_j])
            # Find level / grid
            for level, (start, gh, gw, stride) in enumerate(level_offsets):
                end = start + gh * gw
                if start <= anch < end:
                    local = anch - start
                    gy, gx = divmod(local, gw)
                    reg, obj, cls, _ = grids[level]
                    cell_cx = (gx + 0.5) * float(stride)
                    cell_cy = (gy + 0.5) * float(stride)
                    reg[0, gy, gx] = max(eps, (cell_cx - x1) / float(stride))
                    reg[1, gy, gx] = max(eps, (cell_cy - y1) / float(stride))
                    reg[2, gy, gx] = max(eps, (x2 - cell_cx) / float(stride))
                    reg[3, gy, gx] = max(eps, (y2 - cell_cy) / float(stride))
                    obj[0, gy, gx] = 1.0
                    cls[:, gy, gx] = 0.0
                    if 0 <= cls_id < NUM_CLASSES:
                        cls[cls_id, gy, gx] = 1.0
                    break

    return [
        (reg.to(device), obj.to(device), cls.to(device)) for reg, obj, cls, _ in grids
    ]


def _ltrb_to_xyxy(
    ltrb: torch.Tensor, *, stride: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ltrb [B,4,H,W] in stride units → pixel xyxy grids."""
    _, _, h, w = ltrb.shape
    device = ltrb.device
    ys = torch.arange(h, device=device, dtype=ltrb.dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=device, dtype=ltrb.dtype).view(1, 1, 1, w)
    cx = (xs + 0.5) * float(stride)
    cy = (ys + 0.5) * float(stride)
    x1 = cx - ltrb[:, 0:1] * float(stride)
    y1 = cy - ltrb[:, 1:2] * float(stride)
    x2 = cx + ltrb[:, 2:3] * float(stride)
    y2 = cy + ltrb[:, 3:4] * float(stride)
    return x1, y1, x2, y2


def _giou_loss(
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    tx1: torch.Tensor,
    ty1: torch.Tensor,
    tx2: torch.Tensor,
    ty2: torch.Tensor,
) -> torch.Tensor:
    """Mean 1−GIoU over flattened positive boxes [N]."""
    inter_x1 = torch.maximum(x1, tx1)
    inter_y1 = torch.maximum(y1, ty1)
    inter_x2 = torch.minimum(x2, tx2)
    inter_y2 = torch.minimum(y2, ty2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_p = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + 1e-7
    iou = inter / union
    enc_x1 = torch.minimum(x1, tx1)
    enc_y1 = torch.minimum(y1, ty1)
    enc_x2 = torch.maximum(x2, tx2)
    enc_y2 = torch.maximum(y2, ty2)
    enc = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + 1e-7
    giou = iou - (enc - union) / enc
    return (1.0 - giou).mean()


def detection_loss(
    preds: list[torch.Tensor],
    targets: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
) -> torch.Tensor:
    """preds: list of 3 tensors [B, 5+80, H, W]; targets: per-batch list of 3 (reg,obj,cls).

    Box branch: exp(pred) vs positive LTRB targets + GIoU on decoded xyxy.
    """
    device = preds[0].device
    loss = torch.zeros((), device=device)
    bsz = preds[0].shape[0]
    for level, pred in enumerate(preds):
        stride = STRIDES[level]
        reg_p = pred[:, 0:4]
        obj_p = pred[:, 4:5]
        cls_p = pred[:, 5:]
        reg_t = torch.stack([targets[b][level][0] for b in range(bsz)], dim=0)
        obj_t = torch.stack([targets[b][level][1] for b in range(bsz)], dim=0)
        cls_t = torch.stack([targets[b][level][2] for b in range(bsz)], dim=0)
        pos = obj_t > 0.5
        loss = loss + F.binary_cross_entropy_with_logits(obj_p, obj_t)
        if pos.any():
            pos4 = pos.expand_as(reg_t)
            pos_cls = pos.expand_as(cls_t)
            # Clamp before exp for MPS/AMP stability (covers ~e^8 ≈ 3000 stride units).
            ltrb_p = torch.exp(reg_p.clamp(min=-4.0, max=8.0))
            loss = loss + F.l1_loss(ltrb_p[pos4], reg_t[pos4])
            loss = loss + F.binary_cross_entropy_with_logits(cls_p[pos_cls], cls_t[pos_cls])

            px1, py1, px2, py2 = _ltrb_to_xyxy(ltrb_p, stride=stride)
            tx1, ty1, tx2, ty2 = _ltrb_to_xyxy(reg_t, stride=stride)
            # Flatten positives: pos is [B,1,H,W]
            mask = pos[:, 0]
            loss = loss + _giou_loss(
                px1[:, 0][mask],
                py1[:, 0][mask],
                px2[:, 0][mask],
                py2[:, 0][mask],
                tx1[:, 0][mask],
                ty1[:, 0][mask],
                tx2[:, 0][mask],
                ty2[:, 0][mask],
            )
    return loss


def load_batch(
    samples: list[tuple[Path, Path]],
    indices: list[int],
    *,
    size: int,
    device: torch.device,
    rng: np.random.Generator | None = None,
    augment: bool = False,
) -> tuple[torch.Tensor, list[list[tuple[int, float, float, float, float]]]]:
    """Load images + GT boxes in letterboxed pixel xyxy (no label assign yet)."""
    imgs = []
    boxes_batch: list[list[tuple[int, float, float, float, float]]] = []
    for idx in indices:
        img_path, lab_path = samples[idx]
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        oh, ow = rgb.shape[:2]
        if augment and rng is not None:
            rgb = color_jitter(rgb, rng)
        canvas, scale, pad_x, pad_y = letterbox(rgb, size)
        labels = load_yolo_label(lab_path)
        boxes = boxes_to_xyxy_pixels(
            labels, size=size, scale=scale, pad_x=pad_x, pad_y=pad_y, orig_w=ow, orig_h=oh
        )
        if augment and rng is not None and float(rng.random()) < 0.5:
            canvas = np.ascontiguousarray(canvas[:, ::-1, :])
            boxes = flip_lr_boxes(boxes, size)
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        imgs.append(tensor)
        boxes_batch.append(boxes)
    batch = torch.stack(imgs, dim=0).to(device)
    return batch, boxes_batch


def assign_batch_targets(
    preds: list[torch.Tensor],
    boxes_batch: list[list[tuple[int, float, float, float, float]]],
    *,
    size: int,
    device: torch.device,
    mode: str,
) -> list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Assign labels for a batch. ``mode`` is ``center`` or ``simota``."""
    bsz = preds[0].shape[0]
    out: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    for b in range(bsz):
        boxes = boxes_batch[b]
        if mode == "simota":
            levels = [p[b] for p in preds]
            out.append(
                assign_targets_simota(levels, boxes, size=size, device=device)
            )
        else:
            out.append(assign_targets(boxes, size=size, device=device))
    return out


def preds_to_flat(preds: list[torch.Tensor]) -> np.ndarray:
    """Single-image list of CHW preds → flat concat NHWC for decode."""
    parts = []
    for p in preds:
        # p: 1,C,H,W
        arr = p[0].detach().float().cpu().permute(1, 2, 0).contiguous().numpy()
        parts.append(arr.reshape(-1))
    return np.concatenate(parts, axis=0)


def draw_dets(rgb: np.ndarray, dets, path: Path, max_dets: int = 20) -> None:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for det in dets[:max_dets]:
        x1, x2 = sorted((float(det.x1), float(det.x2)))
        y1, y2 = sorted((float(det.y1), float(det.y2)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1, max(0, y1 - 10)), f"{det.class_id}:{det.score:.2f}", fill=(0, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def pick_device() -> torch.device:
    # Prefer MPS but allow forcing CPU via NETKIT_TRAIN_DEVICE=cpu (MPS can OOM on long runs).
    forced = __import__("os").environ.get("NETKIT_TRAIN_DEVICE", "").strip().lower()
    if forced in ("cpu", "mps", "cuda"):
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "coco_mini")
    parser.add_argument(
        "--source",
        choices=("coco128", "coco_val", "coco_train"),
        default="coco128",
        help="coco128 mini, COCO val2017, or a boxed train2017 subset",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument(
        "--unfreeze-after",
        type=int,
        default=-1,
        help="Unfreeze backbone after this many steps (-1 = keep frozen)",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Warm-start weights only (resets step counter)",
    )
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument(
        "--no-aug",
        action="store_true",
        help="Disable flip + color jitter on the training stream",
    )
    parser.add_argument(
        "--assign",
        choices=("center", "simota"),
        default="simota",
        help="Label assigner: center-radius multi-positive, or SimOTA (pred-aware)",
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "yolox_mnv4_pafpn_mini.pt",
    )
    parser.add_argument(
        "--demo-out",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "yolox_mnv4_pafpn_mini_demo.jpg",
    )
    args = parser.parse_args()

    if args.size % 32 != 0:
        raise SystemExit("--size must be divisible by 32")

    device = pick_device()
    print(f"device={device} source={args.source} assign={args.assign}")

    holdout: list[tuple[Path, Path]]
    if args.source == "coco_val":
        data_root = ensure_coco_val(args.data)
        samples = list_coco_val_samples(data_root, args.max_images)
        if len(samples) < args.holdout + args.batch:
            raise SystemExit(f"need more images than holdout+batch; got {len(samples)}")
        holdout = samples[-args.holdout :]
        train = samples[: -args.holdout]
    elif args.source == "coco_train":
        data_root = ensure_coco_train_subset(args.data, args.max_images)
        train = list_coco_subset_samples(data_root, args.max_images)
        # Fresh hold-out from val2017 (no train overlap).
        val_root = ensure_coco_val(args.data)
        holdout = list_coco_val_samples(val_root, args.holdout)
        if len(train) < args.batch:
            raise SystemExit(f"need at least --batch train images; got {len(train)}")
        if len(holdout) < max(1, args.holdout // 2):
            raise SystemExit(f"need holdout val images; got {len(holdout)}")
    else:
        data_root = ensure_coco128(args.data)
        samples = list_samples(data_root, args.max_images)
        if len(samples) < args.holdout + args.batch:
            raise SystemExit(f"need more images than holdout+batch; got {len(samples)}")
        holdout = samples[-args.holdout :]
        train = samples[: -args.holdout]
    print(f"train images={len(train)}  holdout={len(holdout)}")

    model = MiniDetector(hidden=args.hidden, freeze_backbone=True).to(device)
    losses: list[float] = []
    start_step = 0
    warm = args.init_from if args.init_from is not None else None
    if warm is None and args.resume is not None:
        warm = args.resume
    if warm is not None and warm.is_file():
        ckpt = torch.load(warm, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if args.resume is not None and args.init_from is None:
            losses = list(ckpt.get("losses", []))
            start_step = int(ckpt.get("steps", len(losses)))
            print(f"resumed {warm} at step={start_step}")
        else:
            print(f"init-from {warm} (steps reset)")

    def make_opt(*, include_backbone: bool) -> torch.optim.Optimizer:
        if include_backbone:
            return torch.optim.Adam(
                [
                    {"params": model.neck.parameters(), "lr": args.lr},
                    {"params": model.backbone.parameters(), "lr": args.backbone_lr},
                ]
            )
        return torch.optim.Adam(
            [p for p in model.neck.parameters() if p.requires_grad], lr=args.lr
        )

    frozen = True
    model.set_backbone_frozen(True)
    opt = make_opt(include_backbone=False)

    model.train()
    rng = np.random.default_rng(0)
    use_aug = not args.no_aug
    total_steps = args.steps
    for step in range(start_step, total_steps):
        if (
            args.unfreeze_after >= 0
            and frozen
            and step >= args.unfreeze_after
        ):
            model.set_backbone_frozen(False)
            opt = make_opt(include_backbone=True)
            frozen = False
            print(f"step {step}: unfroze backbone lr={args.backbone_lr}")

        idx = rng.choice(len(train), size=args.batch, replace=len(train) < args.batch)
        images, boxes_batch = load_batch(
            train,
            [int(i) for i in idx],
            size=args.size,
            device=device,
            rng=rng,
            augment=use_aug,
        )
        opt.zero_grad()
        preds = model(images)
        targets = assign_batch_targets(
            preds,
            boxes_batch,
            size=args.size,
            device=device,
            mode=args.assign,
        )
        loss = detection_loss(preds, targets)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if device.type == "mps" and step % 100 == 0:
            torch.mps.empty_cache()
        if step % 50 == 0 or step == total_steps - 1:
            tag = "frozen" if frozen else "unfrozen"
            print(f"step {step:5d}  loss={losses[-1]:.4f}  ({tag})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "losses": losses,
        "steps": total_steps,
        "size": args.size,
        "batch": args.batch,
        "device": str(device),
        "train_images": len(train),
        "holdout_images": len(holdout),
        "source": args.source,
        "unfreeze_after": args.unfreeze_after,
        "backbone_frozen_end": frozen,
        "hidden": args.hidden,
        "num_classes": NUM_CLASSES,
        "augment": use_aug,
        "assign": args.assign,
    }
    torch.save({"state_dict": model.state_dict(), **meta}, args.out)
    (args.out.with_suffix(".json")).write_text(
        json.dumps({k: meta[k] for k in meta if k != "losses"}, indent=2) + "\n"
    )
    print(f"saved {args.out}")

    decreased = losses[0] > 0 and losses[-1] < losses[0] * 0.5
    # Warm-starts often begin mid-loss; accept non-explosion.
    loss_ok = decreased or (losses[0] > 0 and losses[-1] <= losses[0] * 1.1)
    print(
        f"loss start={losses[0]:.4f} end={losses[-1]:.4f} "
        f"decreased={decreased} loss_ok={loss_ok}"
    )

    # Hold-out scoreboard (scores + boxes + rough greedy mAP + COCO-style AP@0.5)
    model.eval()
    hold_max: list[float] = []
    hold_hits = 0
    hold_box_ok = 0
    map_tp = 0
    map_gt = 0
    preds_by_img: list[list[tuple[int, float, float, float, float, float]]] = []
    gts_by_img: list[list[tuple[int, float, float, float, float]]] = []
    min_box_side = 4.0
    demo_thr = 0.05
    for img_path, lab_path in holdout:
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        oh, ow = rgb.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(rgb, args.size)
        tensor = (
            torch.from_numpy(canvas).permute(2, 0, 1).float().unsqueeze(0).to(device)
            / 255.0
        )
        with torch.no_grad():
            preds = model(tensor)
        score_max = 0.0
        for p in preds:
            obj = torch.sigmoid(p[0, 4])
            cls = torch.sigmoid(p[0, 5:]).amax(0)
            score_max = max(score_max, float((obj * cls).max().item()))
        hold_max.append(score_max)
        if score_max >= 0.1:
            hold_hits += 1
        flat = preds_to_flat(preds)
        dets = decode_yolox_output(
            flat,
            num_classes=NUM_CLASSES,
            score_threshold=0.05,
            input_height=args.size,
            input_width=args.size,
        )
        if any(
            d.score >= 0.1
            and (d.x2 - d.x1) >= min_box_side
            and (d.y2 - d.y1) >= min_box_side
            for d in dets
        ):
            hold_box_ok += 1
        gt = boxes_to_xyxy_pixels(
            load_yolo_label(lab_path),
            size=args.size,
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
        # Rough greedy uses score>=0.1; COCO AP uses the denser 0.05 pool.
        rough_preds = [p for p in pred_tuples if p[1] >= 0.1]
        tp, n_gt = rough_map50(rough_preds, gt, iou_thr=0.5)
        map_tp += tp
        map_gt += n_gt
        preds_by_img.append(pred_tuples)
        gts_by_img.append(gt)
    hold_mean = float(np.mean(hold_max)) if hold_max else 0.0
    hold_frac = hold_hits / max(1, len(holdout))
    hold_box_frac = hold_box_ok / max(1, len(holdout))
    rough_map = map_tp / max(1, map_gt)
    coco_map, coco_ap_all, _per_cls = coco_ap_at_iou(
        preds_by_img, gts_by_img, iou_thr=0.5, score_thr=0.05
    )
    print(
        f"holdout mean_max_score={hold_mean:.3f} "
        f"frac_ge_0.1={hold_frac:.2f} ({hold_hits}/{len(holdout)}) "
        f"frac_box_side>={min_box_side:.0f}={hold_box_frac:.2f} ({hold_box_ok}/{len(holdout)}) "
        f"rough_map50={rough_map:.3f} ({map_tp}/{map_gt}) "
        f"coco_map50={coco_map:.3f} coco_ap_pooled={coco_ap_all:.3f}"
    )

    # Demo on best holdout image
    best_i = int(np.argmax(hold_max)) if hold_max else 0
    demo_img, demo_lab = holdout[best_i]
    rgb = np.asarray(Image.open(demo_img).convert("RGB"), dtype=np.uint8)
    oh, ow = rgb.shape[:2]
    canvas, scale, pad_x, pad_y = letterbox(rgb, args.size)
    tensor = (
        torch.from_numpy(canvas).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    )
    with torch.no_grad():
        preds = model(tensor)
    flat = preds_to_flat(preds)
    dets = decode_yolox_output(
        flat,
        num_classes=NUM_CLASSES,
        score_threshold=demo_thr,
        input_height=args.size,
        input_width=args.size,
    )
    print(f"demo image={demo_img.name} thr={demo_thr} detections={len(dets)}")
    for d in dets[:8]:
        print(
            f"  cls={d.class_id} score={d.score:.3f} "
            f"box=({d.x1:.0f},{d.y1:.0f})-({d.x2:.0f},{d.y2:.0f}) "
            f"wh=({d.x2 - d.x1:.0f}x{d.y2 - d.y1:.0f})"
        )
    draw_dets(canvas, dets, args.demo_out)
    print(f"wrote {args.demo_out}")

    gt = boxes_to_xyxy_pixels(
        load_yolo_label(demo_lab),
        size=args.size,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_w=ow,
        orig_h=oh,
    )
    print(f"holdout GT boxes={len(gt)}")
    strong = sum(1 for d in dets if d.score >= 0.1)
    strong_box = sum(
        1
        for d in dets
        if d.score >= 0.1
        and (d.x2 - d.x1) >= min_box_side
        and (d.y2 - d.y1) >= min_box_side
    )
    print(f"strong_dets(score>=0.1)={strong} strong_boxes(side>={min_box_side:.0f})={strong_box}")

    sane = (
        (
            loss_ok
            or (
                hold_frac >= 0.90
                and hold_box_frac >= 0.90
                and (rough_map >= 0.20 or coco_map >= 0.15)
            )
        )
        and hold_frac >= 0.30
        and hold_box_frac >= 0.30
        and strong_box > 0
    )
    # Pack bar for scale-up runs: COCO-style AP@0.5 on the val hold-out.
    pack_worthy = sane and coco_map >= 0.15
    meta_disk = json.loads(args.out.with_suffix(".json").read_text())
    meta_disk.update(
        {
            "holdout_mean_max_score": hold_mean,
            "holdout_frac_ge_0.1": hold_frac,
            "holdout_frac_box_ok": hold_box_frac,
            "holdout_rough_map50": rough_map,
            "holdout_coco_map50": coco_map,
            "holdout_coco_ap_pooled": coco_ap_all,
            "holdout_sane": sane,
            "holdout_pack_worthy": pack_worthy,
            "loss_ok": loss_ok,
            "box_decode": "exp_ltrb",
            "box_loss": "l1_exp+giou",
            "assign": args.assign,
        }
    )
    args.out.with_suffix(".json").write_text(json.dumps(meta_disk, indent=2) + "\n")

    if pack_worthy:
        print(
            f"SUCCESS: hold-out pack-worthy "
            f"(coco_map50={coco_map:.3f} ≥ 0.15 — ready to pack)"
        )
    elif sane:
        print(
            f"PARTIAL: hold-out sane but coco_map50={coco_map:.3f} < 0.15 "
            "(do not pack yet)"
        )
    elif loss_ok and hold_frac >= 0.15 and hold_box_frac >= 0.15:
        print("PARTIAL: some hold-out signal but below pack threshold")
    elif loss_ok:
        print("PARTIAL: loss ok but hold-out still weak")
    else:
        print("WARNING: loss did not clearly decrease")


if __name__ == "__main__":
    main()
