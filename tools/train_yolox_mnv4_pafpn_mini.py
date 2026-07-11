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


def assign_targets(
    boxes: list[tuple[int, float, float, float, float]],
    *,
    size: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Center-assign each box to one grid cell on one stride (by box size).

    Returns per-scale (reg, obj, cls) targets shaped [C,H,W] / [1,H,W] / [80,H,W].
    """
    targets = []
    for stride in STRIDES:
        gh = size // stride
        gw = size // stride
        reg = torch.zeros(4, gh, gw, device=device)
        obj = torch.zeros(1, gh, gw, device=device)
        cls = torch.zeros(NUM_CLASSES, gh, gw, device=device)
        targets.append((reg, obj, cls, gh, gw, stride))

    for cls_id, x1, y1, x2, y2 in boxes:
        bw = x2 - x1
        bh = y2 - y1
        side = max(bw, bh)
        if side < 32:
            level = 0
        elif side < 96:
            level = 1
        else:
            level = 2
        reg, obj, cls, gh, gw, stride = targets[level]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        gx = int(cx / stride)
        gy = int(cy / stride)
        if not (0 <= gx < gw and 0 <= gy < gh):
            continue
        cell_cx = (gx + 0.5) * stride
        cell_cy = (gy + 0.5) * stride
        # l,t,r,b in stride units (matches decode_yolox_output)
        reg[0, gy, gx] = (cell_cx - x1) / stride
        reg[1, gy, gx] = (cell_cy - y1) / stride
        reg[2, gy, gx] = (x2 - cell_cx) / stride
        reg[3, gy, gx] = (y2 - cell_cy) / stride
        obj[0, gy, gx] = 1.0
        if 0 <= cls_id < NUM_CLASSES:
            cls[cls_id, gy, gx] = 1.0

    return [(reg, obj, cls) for reg, obj, cls, *_ in targets]


def detection_loss(
    preds: list[torch.Tensor],
    targets: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
) -> torch.Tensor:
    """preds: list of 3 tensors [B, 5+80, H, W]; targets: per-batch list of 3 (reg,obj,cls)."""
    device = preds[0].device
    loss = torch.zeros((), device=device)
    bsz = preds[0].shape[0]
    for level, pred in enumerate(preds):
        # pred: B,C,H,W with C = 4+1+80
        reg_p = pred[:, 0:4]
        obj_p = pred[:, 4:5]
        cls_p = pred[:, 5:]
        reg_t = torch.stack([targets[b][level][0] for b in range(bsz)], dim=0)
        obj_t = torch.stack([targets[b][level][1] for b in range(bsz)], dim=0)
        cls_t = torch.stack([targets[b][level][2] for b in range(bsz)], dim=0)
        pos = obj_t > 0.5
        loss = loss + F.binary_cross_entropy_with_logits(obj_p, obj_t)
        if pos.any():
            # Broadcast pos [B,1,H,W] over channels
            pos4 = pos.expand_as(reg_t)
            pos_cls = pos.expand_as(cls_t)
            loss = loss + F.l1_loss(reg_p[pos4], reg_t[pos4])
            loss = loss + F.binary_cross_entropy_with_logits(cls_p[pos_cls], cls_t[pos_cls])
    return loss


def load_batch(
    samples: list[tuple[Path, Path]],
    indices: list[int],
    *,
    size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]:
    imgs = []
    tgts = []
    for idx in indices:
        img_path, lab_path = samples[idx]
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        oh, ow = rgb.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(rgb, size)
        labels = load_yolo_label(lab_path)
        boxes = boxes_to_xyxy_pixels(
            labels, size=size, scale=scale, pad_x=pad_x, pad_y=pad_y, orig_w=ow, orig_h=oh
        )
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        imgs.append(tensor)
        tgts.append(assign_targets(boxes, size=size, device=device))
    batch = torch.stack(imgs, dim=0).to(device)
    return batch, tgts


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
        draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline=(0, 255, 0), width=2)
        draw.text((det.x1, max(0, det.y1 - 10)), f"{det.class_id}:{det.score:.2f}", fill=(0, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def pick_device() -> torch.device:
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
        choices=("coco128", "coco_val"),
        default="coco128",
        help="coco128 mini set, or COCO val2017 (~5k boxed images)",
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
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--holdout", type=int, default=8)
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
    print(f"device={device} source={args.source}")

    if args.source == "coco_val":
        data_root = ensure_coco_val(args.data)
        samples = list_coco_val_samples(data_root, args.max_images)
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
    if args.resume is not None and args.resume.is_file():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        losses = list(ckpt.get("losses", []))
        start_step = int(ckpt.get("steps", len(losses)))
        print(f"resumed {args.resume} at step={start_step}")

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
        images, targets = load_batch(
            train, [int(i) for i in idx], size=args.size, device=device
        )
        opt.zero_grad()
        preds = model(images)
        loss = detection_loss(preds, targets)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step % 50 == 0 or step == total_steps - 1:
            tag = "frozen" if frozen else "unfrozen"
            print(f"step {step:5d}  loss={losses[-1]:.4f}  ({tag})")

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
    }
    torch.save({"state_dict": model.state_dict(), **meta}, args.out)
    (args.out.with_suffix(".json")).write_text(
        json.dumps({k: meta[k] for k in meta if k != "losses"}, indent=2) + "\n"
    )
    print(f"saved {args.out}")

    decreased = losses[0] > 0 and losses[-1] < losses[0] * 0.5
    print(f"loss start={losses[0]:.4f} end={losses[-1]:.4f} decreased={decreased}")

    # Hold-out scoreboard (generalization check)
    model.eval()
    hold_max: list[float] = []
    hold_hits = 0
    demo_thr = 0.05
    for img_path, _lab in holdout:
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        canvas, *_ = letterbox(rgb, args.size)
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
    hold_mean = float(np.mean(hold_max)) if hold_max else 0.0
    hold_frac = hold_hits / max(1, len(holdout))
    print(
        f"holdout mean_max_score={hold_mean:.3f} "
        f"frac_ge_0.1={hold_frac:.2f} ({hold_hits}/{len(holdout)})"
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
            f"box=({d.x1:.0f},{d.y1:.0f})-({d.x2:.0f},{d.y2:.0f})"
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
    print(f"strong_dets(score>=0.1)={strong}")

    sane = decreased and hold_frac >= 0.30 and strong > 0
    meta_disk = json.loads(args.out.with_suffix(".json").read_text())
    meta_disk.update(
        {
            "holdout_mean_max_score": hold_mean,
            "holdout_frac_ge_0.1": hold_frac,
            "holdout_sane": sane,
        }
    )
    args.out.with_suffix(".json").write_text(json.dumps(meta_disk, indent=2) + "\n")

    if sane:
        print("SUCCESS: hold-out looks sane (ready to pack)")
    elif decreased and hold_frac >= 0.15:
        print("PARTIAL: some hold-out signal but below pack threshold")
    elif decreased:
        print("PARTIAL: loss down but hold-out still weak")
    else:
        print("WARNING: loss did not clearly decrease")


if __name__ == "__main__":
    main()
