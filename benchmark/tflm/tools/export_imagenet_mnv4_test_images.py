#!/usr/bin/env python3
"""Export 10 ImageNet-normalized 224x224x3 float32 samples for MobileNetV4 host benches.

Downloads one photo per class (10 distinct ILSVRC2012 classes), cached under
benchmark/tflm/generated/imagenet_sample_cache/, applies the timm
mobilenetv4_conv_small preprocess (resize/crop + ImageNet mean/std), and emits
C arrays consumed by netkit, TFLM, and TF Lite ImageNet benches.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GENERATED = HERE.parent / "generated"
CACHE = GENERATED / "imagenet_sample_cache"

# (filename, URL, ImageNet class id, short name)
# Class ids match the standard ILSVRC2012 / timm 1000-way head.
# Wikimedia Commons only serves standard thumb widths (500, 960, ...).
SAMPLES = [
    (
        "tench.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Tinca_tinca_Prague_Vltava_2.jpg/500px-Tinca_tinca_Prague_Vltava_2.jpg",
        0,
        "tench",
    ),
    (
        "goldfish.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Goldfish3.jpg/500px-Goldfish3.jpg",
        1,
        "goldfish",
    ),
    (
        "great_white_shark.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/White_shark.jpg/500px-White_shark.jpg",
        2,
        "great_white_shark",
    ),
    (
        "golden_retriever.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/D%C3%BClmen%2C_Hausd%C3%BClmen%2C_Golden_Retriever_--_2022_--_5945.jpg/500px-D%C3%BClmen%2C_Hausd%C3%BClmen%2C_Golden_Retriever_--_2022_--_5945.jpg",
        207,
        "golden_retriever",
    ),
    (
        "tabby.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/500px-Cat_November_2010-1a.jpg",
        281,
        "tabby",
    ),
    (
        "zebra.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Plains_Zebra_Equus_quagga.jpg/500px-Plains_Zebra_Equus_quagga.jpg",
        340,
        "zebra",
    ),
    (
        "african_elephant.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/178_Male_African_bush_elephant_in_Etosha_National_Park_Photo_by_Giles_Laurent.jpg/500px-178_Male_African_bush_elephant_in_Etosha_National_Park_Photo_by_Giles_Laurent.jpg",
        386,
        "african_elephant",
    ),
    (
        "airliner.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Aeroflot_Airbus_A330_Kustov.jpg/500px-Aeroflot_Airbus_A330_Kustov.jpg",
        404,
        "airliner",
    ),
    (
        "sports_car.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/2013_Porsche_911_Carrera_4S_%28991%29_%289626546987%29.jpg/500px-2013_Porsche_911_Carrera_4S_%28991%29_%289626546987%29.jpg",
        817,
        "sports_car",
    ),
    (
        "banana.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/500px-Banana-Single.jpg",
        954,
        "banana",
    ),
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    print(f"downloading {dest.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "netkit-imagenet-bench/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow required: pip install pillow") from exc
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _center_crop_resize(rgb: np.ndarray, size: int = 224, crop_pct: float = 0.875) -> np.ndarray:
    """Match timm pretrained_cfg: resize shorter side to size/crop_pct, then center crop."""
    h, w = rgb.shape[:2]
    scale_size = int(round(size / crop_pct))
    if h < w:
        new_h = scale_size
        new_w = int(round(w * scale_size / h))
    else:
        new_w = scale_size
        new_h = int(round(h * scale_size / w))
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow required: pip install pillow") from exc
    pil = Image.fromarray(rgb).resize((new_w, new_h), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.uint8)
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    return arr[top : top + size, left : left + size, :]


def preprocess(rgb_u8: np.ndarray) -> np.ndarray:
    cropped = _center_crop_resize(rgb_u8, size=224, crop_pct=0.875)
    x = cropped.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.astype(np.float32)


def _c_float_array(name: str, values: np.ndarray) -> str:
    flat = values.reshape(-1)
    body = ",\n".join(
        ", ".join(f"{v:.8g}f" for v in flat[i : i + 8]) for i in range(0, flat.size, 8)
    )
    return f"alignas(16) const float {name}[] = {{\n{body}\n}};\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-h", type=Path, default=GENERATED / "imagenet_mnv4_test_images.h")
    parser.add_argument("--out-cc", type=Path, default=GENERATED / "imagenet_mnv4_test_images.cc")
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    args = parser.parse_args()

    labels = [s[2] for s in SAMPLES]
    if len(labels) != len(set(labels)):
        raise SystemExit("SAMPLES must use distinct ImageNet class ids")
    if len(SAMPLES) != 10:
        raise SystemExit(f"expected 10 samples, got {len(SAMPLES)}")

    samples = []
    for filename, url, label, short in SAMPLES:
        path = args.cache_dir / filename
        _download(url, path)
        pixels = preprocess(_load_rgb(path))
        samples.append((short, label, pixels))

    n = len(samples)
    input_size = 224 * 224 * 3
    hdr = "\n".join(
        [
            "#pragma once",
            "",
            "#include <cstdint>",
            "",
            f"constexpr int kImagenetMnv4BenchmarkImageCount = {n};",
            f"constexpr int kImagenetMnv4BenchmarkInputSize = {input_size};",
            "constexpr int kImagenetMnv4BenchmarkHeight = 224;",
            "constexpr int kImagenetMnv4BenchmarkWidth = 224;",
            "constexpr int kImagenetMnv4BenchmarkChannels = 3;",
            "constexpr int kImagenetMnv4BenchmarkNumClasses = 1000;",
            "",
            "struct ImagenetMnv4BenchmarkSample {",
            "  const char* name;",
            "  int label;",
            "  const float* pixels;",
            "};",
            "",
            f"extern const ImagenetMnv4BenchmarkSample kImagenetMnv4BenchmarkImages[{n}];",
            "",
        ]
    )
    args.out_h.parent.mkdir(parents=True, exist_ok=True)
    args.out_h.write_text(hdr, encoding="utf-8")

    cc_parts = [
        '#include "imagenet_mnv4_test_images.h"',
        "",
        "#include <cstdint>",
        "",
    ]
    for i, (short, _label, pixels) in enumerate(samples):
        cc_parts.append(_c_float_array(f"kImagenetMnv4Image{i}", pixels))
        cc_parts.append("")

    cc_parts.append(
        f"const ImagenetMnv4BenchmarkSample kImagenetMnv4BenchmarkImages[{n}] = {{"
    )
    for i, (short, label, _pixels) in enumerate(samples):
        cc_parts.append(
            f'  {{"ImageNet {short} (class {label})", {label}, kImagenetMnv4Image{i}}},'
        )
    cc_parts.append("};")
    cc_parts.append("")
    args.out_cc.write_text("\n".join(cc_parts), encoding="utf-8")
    print(f"Wrote {args.out_h} and {args.out_cc} ({n} images, {input_size} floats each)")


if __name__ == "__main__":
    main()
