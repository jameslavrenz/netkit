#!/usr/bin/env python3
"""Generate models/mobilenetv4_imagenet_int8.nk from the float32 ImageNet pack.

Calibrates with the same 10 ImageNet-preprocessed samples used by the host benches
(benchmark/tflm/generated/imagenet_sample_cache/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmark" / "tflm" / "tools"))

from export_imagenet_mnv4_test_images import CACHE, SAMPLES, _load_rgb, preprocess  # noqa: E402
from netkit.quantize import quantize_cnn, quantized_cnn_to_spec  # noqa: E402
from netkit.reader import read_nk  # noqa: E402
from netkit.writer import write_nk  # noqa: E402

SRC = ROOT / "models" / "mobilenetv4_imagenet_f32.nk"
OUT = ROOT / "models" / "mobilenetv4_imagenet_int8.nk"


def calibration_samples() -> np.ndarray:
    samples = []
    for filename, _url, _label, _short in SAMPLES:
        path = CACHE / filename
        if not path.is_file():
            raise SystemExit(
                f"missing {path} — run: make -C benchmark/tflm export-imagenet-mnv4-images"
            )
        samples.append(preprocess(_load_rgb(path)))
    return np.stack(samples, axis=0)


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(
            f"missing {SRC} — run:\n"
            "  python -m netkit pack --arch mobilenetv4_small --pretrained "
            "-o models/mobilenetv4_imagenet_f32.nk --height 224 --width 224 --num-classes 1000"
        )

    print(f"reading {SRC} ...")
    arch, weights = read_nk(SRC)
    cal = calibration_samples()
    print(f"calibrating with {len(cal)} ImageNet samples ({cal.shape[1:]} each) ...")
    pack = quantize_cnn(arch, weights, cal, num_calibration=len(cal))
    spec = quantized_cnn_to_spec(arch, pack)
    write_nk(OUT, spec)
    print(f"Wrote {OUT} ({len(pack.weight_tensors)} weight tensors, {OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
