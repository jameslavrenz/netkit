#!/usr/bin/env python3
"""Host TensorFlow Lite (LiteRT) int8 ImageNet MobileNetV4-Conv-Small bench.

Pairs with benchmark/netkit/src/mobilenetv4_imagenet_main.cc on the int8 .nk and
benchmark/tflm/src/mobilenetv4_imagenet_int8_main.cc:
  - same pretrained int8 .tflite
  - same 10 ImageNet-preprocessed images (quantized to int8 at the input boundary)
  - same methodology (10 images x 5 loops, report top-1 + 10-image mean)
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "benchmark" / "tflm" / "tools"
DEFAULT_MODEL = ROOT / "benchmark" / "tflm" / "generated" / "mobilenetv4_imagenet_int8.tflite"
CACHE = ROOT / "benchmark" / "tflm" / "generated" / "imagenet_sample_cache"

sys.path.insert(0, str(TOOLS))
from export_imagenet_mnv4_test_images import SAMPLES, preprocess  # noqa: E402


def _load_interpreter(model_path: Path, *, num_threads: int, use_xnnpack: bool):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    kwargs = {
        "model_path": str(model_path),
        "num_threads": num_threads,
    }
    if not use_xnnpack:
        kwargs["experimental_op_resolver_type"] = OpResolverType.BUILTIN_REF
    return Interpreter(**kwargs)


def _quantize(pixels: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.rint(pixels / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def _load_images(scale: float, zero_point: int) -> list[tuple[str, int, np.ndarray]]:
    from PIL import Image

    out = []
    for filename, _url, label, short in SAMPLES:
        path = CACHE / filename
        if not path.is_file():
            raise SystemExit(
                f"missing {path} — run: make -C benchmark/tflm export-imagenet-mnv4-images"
            )
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        pixels = preprocess(rgb)
        out.append((short, label, _quantize(pixels, scale, zero_point)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--no-xnnpack",
        action="store_true",
        help="Disable XNNPACK (builtin reference kernels only)",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(
            f"missing {args.model} — run: make -C benchmark/tflm export-mobilenetv4-imagenet-int8"
        )

    use_xnnpack = not args.no_xnnpack
    backend = "xnnpack" if use_xnnpack else "builtin-ref"

    interp = _load_interpreter(
        args.model, num_threads=args.num_threads, use_xnnpack=use_xnnpack
    )
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    if tuple(inp["shape"]) != (1, 224, 224, 3):
        raise SystemExit(f"unexpected input shape {inp['shape']}")
    if inp["dtype"] != np.int8 or out["dtype"] != np.int8:
        raise SystemExit(f"expected int8 I/O, got {inp['dtype']} / {out['dtype']}")
    if int(np.prod(out["shape"])) < 1000:
        raise SystemExit(f"unexpected output shape {out['shape']}")

    in_scale, in_zp = inp["quantization"]
    # Quantize inputs in Python only; C++/interpreter path stays int8 end-to-end.
    images = _load_images(float(in_scale), int(in_zp))
    num_images = len(images)

    print("TF Lite MobileNetV4 ImageNet int8 benchmark")
    print("  runtime:     TensorFlow Lite / LiteRT (MPU interpreter)")
    print(f"  backend:     {backend}")
    print(f"  threads:     {args.num_threads}")
    print("  dtype:       int8")
    print(f"  model:       {args.model}")
    print(f"  model bytes: {args.model.stat().st_size}")
    print(f"  input:       224x224x3  outputs: {int(np.prod(out['shape']))}")
    print(
        f"  method:      {num_images} images x {args.loops} loops = "
        f"{num_images * args.loops} invokes (all timed)"
    )

    samples: list[float] = []
    correct = 0
    for loop in range(args.loops):
        for i, (short, label, pixels) in enumerate(images):
            batch = pixels.reshape(1, 224, 224, 3)
            interp.set_tensor(inp["index"], batch)
            t0 = time.perf_counter()
            interp.invoke()
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1e6)
            logits_i8 = interp.get_tensor(out["index"]).reshape(-1)
            # Positive output scale ⇒ argmax(int8) == argmax(dequant); keep dequant offline.
            pred = int(logits_i8.argmax())
            if loop == 0:
                ok = pred == label
                correct += int(ok)
                print(
                    f"  image {i} ImageNet {short} (class {label})".ljust(48)
                    + f" label={label:4d} pred={pred:4d} {'OK' if ok else 'MISS'}"
                )

    cold_us = samples[0]
    first_pass_mean = statistics.fmean(samples[:num_images])
    warm = samples[1:]
    warm_mean = statistics.fmean(warm)
    warm_median = statistics.median(warm)
    warm_min = min(warm)
    warm_max = max(warm)
    warm_std = statistics.pstdev(warm)
    top1 = 100.0 * correct / num_images

    print()
    print(f"TF Lite MobileNetV4 ImageNet int8 summary ({backend})")
    print(f"  top-1 accuracy:   {correct} / {num_images}  ({top1:.1f}%)")
    print(
        f"  10-image mean:    {first_pass_mean:9.3f} us ({first_pass_mean / 1000.0:7.3f} ms)"
        f"  <- primary latency"
    )
    print(f"  cold invoke:      {cold_us:9.3f} us ({cold_us / 1000.0:7.3f} ms)")
    print(f"  warm median:      {warm_median:9.3f} us ({warm_median / 1000.0:7.3f} ms)")
    print(
        f"  warm mean:        {warm_mean:9.3f} us ({warm_mean / 1000.0:7.3f} ms)"
        f"  over {len(warm)} invokes"
    )
    print(f"  warm min/max:     {warm_min:9.3f} / {warm_max:.3f} us")
    print(f"  warm stddev:      {warm_std:9.3f} us")
    print(
        "BENCHMARK_SUMMARY runtime=tflite model=mobilenetv4_imagenet dtype=int8 "
        f"backend={backend} threads={args.num_threads} top1_correct={correct} "
        f"top1_total={num_images} top1_pct={top1:.1f} ten_image_mean_us={first_pass_mean:.3f} "
        f"warm_median_us={warm_median:.3f} warm_mean_us={warm_mean:.3f} cold_us={cold_us:.3f} "
        f"invokes={len(samples)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
