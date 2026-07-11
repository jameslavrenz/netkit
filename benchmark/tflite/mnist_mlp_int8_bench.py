#!/usr/bin/env python3
"""Host TensorFlow Lite (LiteRT) int8 MNIST MLP bench.

Pairs with benchmark/netkit/src/mnist_mlp_int8_main.cc:
  - same mnist_mlp_int8.tflite / same prequantized digit vectors
  - latency: batch N invokes in one timed window (default 1000) to escape ~1 µs timer noise
  - accuracy: separate untimed pass over the 10 digit images
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "benchmark" / "tflm" / "tools"
DEFAULT_MODEL = ROOT / "benchmark" / "tflm" / "generated" / "mnist_mlp_int8.tflite"
FLOAT_NK = ROOT / "models" / "mnist_mlp.nk"
QUANT_JSON = ROOT / "benchmark" / "tflm" / "generated" / "mnist_mlp_int8_tflite_quant.json"

sys.path.insert(0, str(TOOLS))
from export_int8_test_images import (  # noqa: E402
    INPUT_SIZE,
    NUM_IMAGES,
    _one_per_digit_cases,
    _tflite_input_quant,
)


def _load_interpreter(model_path: Path, *, num_threads: int, use_xnnpack: bool):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    kwargs = {
        "model_path": str(model_path),
        "num_threads": num_threads,
    }
    if not use_xnnpack:
        kwargs["experimental_op_resolver_type"] = OpResolverType.BUILTIN_REF
    return Interpreter(**kwargs)


def _load_int8_images(model_path: Path) -> list[tuple[str, int, np.ndarray]]:
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.quantize import quantize_float_input
    from netkit.reader import read_test_suite

    if not FLOAT_NK.is_file():
        raise SystemExit(f"missing {FLOAT_NK}")
    input_scale, input_zp = _tflite_input_quant(model_path, QUANT_JSON)
    suite = read_test_suite(FLOAT_NK)
    if suite is None:
        raise SystemExit(f"missing TCAS section in {FLOAT_NK}")
    cases = _one_per_digit_cases(suite.cases, num=NUM_IMAGES)
    out: list[tuple[str, int, np.ndarray]] = []
    for case in cases:
        pixels = np.asarray(case.input, dtype=np.float32).reshape(-1)
        if pixels.size != INPUT_SIZE:
            raise SystemExit(f"case {case.name} has {pixels.size} inputs")
        pixels_i8 = quantize_float_input(pixels, input_scale, input_zp)
        out.append((case.name, int(case.label), pixels_i8.astype(np.int8)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Batch passes (discard pass 0). Alias for --batch-passes.",
    )
    parser.add_argument("--batch-passes", type=int, default=None)
    parser.add_argument(
        "--batch-invokes",
        type=int,
        default=1000,
        help="Invokes per timed window (default 1000)",
    )
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--no-xnnpack",
        action="store_true",
        help="Disable XNNPACK (builtin reference kernels only)",
    )
    args = parser.parse_args()
    batch_passes = args.batch_passes if args.batch_passes is not None else args.runs
    if batch_passes < 2:
        raise SystemExit("need at least 2 batch passes to discard cold pass 0")
    if args.batch_invokes < 1:
        raise SystemExit("--batch-invokes must be >= 1")

    if not args.model.is_file():
        raise SystemExit(
            f"missing {args.model} — run: make -C benchmark/tflm export-mlp-int8"
        )

    use_xnnpack = not args.no_xnnpack
    backend = "xnnpack" if use_xnnpack else "builtin-ref"

    interp = _load_interpreter(
        args.model, num_threads=args.num_threads, use_xnnpack=use_xnnpack
    )
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    if tuple(inp["shape"]) not in ((1, 784), (1, 28, 28)):
        raise SystemExit(f"unexpected input shape {inp['shape']}")
    if inp["dtype"] != np.int8 or out["dtype"] != np.int8:
        raise SystemExit(f"expected int8 I/O, got {inp['dtype']} / {out['dtype']}")

    images = _load_int8_images(args.model)
    num_images = len(images)
    in_shape = tuple(int(x) for x in inp["shape"])

    print("TF Lite MNIST MLP int8 benchmark")
    print("  runtime:     TensorFlow Lite / LiteRT (host interpreter)")
    print(f"  backend:     {backend}")
    print(f"  threads:     {args.num_threads}")
    print("  dtype:       int8")
    print(f"  model:       {args.model}")
    print(f"  model bytes: {args.model.stat().st_size}")
    print(f"  images:      {num_images} (accuracy only)")
    print(
        f"  batch:       {args.batch_invokes} invokes x {batch_passes} passes "
        "(discard pass 0)"
    )

    correct = 0
    for i, (name, label, pixels) in enumerate(images):
        batch = pixels.reshape(in_shape)
        interp.set_tensor(inp["index"], batch)
        interp.invoke()
        logits = interp.get_tensor(out["index"]).reshape(-1)
        pred = int(logits.argmax())
        ok = pred == label
        correct += int(ok)
        print(f"  image {i} label={label} pred={pred} {'OK' if ok else 'MISS'} ({name})")
    print(f"  accuracy:    {correct}/{num_images}")

    # Fixed input for timed batch (image 0).
    interp.set_tensor(inp["index"], images[0][2].reshape(in_shape))
    interp.invoke()

    pass_averages: list[float] = []
    for _ in range(batch_passes):
        t0 = time.perf_counter()
        for _n in range(args.batch_invokes):
            interp.invoke()
        t1 = time.perf_counter()
        window_us = (t1 - t0) * 1e6
        pass_averages.append(window_us / args.batch_invokes)

    warm = pass_averages[1:]
    mean_us = float(np.mean(warm))
    print()
    print(f"TF Lite MNIST mlp_int8 benchmark summary ({backend})")
    print(
        f"  method:      discard batch pass 0; mean over {len(warm)} warm passes; "
        f"each pass = {args.batch_invokes} invokes in one timed window"
    )
    print(f"  per-invoke:  window_us / {args.batch_invokes}")
    print()
    print(f"  mean:   {mean_us:8.3f} us ({mean_us / 1000.0:6.3f} ms)")
    print(
        f"BENCHMARK_SUMMARY runtime=tflite model=mlp_int8 backend={backend} "
        f"mean_us={mean_us:.3f} runs={len(warm)} batch_invokes={args.batch_invokes} "
        f"top1_correct={correct} top1_total={num_images}"
    )
    return 0 if correct == num_images else 1


if __name__ == "__main__":
    raise SystemExit(main())
