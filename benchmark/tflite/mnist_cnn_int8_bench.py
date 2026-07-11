#!/usr/bin/env python3
"""Host TensorFlow Lite (LiteRT) int8 MNIST CNN bench.

Pairs with benchmark/netkit/src/mnist_cnn_int8_main.cc:
  - same mnist_cnn_int8.tflite / same prequantized digit vectors
  - same methodology (10 runs x 10 images, discard first invoke each run)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "benchmark" / "tflm" / "tools"
DEFAULT_MODEL = ROOT / "benchmark" / "tflm" / "generated" / "mnist_cnn_int8.tflite"
FLOAT_NK = ROOT / "models" / "mnist_cnn.nk"
QUANT_JSON = ROOT / "benchmark" / "tflm" / "generated" / "mnist_cnn_int8_tflite_quant.json"

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


def _load_int8_images(
    model_path: Path, *, nk_path: Path, quant_json: Path
) -> list[tuple[str, int, np.ndarray]]:
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.quantize import quantize_float_input
    from netkit.reader import read_test_suite

    if not nk_path.is_file():
        raise SystemExit(f"missing {nk_path}")
    input_scale, input_zp = _tflite_input_quant(model_path, quant_json)
    suite = read_test_suite(nk_path)
    if suite is None:
        raise SystemExit(f"missing TCAS section in {nk_path}")
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
        "--nk",
        type=Path,
        default=FLOAT_NK,
        help="Float .nk with TCAS digit vectors (default: models/mnist_cnn.nk)",
    )
    parser.add_argument(
        "--quant-json",
        type=Path,
        default=QUANT_JSON,
        help="TFLite input quant JSON (or written on first use from --model)",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--no-xnnpack",
        action="store_true",
        help="Disable XNNPACK (builtin reference kernels only)",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(
            f"missing {args.model} — run: make -C benchmark/tflm export-cnn-int8"
        )

    use_xnnpack = not args.no_xnnpack
    # Host TF Lite has no CMSIS-NN path; XNNPACK is the optimized peer.
    backend = "xnnpack" if use_xnnpack else "builtin-ref"

    interp = _load_interpreter(
        args.model, num_threads=args.num_threads, use_xnnpack=use_xnnpack
    )
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    if tuple(inp["shape"]) not in ((1, 28, 28, 1), (1, 784)):
        raise SystemExit(f"unexpected input shape {inp['shape']}")
    if inp["dtype"] != np.int8 or out["dtype"] != np.int8:
        raise SystemExit(f"expected int8 I/O, got {inp['dtype']} / {out['dtype']}")

    images = _load_int8_images(args.model, nk_path=args.nk, quant_json=args.quant_json)
    num_images = len(images)
    in_shape = tuple(int(x) for x in inp["shape"])

    print("TF Lite MNIST CNN int8 benchmark")
    print("  runtime:     TensorFlow Lite / LiteRT (host interpreter)")
    print(f"  backend:     {backend}")
    print(f"  threads:     {args.num_threads}")
    print("  dtype:       int8")
    print(f"  model:       {args.model}")
    print(f"  model bytes: {args.model.stat().st_size}")
    print(f"  images:      {num_images} per run")
    print(f"  runs:        {args.runs} (discard first invoke each run)")

    run_averages: list[float] = []
    correct = 0
    for run in range(args.runs):
        run_total = 0.0
        counted = 0
        for i, (name, label, pixels) in enumerate(images):
            batch = pixels.reshape(in_shape)
            interp.set_tensor(inp["index"], batch)
            t0 = time.perf_counter()
            interp.invoke()
            t1 = time.perf_counter()
            elapsed_us = (t1 - t0) * 1e6
            if i > 0:
                run_total += elapsed_us
                counted += 1
            if run == args.runs - 1:
                logits_i8 = interp.get_tensor(out["index"]).reshape(-1)
                pred = int(logits_i8.argmax())
                ok = pred == label
                correct += int(ok)
                print(f"  image {i} label={label} pred={pred} {'OK' if ok else 'MISS'} ({name})")
        run_averages.append(run_total / counted)

    if len(run_averages) < 2:
        raise SystemExit("need at least 2 runs to discard cold first run")
    warm_runs = run_averages[1:]
    mean_us = float(np.mean(warm_runs))
    print()
    print(f"TF Lite MNIST cnn_int8 benchmark summary ({backend})")
    print(
        f"  method:      discard run 0 + first invoke each run; "
        f"mean over {len(warm_runs)} warm runs x images 1-9"
    )
    print("  per-run avg: avg of images 1-9 (us)")
    print()
    print(f"  mean:   {mean_us:8.3f} us ({mean_us / 1000.0:6.3f} ms)")
    print(f"  accuracy:    {correct}/{num_images} on final run")
    print(
        f"BENCHMARK_SUMMARY runtime=tflite model=cnn_int8 backend={backend} "
        f"mean_us={mean_us:.3f} runs={len(warm_runs)} top1_correct={correct} top1_total={num_images}"
    )
    return 0 if correct == num_images else 1


if __name__ == "__main__":
    raise SystemExit(main())
