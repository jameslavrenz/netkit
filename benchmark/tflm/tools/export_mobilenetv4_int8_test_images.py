#!/usr/bin/env python3
"""Export prequantized int8 MobileNetV4 benchmark images (56x56x3 NHWC).

Uses 10 MNIST TCAS cases from models/mnist_cnn.nk (one per digit), upsampled
28x28x1 -> 56x56x3, then quantized in Python (never in C++):

  --quant-source tflite  → TFLite / TFLM input scale/zp (default)
  --quant-source nk      → netkit .nk first-layer input scale/zp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parent
GENERATED = BENCH.parent / "generated"
INPUT_SIZE = 56 * 56 * 3
NUM_IMAGES = 10


def upsample_mnist_to_56x56x3(src28: np.ndarray) -> np.ndarray:
    src28 = np.asarray(src28, dtype=np.float32).reshape(28, 28)
    out = np.empty((56, 56, 3), dtype=np.float32)
    for y in range(56):
        for x in range(56):
            v = src28[y // 2, x // 2]
            out[y, x, :] = v
    return out.reshape(-1)


def _tflite_input_quant(tflite_path: Path) -> tuple[float, int]:
    quant_json = tflite_path.with_name("mobilenetv4_small_int8_tflite_quant.json")
    if quant_json.is_file():
        data = json.loads(quant_json.read_text(encoding="utf-8"))
        if data:
            first = data[0]
            return float(first["input_scale"]), int(first["input_zero_point"])

    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    scale, zp = inp["quantization"]
    quant_json.write_text(
        json.dumps(
            [{"input_scale": float(scale), "input_zero_point": int(zp)}],
            indent=2,
        ),
        encoding="utf-8",
    )
    return float(scale), int(zp)


def _nk_input_quant(nk_path: Path) -> tuple[float, int]:
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.quant_nk_reader import read_quant_nk

    bundle = read_quant_nk(nk_path)
    if not bundle.quant_layers:
        raise SystemExit(f"{nk_path} has no quant layers")
    q0 = bundle.quant_layers[0]
    return float(q0.input_scale), int(q0.input_zero_point)


def export_int8_test_images(
    *,
    mnist_nk: Path,
    input_scale: float,
    input_zp: int,
    out_h: Path,
    out_cc: Path,
) -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.quantize import quantize_float_input
    from netkit.reader import read_test_suite

    if not mnist_nk.is_file():
        raise SystemExit(f"missing {mnist_nk} — run: make export-mnist-cnn")

    suite = read_test_suite(mnist_nk)
    if suite is None or len(suite.cases) < NUM_IMAGES:
        raise RuntimeError(f"expected >= {NUM_IMAGES} TCAS cases in {mnist_nk}")
    cases = suite.cases[:NUM_IMAGES]

    hdr_lines = [
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        f"constexpr int kMobilenetV4Int8BenchmarkImageCount = {NUM_IMAGES};",
        f"constexpr int kMobilenetV4Int8BenchmarkInputSize = {INPUT_SIZE};",
        f"constexpr float kMobilenetV4Int8BenchmarkInputScale = {input_scale:.8f}f;",
        f"constexpr int kMobilenetV4Int8BenchmarkInputZeroPoint = {input_zp};",
        "",
        "struct MobilenetV4Int8BenchmarkSample {",
        "  const char* name;",
        "  int label;",
        "  const int8_t* pixels;",
        "};",
        "",
        f"extern const MobilenetV4Int8BenchmarkSample kMobilenetV4Int8BenchmarkImages[{NUM_IMAGES}];",
        "",
    ]

    cc_lines = [f'#include "{out_h.name}"', ""]
    for idx, case in enumerate(cases):
        pixels = upsample_mnist_to_56x56x3(np.asarray(case.input, dtype=np.float32))
        pixels_i8 = quantize_float_input(pixels, input_scale, input_zp)
        i8_text = ", ".join(str(int(v)) for v in pixels_i8)
        cc_lines.append(
            f"alignas(16) static const int8_t kMobilenetV4Int8Image{idx}[{INPUT_SIZE}] = {{{i8_text}}};"
        )
        cc_lines.append("")

    cc_lines.append(
        f"const MobilenetV4Int8BenchmarkSample kMobilenetV4Int8BenchmarkImages[{NUM_IMAGES}] = {{"
    )
    for idx, case in enumerate(cases):
        cc_lines.append(
            f'  {{"{case.name}", {int(case.label)}, kMobilenetV4Int8Image{idx}}},'
        )
    cc_lines.append("};")
    cc_lines.append("")

    out_h.parent.mkdir(parents=True, exist_ok=True)
    out_h.write_text("\n".join(hdr_lines), encoding="utf-8")
    out_cc.write_text("\n".join(cc_lines), encoding="utf-8")
    print(f"Wrote {out_h} and {out_cc} (input_scale={input_scale}, zp={input_zp})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant-source",
        choices=("tflite", "nk"),
        default="tflite",
    )
    parser.add_argument(
        "--mnist-nk",
        type=Path,
        default=ROOT / "models" / "mnist_cnn.nk",
    )
    parser.add_argument(
        "--tflite",
        type=Path,
        default=GENERATED / "mobilenetv4_small_int8.tflite",
    )
    parser.add_argument(
        "--nk",
        type=Path,
        default=ROOT / "models" / "mobilenetv4_small_int8.nk",
    )
    parser.add_argument("--out-h", type=Path, default=None)
    parser.add_argument("--out-cc", type=Path, default=None)
    args = parser.parse_args()

    if args.out_h is None:
        name = (
            "mobilenetv4_netkit_int8_test_images"
            if args.quant_source == "nk"
            else "mobilenetv4_int8_test_images"
        )
        args.out_h = GENERATED / f"{name}.h"
    if args.out_cc is None:
        args.out_cc = args.out_h.with_suffix(".cc")

    if args.quant_source == "tflite":
        if not args.tflite.is_file():
            raise SystemExit(
                f"missing {args.tflite} — run: make -C benchmark/tflm export-mobilenetv4-int8"
            )
        input_scale, input_zp = _tflite_input_quant(args.tflite)
    else:
        if not args.nk.is_file():
            raise SystemExit(f"missing {args.nk}")
        input_scale, input_zp = _nk_input_quant(args.nk)

    export_int8_test_images(
        mnist_nk=args.mnist_nk,
        input_scale=input_scale,
        input_zp=input_zp,
        out_h=args.out_h,
        out_cc=args.out_cc,
    )


if __name__ == "__main__":
    main()
