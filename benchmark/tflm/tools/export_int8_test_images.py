#!/usr/bin/env python3
"""Export prequantized int8 MNIST CNN benchmark images for int8 firmware/benches.

Writes benchmark/tflm/generated/mnist_cnn_int8_test_images.{h,cc} with int8-only
samples (no float pixels). Quantization uses the TFLite int8 model input tensor
params so netkit and TFLM ingest identical int8 vectors at test time.

Prerequisites:
  - models/mnist_cnn.nk (float TCAS source)
  - benchmark/tflm/generated/mnist_cnn_int8.tflite

Run from repo root:
  python3 benchmark/tflm/tools/export_int8_test_images.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parents[1]
GENERATED = BENCH / "generated"
NUM_IMAGES = 10
MNIST_DIGITS = tuple(range(10))
INPUT_SIZE = 784


def _one_per_digit_cases(cases: list, *, num: int = NUM_IMAGES) -> list:
    if len(cases) < num:
        raise RuntimeError(f"expected >= {num} embedded TCAS cases")
    by_digit: dict[int, object] = {}
    for case in cases:
        label = int(case.label)
        if label in by_digit:
            continue
        by_digit[label] = case
        if len(by_digit) == num:
            break
    missing = [d for d in MNIST_DIGITS if d not in by_digit]
    if missing:
        raise RuntimeError(f"embedded cases missing digits: {missing}")
    return [by_digit[d] for d in MNIST_DIGITS]


def _tflite_input_quant(tflite_path: Path) -> tuple[float, int]:
    quant_json = tflite_path.with_name("mnist_cnn_int8_tflite_quant.json")
    if quant_json.is_file():
        import json

        data = json.loads(quant_json.read_text(encoding="utf-8"))
        if data:
            first = data[0]
            return float(first["input_scale"]), int(first["input_zero_point"])

    sys.path.insert(0, str(ROOT / "python"))
    from netkit.tflite_quant_align import extract_tflite_cnn_quant_specs

    specs = extract_tflite_cnn_quant_specs(tflite_path)
    if not specs:
        raise RuntimeError(f"no quant layers in {tflite_path}")
    first = specs[0]
    return float(first.input_scale), int(first.input_zero_point)


def export_int8_test_images(
    *,
    float_nk: Path,
    tflite_path: Path,
    out_h: Path,
    out_cc: Path,
) -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.quantize import quantize_float_input
    from netkit.reader import read_test_suite

    if not float_nk.is_file():
        raise SystemExit(f"missing {float_nk} — run: make export-mnist-cnn")
    if not tflite_path.is_file():
        raise SystemExit(
            f"missing {tflite_path} — run: make -C benchmark/tflm export-cnn-int8"
        )

    input_scale, input_zp = _tflite_input_quant(tflite_path)
    suite = read_test_suite(float_nk)
    if suite is None:
        raise RuntimeError(f"missing TCAS section in {float_nk}")
    cases = _one_per_digit_cases(suite.cases, num=NUM_IMAGES)

    hdr_lines = [
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "constexpr int kMnistCnnInt8BenchmarkImageCount = 10;",
        "constexpr int kMnistCnnInt8BenchmarkInputSize = 784;",
        f"constexpr float kMnistCnnInt8BenchmarkInputScale = {input_scale:.8f}f;",
        f"constexpr int kMnistCnnInt8BenchmarkInputZeroPoint = {input_zp};",
        "",
        "struct MnistCnnInt8BenchmarkSample {",
        "  const char* name;",
        "  int label;",
        "  const int8_t* pixels;",
        "};",
        "",
        "extern const MnistCnnInt8BenchmarkSample kMnistCnnInt8BenchmarkImages[10];",
        "",
    ]

    cc_lines = [f'#include "{out_h.name}"', ""]
    for idx, case in enumerate(cases):
        pixels = np.asarray(case.input, dtype=np.float32).reshape(-1)
        if pixels.size != INPUT_SIZE:
            raise ValueError(f"case {case.name} has {pixels.size} inputs, expected {INPUT_SIZE}")
        pixels_i8 = quantize_float_input(pixels, input_scale, input_zp)
        i8_text = ", ".join(str(int(v)) for v in pixels_i8)
        cc_lines.append(
            f"alignas(16) static const int8_t kMnistCnnInt8Image{idx}[{INPUT_SIZE}] = {{{i8_text}}};"
        )
        cc_lines.append("")

    cc_lines.append("const MnistCnnInt8BenchmarkSample kMnistCnnInt8BenchmarkImages[10] = {")
    for idx, case in enumerate(cases):
        cc_lines.append(
            f'  {{"{case.name}", {int(case.label)}, kMnistCnnInt8Image{idx}}},'
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
        "--float-nk",
        type=Path,
        default=ROOT / "models" / "mnist_cnn.nk",
        help="Float .nk with TCAS cases (default: models/mnist_cnn.nk)",
    )
    parser.add_argument(
        "--tflite",
        type=Path,
        default=GENERATED / "mnist_cnn_int8.tflite",
        help="Int8 TFLite model for input quant params",
    )
    parser.add_argument(
        "--out-h",
        type=Path,
        default=GENERATED / "mnist_cnn_int8_test_images.h",
    )
    parser.add_argument(
        "--out-cc",
        type=Path,
        default=GENERATED / "mnist_cnn_int8_test_images.cc",
    )
    args = parser.parse_args()
    export_int8_test_images(
        float_nk=args.float_nk,
        tflite_path=args.tflite,
        out_h=args.out_h,
        out_cc=args.out_cc,
    )


if __name__ == "__main__":
    main()
