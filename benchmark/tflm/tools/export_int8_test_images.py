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

VARIANTS = {
    "cnn": {
        "float_nk": ROOT / "models" / "mnist_cnn.nk",
        "tflite": GENERATED / "mnist_cnn_int8.tflite",
        "quant_json": GENERATED / "mnist_cnn_int8_tflite_quant.json",
        "out_h": GENERATED / "mnist_cnn_int8_test_images.h",
        "out_cc": GENERATED / "mnist_cnn_int8_test_images.cc",
        "count_name": "kMnistCnnInt8BenchmarkImageCount",
        "size_name": "kMnistCnnInt8BenchmarkInputSize",
        "scale_name": "kMnistCnnInt8BenchmarkInputScale",
        "zp_name": "kMnistCnnInt8BenchmarkInputZeroPoint",
        "sample_name": "MnistCnnInt8BenchmarkSample",
        "images_name": "kMnistCnnInt8BenchmarkImages",
        "image_symbol": "kMnistCnnInt8Image",
    },
    "mlp": {
        "float_nk": ROOT / "models" / "mnist_mlp.nk",
        "tflite": GENERATED / "mnist_mlp_int8.tflite",
        "quant_json": GENERATED / "mnist_mlp_int8_tflite_quant.json",
        "out_h": GENERATED / "mnist_mlp_int8_test_images.h",
        "out_cc": GENERATED / "mnist_mlp_int8_test_images.cc",
        "count_name": "kMnistMlpInt8BenchmarkImageCount",
        "size_name": "kMnistMlpInt8BenchmarkInputSize",
        "scale_name": "kMnistMlpInt8BenchmarkInputScale",
        "zp_name": "kMnistMlpInt8BenchmarkInputZeroPoint",
        "sample_name": "MnistMlpInt8BenchmarkSample",
        "images_name": "kMnistMlpInt8BenchmarkImages",
        "image_symbol": "kMnistMlpInt8Image",
    },
}


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


def _tflite_input_quant(tflite_path: Path, quant_json: Path) -> tuple[float, int]:
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


def _netkit_input_quant(nk_int8: Path) -> tuple[float, int]:
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.quant_nk_reader import read_quant_nk

    bundle = read_quant_nk(nk_int8)
    if not bundle.quant_layers:
        raise RuntimeError(f"no quant layers in {nk_int8}")
    first = bundle.quant_layers[0]
    return float(first.input_scale), int(first.input_zero_point)


def export_int8_test_images(*, variant: str) -> None:
    spec = VARIANTS[variant]
    float_nk = spec["float_nk"]
    tflite_path = spec["tflite"]
    out_h = spec["out_h"]
    out_cc = spec["out_cc"]
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.quantize import quantize_float_input
    from netkit.reader import read_test_suite

    if not float_nk.is_file():
        raise SystemExit(f"missing {float_nk}")
    if not tflite_path.is_file():
        raise SystemExit(f"missing {tflite_path}")

    input_scale, input_zp = _tflite_input_quant(tflite_path, spec["quant_json"])
    suite = read_test_suite(float_nk)
    if suite is None:
        raise RuntimeError(f"missing TCAS section in {float_nk}")
    cases = _one_per_digit_cases(suite.cases, num=NUM_IMAGES)

    hdr_lines = [
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        f"constexpr int {spec['count_name']} = {len(cases)};",
        f"constexpr int {spec['size_name']} = {INPUT_SIZE};",
        f"constexpr float {spec['scale_name']} = {input_scale:.8f}f;",
        f"constexpr int {spec['zp_name']} = {input_zp};",
        "",
        f"struct {spec['sample_name']} {{",
        "  const char* name;",
        "  int label;",
        "  const int8_t* pixels;",
        "};",
        "",
        f"extern const {spec['sample_name']} {spec['images_name']}[{len(cases)}];",
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
            f"alignas(16) static const int8_t {spec['image_symbol']}{idx}[{INPUT_SIZE}] = {{{i8_text}}};"
        )
        cc_lines.append("")

    cc_lines.append(f"const {spec['sample_name']} {spec['images_name']}[{len(cases)}] = {{")
    for idx, case in enumerate(cases):
        cc_lines.append(
            f'  {{"{case.name}", {int(case.label)}, {spec["image_symbol"]}{idx}}},'
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
        "--variant",
        choices=tuple(VARIANTS.keys()),
        default="cnn",
        help="Which int8 model variant to export test images for.",
    )
    args = parser.parse_args()
    export_int8_test_images(variant=args.variant)


if __name__ == "__main__":
    main()
