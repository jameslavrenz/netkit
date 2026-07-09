#!/usr/bin/env python3
"""Export prequantized int8 ImageNet MobileNetV4 benchmark images (224x224x3 NHWC).

Uses the same 10 ImageNet samples as the float benches. Quantization happens here
in Python (never in C++):

  --quant-source tflite  → TFLite / TFLM input scale/zp (default)
  --quant-source nk      → netkit .nk first-layer input scale/zp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
GENERATED = HERE.parent / "generated"
CACHE = GENERATED / "imagenet_sample_cache"
INPUT_SIZE = 224 * 224 * 3
NUM_IMAGES = 10

sys.path.insert(0, str(HERE))
from export_imagenet_mnv4_test_images import SAMPLES, _load_rgb, preprocess  # noqa: E402


def _tflite_input_quant(tflite_path: Path) -> tuple[float, int]:
    quant_json = tflite_path.with_name("mobilenetv4_imagenet_int8_tflite_quant.json")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant-source",
        choices=("tflite", "nk"),
        default="tflite",
        help="Where to read input scale/zero-point (Python-side quantize only)",
    )
    parser.add_argument(
        "--tflite",
        type=Path,
        default=GENERATED / "mobilenetv4_imagenet_int8.tflite",
    )
    parser.add_argument(
        "--nk",
        type=Path,
        default=ROOT / "models" / "mobilenetv4_imagenet_int8.nk",
    )
    parser.add_argument("--out-h", type=Path, default=None)
    parser.add_argument("--out-cc", type=Path, default=None)
    args = parser.parse_args()

    if args.out_h is None:
        name = (
            "imagenet_mnv4_netkit_int8_test_images"
            if args.quant_source == "nk"
            else "imagenet_mnv4_int8_test_images"
        )
        args.out_h = GENERATED / f"{name}.h"
    if args.out_cc is None:
        args.out_cc = args.out_h.with_suffix(".cc")

    sys.path.insert(0, str(ROOT / "python"))
    from netkit.quantize import quantize_float_input

    if args.quant_source == "tflite":
        if not args.tflite.is_file():
            raise SystemExit(
                f"missing {args.tflite} — run: make -C benchmark/tflm export-mobilenetv4-imagenet-int8"
            )
        input_scale, input_zp = _tflite_input_quant(args.tflite)
    else:
        if not args.nk.is_file():
            raise SystemExit(
                f"missing {args.nk} — run: python3 tools/write_mobilenetv4_imagenet_int8.py"
            )
        input_scale, input_zp = _nk_input_quant(args.nk)

    if len(SAMPLES) != NUM_IMAGES:
        raise SystemExit(f"expected {NUM_IMAGES} samples, got {len(SAMPLES)}")

    hdr_lines = [
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        f"constexpr int kImagenetMnv4Int8BenchmarkImageCount = {NUM_IMAGES};",
        f"constexpr int kImagenetMnv4Int8BenchmarkInputSize = {INPUT_SIZE};",
        "constexpr int kImagenetMnv4Int8BenchmarkHeight = 224;",
        "constexpr int kImagenetMnv4Int8BenchmarkWidth = 224;",
        "constexpr int kImagenetMnv4Int8BenchmarkChannels = 3;",
        "constexpr int kImagenetMnv4Int8BenchmarkNumClasses = 1000;",
        f"constexpr float kImagenetMnv4Int8BenchmarkInputScale = {input_scale:.8f}f;",
        f"constexpr int kImagenetMnv4Int8BenchmarkInputZeroPoint = {input_zp};",
        "",
        "struct ImagenetMnv4Int8BenchmarkSample {",
        "  const char* name;",
        "  int label;",
        "  const int8_t* pixels;",
        "};",
        "",
        f"extern const ImagenetMnv4Int8BenchmarkSample kImagenetMnv4Int8BenchmarkImages[{NUM_IMAGES}];",
        "",
    ]

    cc_lines = [f'#include "{args.out_h.name}"', ""]
    for idx, (filename, _url, label, short) in enumerate(SAMPLES):
        path = CACHE / filename
        if not path.is_file():
            raise SystemExit(
                f"missing {path} — run: make -C benchmark/tflm export-imagenet-mnv4-images"
            )
        pixels = preprocess(_load_rgb(path)).reshape(-1)
        pixels_i8 = quantize_float_input(pixels, input_scale, input_zp)
        i8_text = ", ".join(str(int(v)) for v in pixels_i8)
        cc_lines.append(
            f"alignas(16) static const int8_t kImagenetMnv4Int8Image{idx}[{INPUT_SIZE}] = {{{i8_text}}};"
        )
        cc_lines.append("")

    cc_lines.append(
        f"const ImagenetMnv4Int8BenchmarkSample kImagenetMnv4Int8BenchmarkImages[{NUM_IMAGES}] = {{"
    )
    for idx, (_filename, _url, label, short) in enumerate(SAMPLES):
        cc_lines.append(
            f'  {{"ImageNet {short} (class {label})", {int(label)}, kImagenetMnv4Int8Image{idx}}},'
        )
    cc_lines.append("};")
    cc_lines.append("")

    args.out_h.parent.mkdir(parents=True, exist_ok=True)
    args.out_h.write_text("\n".join(hdr_lines), encoding="utf-8")
    args.out_cc.write_text("\n".join(cc_lines), encoding="utf-8")
    print(
        f"Wrote {args.out_h} and {args.out_cc} "
        f"(source={args.quant_source}, input_scale={input_scale}, zp={input_zp})"
    )


if __name__ == "__main__":
    main()
