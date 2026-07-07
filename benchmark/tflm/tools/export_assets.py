#!/usr/bin/env python3
"""Export MNIST MLP/CNN TFLite models + embedded test images for benchmarks.

Writes under benchmark/tflm/generated/:
  mnist_mlp.tflite, mnist_test_images.{h,cc}
  mnist_cnn.tflite, mnist_cnn_test_images.{h,cc}
  mnist_cnn_int8.tflite, mnist_cnn_int8_model_data.{h,cc} (full int8 I/O)

Run from repo root:
  python3 benchmark/tflm/tools/export_assets.py
  python3 benchmark/tflm/tools/export_assets.py --model cnn
  python3 benchmark/tflm/tools/export_assets.py --model cnn-int8
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
GENERATED = BENCH / "generated"
NUM_IMAGES = 10
MNIST_DIGITS = tuple(range(10))
NUM_CALIBRATION = 128
IMG_H = 28
IMG_W = 28
IMG_C = 1


def _one_per_digit_cases(cases: list, *, num: int = NUM_IMAGES) -> list:
    """Keep one sample per MNIST digit (0–9) for benchmark vectors."""
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


MODELS = {
    "mlp": {
        "nk": ROOT / "models" / "mnist_mlp.nk",
        "onnx": ROOT / "models" / "mnist_mlp.onnx",
        "tflite": GENERATED / "mnist_mlp.tflite",
        "saved_model": GENERATED / "mlp_saved_model",
        "images_h": GENERATED / "mnist_test_images.h",
        "images_cc": GENERATED / "mnist_test_images.cc",
        "input_size": 784,
        "prefix": "Mnist",
        "array_prefix": "kMnist",
        "export_make": "export-mnist",
        "legacy_names": True,
    },
    "cnn": {
        "nk": ROOT / "models" / "mnist_cnn.nk",
        "onnx": ROOT / "models" / "mnist_cnn.onnx",
        "tflite": GENERATED / "mnist_cnn.tflite",
        "saved_model": GENERATED / "cnn_saved_model",
        "images_h": GENERATED / "mnist_cnn_test_images.h",
        "images_cc": GENERATED / "mnist_cnn_test_images.cc",
        "input_size": 784,
        "prefix": "MnistCnn",
        "array_prefix": "kMnistCnn",
        "export_make": "export-mnist-cnn",
        "legacy_names": False,
    },
    "cnn-int8": {
        "nk": ROOT / "models" / "mnist_cnn.nk",
        "onnx": ROOT / "models" / "mnist_cnn.onnx",
        "tflite": GENERATED / "mnist_cnn_int8.tflite",
        "saved_model": GENERATED / "cnn_saved_model",
        "images_h": GENERATED / "mnist_cnn_test_images.h",
        "images_cc": GENERATED / "mnist_cnn_test_images.cc",
        "model_data_h": GENERATED / "mnist_cnn_int8_model_data.h",
        "model_data_cc": GENERATED / "mnist_cnn_int8_model_data.cc",
        "model_array": "g_mnist_cnn_int8_model_data",
        "input_size": 784,
        "prefix": "MnistCnn",
        "array_prefix": "kMnistCnn",
        "export_make": "export-mnist-cnn",
        "legacy_names": False,
    },
}


def _import_reader():
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.reader import read_test_suite

    return read_test_suite


def _onnx2tf_bin() -> str:
    found = shutil.which("onnx2tf")
    if found:
        return found
    venv_bin = BENCH / ".venv" / "bin" / "onnx2tf"
    if venv_bin.is_file():
        return str(venv_bin)
    raise SystemExit(
        "onnx2tf not found. Install with:\n"
        "  python3 -m venv benchmark/tflm/.venv\n"
        "  benchmark/tflm/.venv/bin/pip install onnx onnx2tf"
    )


def _import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "tensorflow not found. Install export deps with:\n"
            "  python3 -m pip install -r benchmark/tflm/requirements-export.txt"
        ) from exc
    return tf


def _load_mnist_calibration(num_samples: int) -> np.ndarray:
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.datasets import load_mnist

    x_train, _, _, _ = load_mnist()
    if x_train.shape[0] < num_samples:
        raise RuntimeError(f"MNIST train set has {x_train.shape[0]} rows, need {num_samples}")
    return np.asarray(x_train[:num_samples], dtype=np.float32)


def _export_int8_tflite_from_saved_model(
    saved_model_dir: Path,
    out: Path,
    calibration: np.ndarray,
) -> None:
    tf = _import_tensorflow()

    def representative_dataset():
        for sample in calibration:
            yield [sample.reshape(1, IMG_H, IMG_W, IMG_C).astype(np.float32)]

    saved = tf.saved_model.load(str(saved_model_dir))

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, IMG_H, IMG_W, IMG_C], dtype=tf.float32),
        ]
    )
    def serving_fn(x):
        return saved(x)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving_fn.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print(f"Quantizing {saved_model_dir.name} to full int8 ...")
    tflite_model = converter.convert()
    out.write_bytes(tflite_model)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


def _write_tflite_model_arrays(spec: dict) -> None:
    sys.path.insert(0, str(TOOLS))
    from tflite_model_data import write_tflite_model_arrays

    write_tflite_model_arrays(
        spec["tflite"],
        out_h=spec["model_data_h"],
        out_cc=spec["model_data_cc"],
        array_name=spec["model_array"],
    )
    print(f"Wrote {spec['model_data_h'].name} and {spec['model_data_cc'].name}")


def _export_tflite_via_onnx2tf(onnx_path: Path, out: Path, saved_model_dir: Path) -> None:
    if not onnx_path.is_file():
        raise SystemExit(f"missing {onnx_path} — run: make {MODELS['mlp' if 'mlp' in onnx_path.name else 'cnn']['export_make']}")

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    print(f"Converting {onnx_path.name} with onnx2tf ...")
    subprocess.run(
        [_onnx2tf_bin(), "-i", str(onnx_path), "-o", str(saved_model_dir), "-nuo"],
        check=True,
    )

    candidates = sorted(saved_model_dir.glob("*float32.tflite"))
    if not candidates:
        raise SystemExit(f"onnx2tf did not emit a float32 .tflite under {saved_model_dir}")
    shutil.copy2(candidates[0], out)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


def _write_test_image_arrays(read_test_suite, spec: dict) -> None:
    suite = read_test_suite(spec["nk"])
    if suite is None:
        raise RuntimeError(f"missing TCAS section in {spec['nk']}")

    cases = _one_per_digit_cases(suite.cases, num=NUM_IMAGES)
    prefix = spec["prefix"]
    array_prefix = spec["array_prefix"]
    input_size = spec["input_size"]
    legacy = spec.get("legacy_names", False)

    count_name = "kMnistBenchmarkImageCount" if legacy else f"k{prefix}BenchmarkImageCount"
    size_name = "kMnistBenchmarkInputSize" if legacy else f"k{prefix}BenchmarkInputSize"
    sample_name = "MnistBenchmarkSample" if legacy else f"{prefix}BenchmarkSample"
    images_name = "kMnistBenchmarkImages" if legacy else f"k{prefix}BenchmarkImages"
    image_symbol = f"{array_prefix}Image" if legacy else f"{array_prefix}Image"

    hdr_lines = [
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        f"constexpr int {count_name} = {len(cases)};",
        f"constexpr int {size_name} = {input_size};",
        "",
        f"struct {sample_name} {{",
        "  const char* name;",
        "  int label;",
        "  const float* pixels;",
        "};",
        "",
        f"extern const {sample_name} {images_name}[{len(cases)}];",
        "",
    ]
    cc_lines = [f'#include "{spec["images_h"].name}"', ""]

    for idx, case in enumerate(cases):
        pixels = np.asarray(case.input, dtype=np.float32).reshape(-1)
        if pixels.size != input_size:
            raise ValueError(f"case {case.name} has {pixels.size} inputs, expected {input_size}")
        pixel_text = ", ".join(f"{v:.8f}f" for v in pixels)
        cc_lines.append(
            f"alignas(16) static const float {image_symbol}{idx}[{input_size}] = {{{pixel_text}}};"
        )
        cc_lines.append("")

    cc_lines.append(f"const {sample_name} {images_name}[{len(cases)}] = {{")
    for idx, case in enumerate(cases):
        cc_lines.append(
            f'  {{"{case.name}", {int(case.label)}, {image_symbol}{idx}}},'
        )
    cc_lines.append("};")
    cc_lines.append("")

    spec["images_h"].write_text("\n".join(hdr_lines) + "\n", encoding="utf-8")
    spec["images_cc"].write_text("\n".join(cc_lines) + "\n", encoding="utf-8")


def export_model(name: str, read_test_suite, *, images_only: bool) -> None:
    spec = MODELS[name]
    GENERATED.mkdir(parents=True, exist_ok=True)

    if not spec["nk"].is_file():
        raise SystemExit(f"missing {spec['nk']} — run: make {spec['export_make']}")

    if images_only:
        if not spec["tflite"].is_file():
            raise SystemExit(f"{spec['tflite']} missing — run export without --images-only")
        if name == "cnn-int8":
            subprocess.run(
                [sys.executable, str(TOOLS / "export_int8_test_images.py")],
                check=True,
                cwd=ROOT,
            )
            return
    elif name == "cnn-int8":
        if not spec["saved_model"].is_dir():
            print(f"SavedModel missing — converting {spec['onnx'].name} first ...")
            _export_tflite_via_onnx2tf(
                spec["onnx"],
                GENERATED / "mnist_cnn.tflite",
                spec["saved_model"],
            )
        calibration = _load_mnist_calibration(NUM_CALIBRATION)
        _export_int8_tflite_from_saved_model(spec["saved_model"], spec["tflite"], calibration)
        _write_tflite_model_arrays(spec)
    else:
        _export_tflite_via_onnx2tf(spec["onnx"], spec["tflite"], spec["saved_model"])

    _write_test_image_arrays(read_test_suite, spec)
    print(f"Wrote {spec['images_h'].name} and {spec['images_cc'].name}")

    if name == "cnn-int8":
        subprocess.run(
            [sys.executable, str(TOOLS / "export_int8_test_images.py")],
            check=True,
            cwd=ROOT,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("mlp", "cnn", "cnn-int8", "all"),
        default="all",
        help="Which model assets to export (default: all).",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only refresh embedded test image arrays (requires existing .tflite).",
    )
    args = parser.parse_args()

    read_test_suite = _import_reader()
    if args.model == "all":
        names = ("mlp", "cnn", "cnn-int8")
    else:
        names = (args.model,)
    for name in names:
        export_model(name, read_test_suite, images_only=args.images_only)


if __name__ == "__main__":
    main()
