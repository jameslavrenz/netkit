#!/usr/bin/env python3
"""Quantize MNIST CNN to int8 and export models/mnist_cnn_int8.nk.

Default: reuse weights from models/mnist_cnn.nk (no training — seconds).
Use --retrain to train from scratch (~minutes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit import RegressionSuite
from netkit.datasets import load_mnist
from netkit.quantize import forward_quantized_cnn, quantize_cnn, quantize_float_input, quantized_cnn_to_spec
from netkit.reader import read_nk
from netkit.tflite_quant_align import (
    extract_tflite_cnn_quant_specs,
    load_tflite_quant_json,
    write_tflite_quant_json,
)
from netkit.torch_models import TutorialCnn28
from netkit.torch_pack import forward_cnn_netkit, pack_tutorial_cnn
from netkit.torch_train import train_cnn_classifier
from netkit.writer import RegressionCase, write_nk_bytes

MODELS = ROOT / "models"
DEFAULT_FLOAT_NK = MODELS / "mnist_cnn.nk"
DEFAULT_TFLITE = ROOT / "benchmark" / "tflm" / "generated" / "mnist_cnn_int8.tflite"
DEFAULT_TFLITE_QUANT_JSON = ROOT / "benchmark" / "tflm" / "generated" / "mnist_cnn_int8_tflite_quant.json"
OUT_PATH = MODELS / "mnist_cnn_int8.nk"

IMG_H = 28
IMG_W = 28
IMG_C = 1
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42
NUM_CASES = 10
NUM_CALIBRATION = 128
NUM_CALIBRATION_FAST = 64
ACCURACY_SAMPLES = 256


def _predict_quantized_batch(
    x: np.ndarray,
    pack,
) -> np.ndarray:
    """Run forward_quantized_cnn on one row or many rows."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return forward_quantized_cnn(x, ARCH, pack)
    return np.stack([forward_quantized_cnn(row, ARCH, pack) for row in x], axis=0)


def _select_digit_cases_fast(
    x_test: np.ndarray,
    y_test: np.ndarray,
    pack,
    *,
    num_cases: int,
    name_fmt: str,
) -> list:
    """Pick regression cases without a full-test forward pass."""
    cases = []
    used_digits: set[int] = set()
    for i in range(x_test.shape[0]):
        probs = forward_quantized_cnn(x_test[i], ARCH, pack, output_float=True)
        pred = int(np.argmax(forward_quantized_cnn(x_test[i], ARCH, pack)))
        label = int(y_test[i])
        if pred != label or label in used_digits:
            continue
        # TCAS stores floats; embed prequantized int8 as float values in [-128, 127].
        input_i8 = quantize_float_input(
            x_test[i].reshape(-1), pack.quant_layers[0].input_scale, pack.quant_layers[0].input_zero_point
        )
        cases.append(
            RegressionCase(
                name=name_fmt.format(digit=label, i=i),
                input=input_i8.astype(np.float32),
                expected=probs,
                label=label,
            )
        )
        used_digits.add(label)
        if len(cases) >= num_cases:
            break
    if len(cases) < num_cases:
        raise RuntimeError(f"found only {len(cases)}/{num_cases} digit cases — check model weights")
    return cases

ARCH = {
    "network": "cnn",
    "input": [IMG_H, IMG_W, IMG_C],
    "layers": [
        {"type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 32, "activation": "relu"},
        {"type": "max_pool2d", "pool_size": 2, "stride": 2},
        {"type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 64, "activation": "relu"},
        {"type": "max_pool2d", "pool_size": 2, "stride": 2},
        {"type": "flatten"},
        {"type": "dense", "units": 128, "activation": "relu"},
        {"type": "dense", "units": 10, "activation": "softmax"},
    ],
}


def _load_float_weights(from_nk: Path) -> np.ndarray:
    arch, weights = read_nk(from_nk)
    if arch.get("network") != "cnn":
        raise ValueError(f"{from_nk} is not a CNN model")
    return np.asarray(weights, dtype=np.float32)


def _train_float_weights(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    print(f"Training MNIST CNN ({EPOCHS} epochs) ...")
    model = TutorialCnn28()
    train_cnn_classifier(
        model,
        x_train,
        y_train,
        forward_logits=model.forward_logits,
        img_h=IMG_H,
        img_w=IMG_W,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED,
    )
    model.eval()
    return pack_tutorial_cnn(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MNIST CNN int8 .nk")
    parser.add_argument(
        "--from-nk",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Float weights source (default: {DEFAULT_FLOAT_NK} if it exists)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train CNN from scratch instead of reusing float .nk weights",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help=f"Skip if {OUT_PATH.name} already exists",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fewer calibration samples, skip full-test accuracy (seconds)",
    )
    parser.add_argument("-o", "--output", type=Path, default=OUT_PATH, help="Output .nk path")
    parser.add_argument(
        "--align-tflite",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Use TFLite per-layer quant params (default: {DEFAULT_TFLITE} if present)",
    )
    parser.add_argument(
        "--write-tflite-quant-json",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Write extracted TFLite quant JSON (default: {DEFAULT_TFLITE_QUANT_JSON} when aligning)",
    )
    args = parser.parse_args()

    if args.skip_if_exists and args.output.is_file():
        print(f"Skip: {args.output} already exists ({args.output.stat().st_size} bytes)")
        return

    x_train, y_train, x_test, y_test = load_mnist()

    if args.retrain:
        float_weights = _train_float_weights(x_train, y_train)
    else:
        src = args.from_nk
        if src is None and DEFAULT_FLOAT_NK.is_file():
            src = DEFAULT_FLOAT_NK
        if src is None or not src.is_file():
            print(
                f"No float weights at {DEFAULT_FLOAT_NK}. "
                "Run: make export-mnist-cnn   OR   use --retrain",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Loading float weights from {src} ...")
        float_weights = _load_float_weights(src)

    print("Quantizing to int8 ...")
    cal_samples = NUM_CALIBRATION_FAST if args.fast else NUM_CALIBRATION
    aligned_quants = None
    tflite_path = args.align_tflite
    if tflite_path is None and DEFAULT_TFLITE.is_file():
        tflite_path = DEFAULT_TFLITE
    if tflite_path is not None:
        if not tflite_path.is_file():
            print(f"TFLite model not found: {tflite_path}", file=sys.stderr)
            sys.exit(1)
        json_out = args.write_tflite_quant_json
        if json_out is None:
            json_out = DEFAULT_TFLITE_QUANT_JSON
        if json_out.is_file():
            aligned_quants = load_tflite_quant_json(json_out)
            print(f"Aligned quant params from {json_out} ({len(aligned_quants)} layers)")
        else:
            aligned_quants = write_tflite_quant_json(tflite_path, json_out)
            print(f"Aligned quant params from {tflite_path} ({len(aligned_quants)} layers)")
            print(f"Wrote {json_out}")
    elif args.align_tflite is not None:
        print(f"TFLite model not found: {args.align_tflite}", file=sys.stderr)
        sys.exit(1)

    pack = quantize_cnn(
        ARCH,
        float_weights,
        x_train,
        num_calibration=cal_samples,
        aligned_quants=aligned_quants,
    )

    if args.fast:
        check_x = x_test[:ACCURACY_SAMPLES]
        check_y = y_test[:ACCURACY_SAMPLES]
    else:
        check_x = x_test
        check_y = y_test
    quant_probs = _predict_quantized_batch(check_x, pack)
    print(
        f"Quantized accuracy ({check_x.shape[0]} samples): "
        f"{(quant_probs.argmax(axis=1) == check_y).mean() * 100:.2f}%"
    )

    cases = _select_digit_cases_fast(
        x_test,
        y_test,
        pack,
        num_cases=NUM_CASES,
        name_fmt="MNIST CNN digit {digit} (test idx {i})",
    )

    spec = quantized_cnn_to_spec(ARCH, pack)
    spec.tests = RegressionSuite(tolerance=0.08, cases=cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    blob = write_nk_bytes(spec)
    args.output.write_bytes(blob)
    print(f"Wrote {args.output} ({len(blob)} bytes, {len(cases)} embedded test cases)")


if __name__ == "__main__":
    main()
