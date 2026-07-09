#!/usr/bin/env python3
"""Train or quantize MNIST MLP to int8 and export models/mnist_mlp_int8.nk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit import RegressionSuite
from netkit.datasets import load_mnist
from netkit.quantize import forward_quantized_mlp, quantize_float_input, quantize_mlp, quantized_mlp_to_spec
from netkit.reader import read_nk
from netkit.torch_models import TutorialMlp784
from netkit.torch_pack import forward_mlp_netkit, pack_tutorial_mlp
from netkit.torch_train import train_classifier
from netkit.writer import RegressionCase, write_nk_bytes

from netkit.tflite_quant_align import load_tflite_quant_json, write_tflite_quant_json

MODELS = ROOT / "models"
DEFAULT_FLOAT_NK = MODELS / "mnist_mlp.nk"
OUT_PATH = MODELS / "mnist_mlp_int8.nk"
DEFAULT_TFLITE = ROOT / "benchmark" / "tflm" / "generated" / "mnist_mlp_int8.tflite"
DEFAULT_TFLITE_QUANT_JSON = ROOT / "benchmark" / "tflm" / "generated" / "mnist_mlp_int8_tflite_quant.json"

EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42
NUM_CASES = 10
NUM_CALIBRATION = 512

ARCH = {
    "network": "mlp",
    "input": [1, 784],
    "layers": [
        {"type": "dense", "units": 128, "activation": "relu"},
        {"type": "dense", "units": 10, "activation": "softmax"},
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MNIST MLP int8 .nk")
    parser.add_argument(
        "--from-nk",
        type=Path,
        default=DEFAULT_FLOAT_NK,
        help="Float .nk weights source (default: models/mnist_mlp.nk)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train MLP from scratch instead of reusing float .nk weights",
    )
    parser.add_argument(
        "--align-tflite",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Align per-layer quant params to TFLite (default: {DEFAULT_TFLITE} if present)",
    )
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_mnist()

    if args.retrain:
        print(
            f"Training MNIST MLP on {x_train.shape[0]} images "
            f"(PyTorch Adam lr={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={EPOCHS}) ..."
        )
        model = TutorialMlp784()
        train_classifier(
            model,
            x_train,
            y_train,
            forward_logits=model.forward_logits,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            seed=SEED,
        )
        model.eval()
        float_weights = pack_tutorial_mlp(model)
    else:
        if not args.from_nk.is_file():
            raise SystemExit(f"missing {args.from_nk} — run: make export-mnist")
        _, float_weights = read_nk(args.from_nk)
        float_weights = np.asarray(float_weights, dtype=np.float32)
        print(f"Quantizing weights from {args.from_nk.name} (no retrain) ...")

    float_probs = None
    if args.retrain:
        float_probs = forward_mlp_netkit(model, x_test)
        float_acc = (float_probs.argmax(axis=1) == y_test).mean()
        print(f"Float test accuracy: {float_acc * 100:.2f}%")

    aligned_quants = None
    tflite_path = args.align_tflite
    if tflite_path is None and DEFAULT_TFLITE.is_file():
        tflite_path = DEFAULT_TFLITE
    if tflite_path is not None and tflite_path.is_file():
        json_out = DEFAULT_TFLITE_QUANT_JSON
        if json_out.is_file():
            aligned_quants = load_tflite_quant_json(json_out)
            print(f"Aligned quant params from {json_out} ({len(aligned_quants)} layers)")
        else:
            aligned_quants = write_tflite_quant_json(tflite_path, json_out)
            print(f"Aligned quant params from {tflite_path.name} ({len(aligned_quants)} layers)")

    pack = quantize_mlp(
        ARCH,
        float_weights,
        x_train,
        num_calibration=NUM_CALIBRATION,
        aligned_quants=aligned_quants,
    )

    quant_probs = np.stack(
        [forward_quantized_mlp(x, ARCH, pack, output_float=True) for x in x_test],
        axis=0,
    )
    quant_acc = (quant_probs.argmax(axis=1) == y_test).mean()
    print(f"Quantized test accuracy: {quant_acc * 100:.2f}%")

    cases: list[RegressionCase] = []
    used_digits: set[int] = set()
    input_scale = pack.quant_layers[0].input_scale
    input_zp = pack.quant_layers[0].input_zero_point
    for i in range(x_test.shape[0]):
        label = int(y_test[i])
        if label in used_digits:
            continue
        probs = forward_quantized_mlp(x_test[i], ARCH, pack, output_float=True)
        # TCAS stores floats; embed prequantized int8 as float values in [-128, 127].
        input_i8 = quantize_float_input(x_test[i], input_scale, input_zp)
        cases.append(
            RegressionCase(
                name=f"MNIST digit {label} (test idx {i})",
                input=input_i8.astype(np.float32),
                expected=probs,
                label=label,
            )
        )
        used_digits.add(label)
        if len(cases) >= NUM_CASES:
            break
    if len(cases) < NUM_CASES:
        raise RuntimeError(f"only found {len(cases)} digit cases for embedded TCAS")

    spec = quantized_mlp_to_spec(ARCH, pack)
    spec.tests = RegressionSuite(tolerance=0.05, cases=cases)

    MODELS.mkdir(parents=True, exist_ok=True)
    out_path = MODELS / "mnist_mlp_int8.nk"
    blob = write_nk_bytes(spec)
    out_path.write_bytes(blob)
    print(f"Wrote {out_path} ({len(blob)} bytes, {len(cases)} embedded test cases)")


if __name__ == "__main__":
    main()
