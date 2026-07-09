#!/usr/bin/env python3
"""Export full int8 MNIST MLP .tflite + embedded arrays from models/mnist_mlp.nk weights."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parents[1]
GENERATED = BENCH / "generated"
TOOLS = Path(__file__).resolve().parent

NUM_CALIBRATION = 128
IMG_FLAT = 784


def _import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "tensorflow not found — use /tmp/netkit-tf-venv or pip install tensorflow==2.16"
        ) from exc
    return tf


def _load_mnist_calibration(num_samples: int) -> np.ndarray:
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.datasets import load_mnist

    x_train, _, _, _ = load_mnist()
    return np.asarray(x_train[:num_samples], dtype=np.float32).reshape(-1, IMG_FLAT)


def _load_keras_weights(nk_path: Path) -> tuple[list[int], list[np.ndarray]]:
    sys.path.insert(0, str(ROOT / "python"))
    from netkit.reader import read_nk

    arch, weights = read_nk(nk_path)
    if arch.get("network") != "mlp":
        raise ValueError(f"{nk_path} is not an MLP model")
    w = np.asarray(weights, dtype=np.float32)
    # TutorialMlp784: W1[128,784], b1[128], W2[10,128], b2[10]
    w1 = w[: IMG_FLAT * 128].reshape(128, IMG_FLAT).T
    b1 = w[IMG_FLAT * 128 : IMG_FLAT * 128 + 128]
    off = IMG_FLAT * 128 + 128
    w2 = w[off : off + 128 * 10].reshape(10, 128).T
    b2 = w[off + 128 * 10 : off + 128 * 10 + 10]
    return [128, 10], [w1, b1, w2, b2]


def export_mnist_mlp_int8_tflite(
    *,
    nk_path: Path = ROOT / "models" / "mnist_mlp.nk",
    out_tflite: Path = GENERATED / "mnist_mlp_int8.tflite",
) -> None:
    tf = _import_tensorflow()
    _, weight_arrays = _load_keras_weights(nk_path)
    calibration = _load_mnist_calibration(NUM_CALIBRATION)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_FLAT,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.set_weights(weight_arrays)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, IMG_FLAT], dtype=tf.float32)],
    )
    def serving_fn(x):
        return model(x, training=False)

    def representative_dataset():
        for sample in calibration:
            yield [sample.reshape(1, IMG_FLAT).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving_fn.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.experimental_new_quantizer = False

    print("Quantizing MNIST MLP keras model to full int8 ...")
    tflite_model = converter.convert()
    out_tflite.parent.mkdir(parents=True, exist_ok=True)
    out_tflite.write_bytes(tflite_model)
    print(f"Wrote {out_tflite} ({out_tflite.stat().st_size} bytes)")

    sys.path.insert(0, str(TOOLS))
    from tflite_model_data import write_tflite_model_arrays

    write_tflite_model_arrays(
        out_tflite,
        out_h=GENERATED / "mnist_mlp_int8_model_data.h",
        out_cc=GENERATED / "mnist_mlp_int8_model_data.cc",
        array_name="g_mnist_mlp_int8_model_data",
    )

    sys.path.insert(0, str(ROOT / "python"))
    from netkit.tflite_quant_align import write_tflite_quant_json

    write_tflite_quant_json(out_tflite, GENERATED / "mnist_mlp_int8_tflite_quant.json")


def main() -> None:
    nk = ROOT / "models" / "mnist_mlp.nk"
    if not nk.is_file():
        raise SystemExit(f"missing {nk} — run: make export-mnist")
    export_mnist_mlp_int8_tflite(nk_path=nk)


if __name__ == "__main__":
    main()
