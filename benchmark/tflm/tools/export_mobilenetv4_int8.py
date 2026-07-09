#!/usr/bin/env python3
"""Export full int8 MobileNetV4-Conv-Small .tflite matching netkit mobilenetv4_small_int8.nk.

Uses the same Keras graph as export_mobilenetv4.py and calibrates with TCAS inputs from
models/mobilenetv4_small.nk (upsampled 28x28x1 -> 56x56x3).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
sys.path.insert(0, str(TOOLS))
from export_mobilenetv4 import build  # noqa: E402

sys.path.insert(0, str(ROOT / "python"))
from netkit.reader import read_test_suite  # noqa: E402


def upsample_mnist_to_56x56x3(src28: np.ndarray) -> np.ndarray:
    src28 = np.asarray(src28, dtype=np.float32).reshape(28, 28)
    out = np.empty((56, 56, 3), dtype=np.float32)
    for y in range(56):
        for x in range(56):
            v = src28[y // 2, x // 2]
            out[y, x, :] = v
    return out


def calibration_samples(nk_path: Path) -> np.ndarray:
    suite = read_test_suite(nk_path)
    if suite is None or not suite.cases:
        raise RuntimeError(f"missing TCAS in {nk_path}")
    samples = []
    for case in suite.cases:
        inp = np.asarray(case.input, dtype=np.float32)
        if inp.size == 28 * 28:
            samples.append(upsample_mnist_to_56x56x3(inp))
        elif inp.size == 56 * 56 * 3:
            samples.append(inp.reshape(56, 56, 3))
        else:
            raise ValueError(f"unexpected input size {inp.size} in {case.name}")
    return np.stack(samples, axis=0)


def main() -> None:
    here = TOOLS
    gen = here.parent / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    nk = ROOT / "models" / "mobilenetv4_small.nk"

    model = build()
    cal = calibration_samples(nk)

    def representative_dataset():
        for sample in cal:
            yield [sample.reshape(1, 56, 56, 3).astype(np.float32)]

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, 56, 56, 3], dtype=tf.float32)],
    )
    def serving_fn(x):
        return model(x, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving_fn.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.experimental_new_quantizer = False

    tfl = converter.convert()
    out_path = gen / "mobilenetv4_small_int8.tflite"
    out_path.write_bytes(tfl)
    print(f"wrote {out_path} ({len(tfl)} bytes)")

    from tflite_model_data import write_tflite_model_arrays

    write_tflite_model_arrays(
        out_path,
        out_h=gen / "mobilenetv4_small_int8_model_data.h",
        out_cc=gen / "mobilenetv4_small_int8_model_data.cc",
        array_name="g_mobilenetv4_small_int8_model_data",
    )

    interp = tf.lite.Interpreter(model_content=tfl)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print("input:", inp["shape"], inp["dtype"], inp["quantization"])
    print("output:", out["shape"], out["dtype"], out["quantization"])
    ops = sorted({d["op_name"] for d in interp._get_ops_details()})
    print("ops:", ops)


if __name__ == "__main__":
    main()
