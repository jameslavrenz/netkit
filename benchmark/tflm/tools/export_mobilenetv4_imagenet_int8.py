#!/usr/bin/env python3
"""Export full int8 MobileNetV4-Conv-Small ImageNet .tflite (224x224, 1000 classes).

Loads the same timm checkpoint as the float ImageNet export, converts ONNX→TF via
onnx2tf, then applies TFLite full-integer PTQ calibrated on the 10 ImageNet bench
samples.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GENERATED = HERE.parent / "generated"
CACHE = GENERATED / "imagenet_sample_cache"
PYTHON_DIR = ROOT / "python"
OUT = GENERATED / "mobilenetv4_imagenet_int8.tflite"

sys.path.insert(0, str(HERE))
from export_imagenet_mnv4_test_images import SAMPLES, _load_rgb, preprocess  # noqa: E402


def _onnx2tf_bin() -> str:
    found = shutil.which("onnx2tf")
    if found:
        return found
    venv_bin = HERE.parent / ".venv" / "bin" / "onnx2tf"
    if venv_bin.is_file():
        return str(venv_bin)
    raise SystemExit(
        "onnx2tf not found. Install with:\n"
        "  python3 -m venv benchmark/tflm/.venv && "
        "benchmark/tflm/.venv/bin/pip install onnx onnx2tf"
    )


def calibration_batches() -> list[np.ndarray]:
    batches = []
    for filename, _url, _label, _short in SAMPLES:
        path = CACHE / filename
        if not path.is_file():
            raise SystemExit(
                f"missing {path} — run: make -C benchmark/tflm export-imagenet-mnv4-images"
            )
        batches.append(preprocess(_load_rgb(path)).reshape(1, 224, 224, 3).astype(np.float32))
    return batches


def _find_saved_model(root: Path) -> Path | None:
    if (root / "saved_model.pb").is_file() or (root / "variables").is_dir():
        return root
    for child in root.rglob("saved_model.pb"):
        return child.parent
    return None


def main() -> None:
    import tensorflow as tf

    sys.path.insert(0, str(PYTHON_DIR))
    from netkit.torch_backbone_pack import load_backbone_model

    import torch

    batches = calibration_batches()

    def representative_dataset():
        for batch in batches:
            yield [batch]

    model = load_backbone_model("mobilenetv4_small", num_classes=1000, pretrained=True)
    model.eval()

    with tempfile.TemporaryDirectory(prefix="mnv4_imagenet_int8_") as tmp:
        tmp_path = Path(tmp)
        onnx_path = tmp_path / "mobilenetv4_imagenet.onnx"
        tf_out = tmp_path / "tf_out"
        dummy = torch.randn(1, 3, 224, 224)
        print("exporting ONNX ...")
        try:
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["logits"],
                opset_version=17,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["logits"],
                opset_version=17,
            )

        # onnx2tf downloads a calibration .npy into CWD; a corrupt/cached copy
        # raises allow_pickle=False. Seed a valid float32 array first.
        cal_npy = Path.cwd() / "calibration_image_sample_data_20x128x128x3_float32.npy"
        if not cal_npy.is_file() or cal_npy.stat().st_size < 1000:
            np.save(cal_npy, np.zeros((20, 128, 128, 3), dtype=np.float32))

        print(f"converting with onnx2tf -> {tf_out} ...")
        subprocess.run(
            [_onnx2tf_bin(), "-i", str(onnx_path), "-o", str(tf_out), "-nuo"],
            check=True,
        )

        saved = _find_saved_model(tf_out)
        if saved is None:
            raise SystemExit(f"onnx2tf did not emit a SavedModel under {tf_out}")

        print(f"PTQ from SavedModel {saved} ...")
        loaded = tf.saved_model.load(str(saved))
        # onnx2tf SavedModels often have empty .signatures; wrap __call__.
        if loaded.signatures:
            concrete = loaded.signatures[next(iter(loaded.signatures))]
            converter = tf.lite.TFLiteConverter.from_concrete_functions(
                [concrete], loaded
            )
        else:
            @tf.function(
                input_signature=[
                    tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)
                ]
            )
            def serving_fn(x):
                out = loaded(x)
                if isinstance(out, (list, tuple)):
                    return out[0]
                if isinstance(out, dict):
                    return next(iter(out.values()))
                return out

            converter = tf.lite.TFLiteConverter.from_concrete_functions(
                [serving_fn.get_concrete_function()], loaded
            )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.experimental_new_quantizer = False
        tfl = converter.convert()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(tfl)
    print(f"Wrote {OUT} ({len(tfl)} bytes)")

    interp = tf.lite.Interpreter(model_content=tfl)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print("input:", inp["shape"], inp["dtype"], inp["quantization"])
    print("output:", out["shape"], out["dtype"], out["quantization"])


if __name__ == "__main__":
    main()
