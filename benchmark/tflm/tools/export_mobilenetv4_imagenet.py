#!/usr/bin/env python3
"""Export pretrained MobileNetV4-Conv-Small float32 .tflite for the ImageNet host bench.

Loads the same timm checkpoint used by `python -m netkit pack --pretrained`, exports
ONNX (NCHW), converts with onnx2tf to float32 TFLite (NHWC), and writes the .tflite
under benchmark/tflm/generated/.

Requires: torch, timm, onnx, and onnx2tf (benchmark/tflm/.venv recommended).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GENERATED = Path(__file__).resolve().parents[1] / "generated"
PYTHON_DIR = ROOT / "python"


def _onnx2tf_bin() -> str:
    found = shutil.which("onnx2tf")
    if found:
        return found
    venv_bin = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "onnx2tf"
    if venv_bin.is_file():
        return str(venv_bin)
    raise SystemExit(
        "onnx2tf not found. Install with:\n"
        "  python3 -m venv benchmark/tflm/.venv && "
        "benchmark/tflm/.venv/bin/pip install onnx onnx2tf"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument(
        "--out",
        type=Path,
        default=GENERATED / "mobilenetv4_imagenet_f32.tflite",
    )
    parser.add_argument(
        "--saved-model-dir",
        type=Path,
        default=GENERATED / "mobilenetv4_imagenet_saved_model",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(PYTHON_DIR))
    from netkit.torch_backbone_pack import load_backbone_model

    import torch

    model = load_backbone_model(
        "mobilenetv4_small",
        num_classes=args.num_classes,
        pretrained=True,
    )
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mnv4_imagenet_onnx_") as tmp:
        onnx_path = Path(tmp) / "mobilenetv4_imagenet.onnx"
        dummy = torch.randn(1, 3, args.height, args.width)
        print(f"exporting ONNX {onnx_path.name} ({args.height}x{args.width}, {args.num_classes} classes) ...")
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

        if args.saved_model_dir.exists():
            shutil.rmtree(args.saved_model_dir)
        print(f"converting with onnx2tf -> {args.saved_model_dir} ...")
        subprocess.run(
            [_onnx2tf_bin(), "-i", str(onnx_path), "-o", str(args.saved_model_dir), "-nuo"],
            check=True,
        )

    candidates = sorted(args.saved_model_dir.glob("*float32.tflite"))
    if not candidates:
        raise SystemExit(f"onnx2tf did not emit a float32 .tflite under {args.saved_model_dir}")
    shutil.copy2(candidates[0], args.out)
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")

    # Sanity: allocate and print shapes (needs tensorflow / tflite runtime).
    try:
        import tensorflow as tf

        interp = tf.lite.Interpreter(model_path=str(args.out))
        interp.allocate_tensors()
        print("input:", interp.get_input_details()[0]["shape"], interp.get_input_details()[0]["dtype"])
        print("output:", interp.get_output_details()[0]["shape"])
        ops = sorted({d["op_name"] for d in interp._get_ops_details()})
        print("ops:", ops)
    except Exception as exc:  # noqa: BLE001
        print(f"(skip tflite inspect: {exc})")


if __name__ == "__main__":
    main()
