#!/usr/bin/env bash
# Prefer an isolated TF venv when system TensorFlow is broken (mutex/segfault on import).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${NETKIT_TF_VENV:-/tmp/netkit-tf-venv}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
  "${VENV}/bin/pip" install -q 'tensorflow==2.16.2' pillow
fi
# onnx2tf + torch/timm live in the TFLM bench venv when present.
TFLM_VENV="${HERE}/../.venv"
export PATH="${TFLM_VENV}/bin:${PATH}"
export PYTHONPATH="${HERE}/../../python:${PYTHONPATH:-}"
exec "${VENV}/bin/python" "${HERE}/export_mobilenetv4_imagenet_int8.py" "$@"
