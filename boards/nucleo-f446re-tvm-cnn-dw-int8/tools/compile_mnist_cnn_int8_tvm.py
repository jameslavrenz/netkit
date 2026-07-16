#!/usr/bin/env python3
"""Compile mnist_cnn_int8.tflite to microTVM AOT MLF (CMSIS-NN + C, Cortex-M4).

Requires a Relay-era TVM build with USE_MICRO=ON and USE_CMSISNN=ON
(e.g. apache-tvm v0.14). Point at it with TVM_HOME or PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path


def _ensure_tvm() -> None:
    tvm_home = os.environ.get("TVM_HOME")
    if tvm_home:
        py = Path(tvm_home) / "python"
        if py.is_dir() and str(py) not in sys.path:
            sys.path.insert(0, str(py))
    try:
        import tvm  # noqa: F401
        from tvm.relay.op.contrib import cmsisnn  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Need Relay-era TVM with CMSIS-NN (e.g. v0.14 built with "
            "USE_MICRO=ON USE_CMSISNN=ON). Set TVM_HOME to that tree.\n"
            f"Import error: {exc}"
        ) from exc


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        type=Path,
        default=root / "benchmark/tflm/generated/mnist_cnn_int8.tflite",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "generated" / "mlf",
    )
    ap.add_argument("--module-name", default="mnist_cnn_int8")
    ap.add_argument(
        "--no-cmsisnn",
        action="store_true",
        help="Skip CMSIS-NN BYOC; emit pure C AOT (reference/fair A/B).",
    )
    args = ap.parse_args()

    _ensure_tvm()
    from tvm import relay
    from tvm.micro import export_model_library_format
    from tvm.relay.backend import Executor, Runtime
    if not args.no_cmsisnn:
        from tvm.relay.op.contrib import cmsisnn

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"missing model: {model_path}")

    tflite_model_buf = model_path.read_bytes()
    try:
        import tflite
    except ImportError:
        # Older TVM accepts flatbuffer bytes via relay frontend helper.
        tflite = None  # noqa: F841

    try:
        from tvm.relay.frontend import from_tflite
    except ImportError:
        from tvm import relay as _relay

        from_tflite = _relay.frontend.from_tflite

    import tflite as tflite_mod

    model_obj = tflite_mod.Model.GetRootAsModel(tflite_model_buf, 0)
    subgraph = model_obj.Subgraphs(0)
    in_tensor = subgraph.Tensors(subgraph.Inputs(0))
    in_name = in_tensor.Name().decode() if isinstance(in_tensor.Name(), bytes) else in_tensor.Name()
    in_shape = tuple(int(x) for x in in_tensor.ShapeAsNumpy())
    shape_dict = {in_name: in_shape}
    dtype_dict = {in_name: "int8"}
    print(f"tflite input: {in_name} shape={in_shape}")

    try:
        mod, params = from_tflite(
            model_obj, shape_dict=shape_dict, dtype_dict=dtype_dict
        )
    except TypeError:
        mod, params = from_tflite(
            tflite_model_buf, shape_dict=shape_dict, dtype_dict=dtype_dict
        )

    import tvm

    runtime = Runtime("crt")
    executor = Executor(
        "aot",
        {
            "interface-api": "c",
            "unpacked-api": True,
            "link-params": True,
        },
    )
    target = tvm.target.Target("c -keys=arm_cpu -mcpu=cortex-m4")
    pass_cfg = {
        "tir.disable_vectorize": True,
        "tir.usmp.enable": True,
        "tir.usmp.algorithm": "hill_climb",
        "tir.disable_storage_rewrite": True,
    }

    build_mod = mod
    if args.no_cmsisnn:
        print("CMSIS-NN partition: skipped (--no-cmsisnn)")
        backend_label = "c"
        comps: list[str] = []
    else:
        # Prefer CMSIS-NN BYOC. Per-channel qnn.dense is left on the C path
        # (TVM 0.14 CMSIS-NN FC expects scalar scales; see cmsisnn.py check).
        build_mod = cmsisnn.partition_for_cmsisnn(mod, params, mcpu="cortex-m4")
        print("CMSIS-NN partition: ok")
        backend_label = "cmsis-nn+c"
        import re

        comps = sorted(set(re.findall(r'Composite="([^"]+)"', build_mod.astext(show_meta_data=False))))
    with tvm.transform.PassContext(opt_level=3, config=pass_cfg):
        factory = relay.build(
            build_mod,
            target=target,
            params=params,
            runtime=runtime,
            executor=executor,
            mod_name=args.module_name,
        )
    print(f"backend: {backend_label}")
    print("cmsis composites:", ", ".join(comps) if comps else "(none)")

    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    tar_path = out_dir.parent / f"{args.module_name}_mlf.tar"
    if tar_path.exists():
        tar_path.unlink()
    export_model_library_format(factory, str(tar_path))
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(out_dir)

    print(f"wrote MLF tar: {tar_path}")
    print(f"extracted to: {out_dir}")
    hdr = next(out_dir.glob("codegen/host/include/tvmgen_*.h"), None)
    if hdr:
        print(f"header: {hdr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
