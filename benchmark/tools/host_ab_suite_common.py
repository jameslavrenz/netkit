"""Shared host A/B suite harness (netkit vs TF Lite).

Fairness policy (cold-start / rebuild bias removed):
  1. Pre-build every netkit binary before any timed run.
  2. For every timed slot: run the exact command twice — discard the first
     process (still warming I-cache / runtime), keep only the second.
  3. Before each order pass, also warm the opposite runtime so the upcoming
     first-to-run side is hot when its kept (2nd) process starts.
  4. Timed metrics come from already-built binaries / Python benches only.
  5. Order swaps (netkit→TF Lite, then TF Lite→netkit).
  6. Equalize MNIST MLP batch timing (1000 invokes/window × 10 passes both sides).
  7. LiteRT-matched -O3 flags for netkit (BENCH_FLAG_PROFILE=tflite).
  8. Within each kept process, benches never report cold inference:
     MNIST CNN discards run 0 and image 0 of every run; MLP uses batched windows
     (discard batch pass 0); ImageNet warm_mean discards the entire first image pass.

Flash/RAM (MCU-style): netkit bench ELF TEXT/DATA minus hard-coded test-image
`.o` fixtures; TF Lite = core LiteRT CPU libs. Models (`.nk` / `.tflite`) and
fixture images are excluded — production would not embed those vectors.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
NETKIT = ROOT / "benchmark" / "netkit"
TFLITE = ROOT / "benchmark" / "tflite"
TFLITE_PY = TFLITE / ".venv" / "bin" / "python"
SUMMARY_RE = re.compile(r"^BENCHMARK_SUMMARY\s+(.*)$", re.M)

MLP_BATCH_INVOKES = 1000
MLP_BATCH_PASSES = 10
CNN_RUNS = 10

MODEL_CHOICES = ("mlp", "cnn", "cnn_dw", "imagenet")


@dataclass(frozen=True)
class DtypeProfile:
    """Per-dtype paths and make/target wiring."""

    name: str  # "float32" | "int8"
    tag: str  # short tag for binary/objdir names: "f32" | "i8"
    header_label: str  # printed in report banner


@dataclass
class SuiteCfg:
    dtype: DtypeProfile

    @property
    def suite_tag(self) -> str:
        return self.dtype.tag


@dataclass
class RunResult:
    runtime: str
    model: str
    xnnpack: bool
    order: str
    metric_name: str
    metric_us: float
    raw_summary: str
    elapsed_s: float


@dataclass
class PairAvg:
    model: str
    xnnpack: bool
    netkit_us: float
    tflite_us: float
    metric_name: str
    netkit_flash_bytes: int = 0
    tflite_flash_bytes: int = 0
    netkit_ram_bytes: int = 0
    tflite_ram_bytes: int = 0
    netkit_runs: list[float] = field(default_factory=list)
    tflite_runs: list[float] = field(default_factory=list)

    @property
    def speedup(self) -> float:
        if self.netkit_us <= 0:
            return float("nan")
        return self.tflite_us / self.netkit_us

    @property
    def flash_ratio(self) -> float:
        if self.netkit_flash_bytes <= 0:
            return float("nan")
        return self.tflite_flash_bytes / self.netkit_flash_bytes

    @property
    def ram_ratio(self) -> float:
        if self.netkit_ram_bytes <= 0:
            return float("nan")
        return self.tflite_ram_bytes / self.netkit_ram_bytes


FLOAT32 = DtypeProfile(name="float32", tag="f32", header_label="FLOAT32")
INT8 = DtypeProfile(name="int8", tag="i8", header_label="INT8")


def _xnn_tag(xnnpack: bool) -> str:
    return "xnn" if xnnpack else "ref"


def _parse_summary(text: str) -> dict[str, str]:
    matches = SUMMARY_RE.findall(text)
    if not matches:
        raise RuntimeError("no BENCHMARK_SUMMARY line in output")
    fields: dict[str, str] = {}
    for tok in matches[-1].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            fields[k] = v
    return fields


def _metric_from_summary(fields: dict[str, str], *, imagenet: bool) -> tuple[str, float]:
    key = "warm_mean_us" if imagenet else "mean_us"
    if key not in fields:
        raise RuntimeError(f"missing {key} in {fields}")
    return key, float(fields[key])


def _run(cmd: list[str], *, cwd: Path) -> tuple[str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - t0
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{out[-4000:]}"
        )
    return out, elapsed


def _binary_flash_ram(path: Path) -> tuple[int, int]:
    """MCU-style sizes for a runtime binary/lib: flash≈TEXT/text, ram≈DATA/(data+bss).

    Excludes models and other support files — only the linked runtime image.
    """
    if not path.is_file():
        return 0, 0
    proc = subprocess.run(
        ["size", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        # Fallback: whole file as flash proxy; ram unknown.
        return path.stat().st_size, 0

    # Darwin / Mach-O summary line:
    #   __TEXT  __DATA  __OBJC  others  dec  hex
    #   1409024 196608  0       ...
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if "__TEXT" in ln and "__DATA" in ln and i + 1 < len(lines):
            nums = lines[i + 1].split()
            if len(nums) >= 2 and nums[0].isdigit() and nums[1].isdigit():
                return int(nums[0]), int(nums[1])

    # GNU binutils:
    #   text  data  bss  dec  hex  filename
    for i, ln in enumerate(lines):
        if ln.lower().startswith("text") and "data" in ln.lower() and i + 1 < len(lines):
            nums = lines[i + 1].split()
            if len(nums) >= 3 and nums[0].isdigit():
                return int(nums[0]), int(nums[1]) + int(nums[2])

    return path.stat().st_size, 0


def _tflite_runtime_lib_paths() -> list[Path]:
    """Core LiteRT CPU interpreter libs (excludes Metal GPU and model files)."""
    site_roots = list((TFLITE / ".venv").glob("lib/python*/site-packages/ai_edge_litert"))
    if not site_roots:
        return []
    keep_names = {
        "libpywrap_litert_common.dylib",
        "libpywrap_litert_common.so",
        "libLiteRt.dylib",
        "libLiteRt.so",
        "_pywrap_litert_interpreter_wrapper.so",
        "_pywrap_litert_interpreter_wrapper.abi3.so",
    }
    found: list[Path] = []
    for root in site_roots:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.name in keep_names or (
                p.suffix in {".so", ".dylib"}
                and "interpreter_wrapper" in p.name
                and "Metal" not in p.name
            ):
                found.append(p)
    # Unique by resolve()
    uniq: dict[Path, Path] = {p.resolve(): p for p in found}
    return list(uniq.values())


def _netkit_fixture_image_objs(model: str, cfg: SuiteCfg) -> list[Path]:
    """Hard-coded bench fixture image .o files linked into the ELF (not production)."""
    gen = ROOT / "benchmark" / "tflm" / "generated"
    if cfg.dtype.name == "float32":
        return {
            "mlp": [gen / "mnist_test_images.o"],
            "cnn": [gen / "mnist_cnn_test_images.o"],
            "cnn_dw": [gen / "cnn_dw" / "mnist_cnn_test_images.o"],
            "imagenet": [gen / "imagenet_mnv4_test_images.o"],
        }[model]
    return {
        "mlp": [gen / "mnist_mlp_int8_test_images.o"],
        "cnn": [gen / "mnist_cnn_int8_test_images.o"],
        "cnn_dw": [gen / "cnn_dw" / "mnist_cnn_int8_test_images.o"],
        "imagenet": [gen / "imagenet_mnv4_netkit_int8_test_images.o"],
    }[model]


def _netkit_runtime_flash_ram(
    model: str, cfg: SuiteCfg, *, xnnpack: bool
) -> tuple[int, int]:
    """Bench ELF TEXT/DATA minus hard-coded test-image .o contribution."""
    flash, ram = _binary_flash_ram(_netkit_binary(model, cfg, xnnpack=xnnpack))
    for obj in _netkit_fixture_image_objs(model, cfg):
        of, ore = _binary_flash_ram(obj)
        flash = max(0, flash - of)
        ram = max(0, ram - ore)
    return flash, ram


def _tflite_runtime_flash_ram() -> tuple[int, int]:
    flash = 0
    ram = 0
    for lib in _tflite_runtime_lib_paths():
        f, r = _binary_flash_ram(lib)
        flash += f
        ram += r
    return flash, ram


def _ensure_venv() -> None:
    if not TFLITE_PY.is_file():
        _run(["make", "venv"], cwd=TFLITE)


def ensure_assets_float32() -> None:
    needed = [
        ROOT / "models" / "mnist_mlp.nk",
        ROOT / "models" / "mnist_cnn.nk",
        ROOT / "models" / "mnist_cnn_dw.nk",
        ROOT / "models" / "mobilenetv4_imagenet_f32.nk",
        ROOT / "benchmark" / "tflm" / "generated" / "mnist_mlp.tflite",
        ROOT / "benchmark" / "tflm" / "generated" / "mnist_cnn.tflite",
        ROOT / "benchmark" / "tflm" / "generated" / "mnist_cnn_dw.tflite",
        ROOT / "benchmark" / "tflm" / "generated" / "mobilenetv4_imagenet_f32.tflite",
        ROOT / "benchmark" / "tflm" / "generated" / "cnn_dw" / "mnist_cnn_test_images.cc",
    ]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        raise SystemExit("missing assets:\n  " + "\n  ".join(str(p) for p in missing))
    _ensure_venv()


def ensure_assets_int8() -> None:
    needed = [
        ROOT / "models" / "mnist_mlp_int8.nk",
        ROOT / "models" / "mnist_cnn_int8.nk",
        ROOT / "models" / "mnist_cnn_dw_int8.nk",
        ROOT / "models" / "mobilenetv4_imagenet_int8.nk",
        ROOT / "benchmark" / "tflm" / "generated" / "mnist_mlp_int8.tflite",
        ROOT / "benchmark" / "tflm" / "generated" / "mnist_cnn_int8.tflite",
        ROOT / "benchmark" / "tflm" / "generated" / "mnist_cnn_dw_int8.tflite",
        ROOT
        / "benchmark"
        / "tflm"
        / "generated"
        / "mnist_cnn_dw_int8_tflite_quant.json",
        ROOT / "benchmark" / "tflm" / "generated" / "mobilenetv4_imagenet_int8.tflite",
        ROOT
        / "benchmark"
        / "tflm"
        / "generated"
        / "cnn_dw"
        / "mnist_cnn_int8_test_images.cc",
    ]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        raise SystemExit("missing assets:\n  " + "\n  ".join(str(p) for p in missing))
    _ensure_venv()


def _make_common(cfg: SuiteCfg, *, xnnpack: bool) -> list[str]:
    x = 1 if xnnpack else 0
    xt = _xnn_tag(xnnpack)
    backend = "xnnpack" if xnnpack else "reference"
    return [
        "make",
        "-f",
        "bench.mk",
        f"BACKEND={backend}",
        "CMSIS_NN=0",
        f"XNNPACK={x}",
        "NETKIT_IM2COL=0",
        "NETKIT_LOOP_UNROLL=0",
        "BENCH_FLAG_PROFILE=tflite",
        f"BENCH_OBJDIR=bench_obj_ab_{cfg.suite_tag}_{xt}",
        f"BENCH_LIB=libnetkit_bench_ab_{cfg.suite_tag}_{xt}.a",
    ]


def _bench_name(model: str, cfg: SuiteCfg, *, xnnpack: bool) -> str:
    xt = _xnn_tag(xnnpack)
    if cfg.dtype.name == "float32":
        prefix = {
            "mlp": "mnist_mlp_bench",
            "cnn": "mnist_cnn_bench",
            "cnn_dw": "mnist_cnn_dw_bench",
            "imagenet": "mobilenetv4_imagenet_bench",
        }[model]
    else:
        prefix = {
            "mlp": "mnist_mlp_int8_bench",
            "cnn": "mnist_cnn_int8_bench",
            "cnn_dw": "mnist_cnn_dw_int8_bench",
            "imagenet": "mobilenetv4_imagenet_int8_bench",
        }[model]
    return f"{prefix}_ab_{cfg.suite_tag}_{xt}"


def _netkit_binary(model: str, cfg: SuiteCfg, *, xnnpack: bool) -> Path:
    return NETKIT / _bench_name(model, cfg, xnnpack=xnnpack)


def _netkit_model_path(model: str, cfg: SuiteCfg) -> str:
    if cfg.dtype.name == "float32":
        return {
            "mlp": "models/mnist_mlp.nk",
            "cnn": "models/mnist_cnn.nk",
            "cnn_dw": "models/mnist_cnn_dw.nk",
            "imagenet": "models/mobilenetv4_imagenet_f32.nk",
        }[model]
    return {
        "mlp": "models/mnist_mlp_int8.nk",
        "cnn": "models/mnist_cnn_int8.nk",
        "cnn_dw": "models/mnist_cnn_dw_int8.nk",
        "imagenet": "models/mobilenetv4_imagenet_int8.nk",
    }[model]


def _netkit_build_cmd(model: str, cfg: SuiteCfg, *, xnnpack: bool) -> list[str]:
    name = _bench_name(model, cfg, xnnpack=xnnpack)
    common = _make_common(cfg, xnnpack=xnnpack)
    tag = f"{cfg.suite_tag}_{_xnn_tag(xnnpack)}"

    if cfg.dtype.name == "float32":
        if model == "mlp":
            return common + [
                f"MLP_BENCH={name}",
                f"MLP_MAIN_OBJ=src/main_ab_{tag}.o",
                "build-mlp",
            ]
        if model == "cnn":
            return common + [
                f"CNN_BENCH={name}",
                f"CNN_MAIN_OBJ=src/mnist_cnn_main_ab_{tag}.o",
                "build-cnn",
            ]
        if model == "cnn_dw":
            return common + [
                "EXTRA_BENCH_INCLUDES=-I../tflm/generated/cnn_dw",
                f"CNN_BENCH={name}",
                f"CNN_MAIN_OBJ=src/mnist_cnn_dw_main_ab_{tag}.o",
                "CNN_IMAGES_CC=../tflm/generated/cnn_dw/mnist_cnn_test_images.cc",
                "CNN_IMAGES_OBJ=../tflm/generated/cnn_dw/mnist_cnn_test_images.o",
                "build-cnn",
            ]
        if model == "imagenet":
            return common + [
                f"MNV4_IMAGENET_BENCH={name}",
                f"MNV4_IMAGENET_MAIN_OBJ=src/mobilenetv4_imagenet_main_ab_{tag}.o",
                "build-mobilenetv4-imagenet",
            ]
    else:
        if model == "mlp":
            return common + [
                f"MLP_INT8_BENCH={name}",
                f"MLP_INT8_MAIN_OBJ=src/mnist_mlp_int8_main_ab_{tag}.o",
                "build-mlp-int8",
            ]
        if model == "cnn":
            return common + [
                f"CNN_INT8_BENCH={name}",
                f"CNN_INT8_MAIN_OBJ=src/mnist_cnn_int8_main_ab_{tag}.o",
                "build-cnn-int8",
            ]
        if model == "cnn_dw":
            return common + [
                "EXTRA_BENCH_INCLUDES=-I../tflm/generated/cnn_dw",
                f"CNN_INT8_BENCH={name}",
                f"CNN_INT8_MAIN_OBJ=src/mnist_cnn_dw_int8_main_ab_{tag}.o",
                "CNN_INT8_IMAGES_CC=../tflm/generated/cnn_dw/mnist_cnn_int8_test_images.cc",
                "CNN_INT8_IMAGES_OBJ=../tflm/generated/cnn_dw/mnist_cnn_int8_test_images.o",
                "build-cnn-int8",
            ]
        if model == "imagenet":
            return common + [
                f"MNV4_IMAGENET_INT8_BENCH={name}",
                f"MNV4_IMAGENET_INT8_MAIN_OBJ=src/mobilenetv4_imagenet_int8_main_ab_{tag}.o",
                "build-mobilenetv4-imagenet-int8",
            ]
    raise ValueError(model)


def _netkit_run_cmd(model: str, cfg: SuiteCfg, *, xnnpack: bool) -> list[str]:
    cmd = [
        str(_netkit_binary(model, cfg, xnnpack=xnnpack)),
        _netkit_model_path(model, cfg),
    ]
    if model == "mlp":
        cmd.extend([str(MLP_BATCH_INVOKES), str(MLP_BATCH_PASSES)])
    return cmd


def _tflite_cmd(model: str, cfg: SuiteCfg, *, xnnpack: bool) -> list[str]:
    py = str(TFLITE_PY)
    no_xnn = [] if xnnpack else ["--no-xnnpack"]
    if cfg.dtype.name == "float32":
        if model == "mlp":
            return [
                py,
                str(TFLITE / "mnist_mlp_bench.py"),
                "--num-threads",
                "1",
                "--batch-invokes",
                str(MLP_BATCH_INVOKES),
                "--runs",
                str(MLP_BATCH_PASSES),
                *no_xnn,
            ]
        if model == "cnn":
            return [
                py,
                str(TFLITE / "mnist_cnn_bench.py"),
                "--num-threads",
                "1",
                "--runs",
                str(CNN_RUNS),
                *no_xnn,
            ]
        if model == "cnn_dw":
            return [
                py,
                str(TFLITE / "mnist_cnn_bench.py"),
                "--num-threads",
                "1",
                "--runs",
                str(CNN_RUNS),
                "--model",
                str(ROOT / "benchmark" / "tflm" / "generated" / "mnist_cnn_dw.tflite"),
                "--nk",
                str(ROOT / "models" / "mnist_cnn_dw.nk"),
                *no_xnn,
            ]
        if model == "imagenet":
            return [
                py,
                str(TFLITE / "mobilenetv4_imagenet_bench.py"),
                "--num-threads",
                "1",
                *no_xnn,
            ]
    else:
        if model == "mlp":
            return [
                py,
                str(TFLITE / "mnist_mlp_int8_bench.py"),
                "--num-threads",
                "1",
                "--batch-invokes",
                str(MLP_BATCH_INVOKES),
                "--runs",
                str(MLP_BATCH_PASSES),
                *no_xnn,
            ]
        if model == "cnn":
            return [
                py,
                str(TFLITE / "mnist_cnn_int8_bench.py"),
                "--num-threads",
                "1",
                "--runs",
                str(CNN_RUNS),
                *no_xnn,
            ]
        if model == "cnn_dw":
            return [
                py,
                str(TFLITE / "mnist_cnn_int8_bench.py"),
                "--num-threads",
                "1",
                "--runs",
                str(CNN_RUNS),
                "--model",
                str(
                    ROOT / "benchmark" / "tflm" / "generated" / "mnist_cnn_dw_int8.tflite"
                ),
                "--nk",
                str(ROOT / "models" / "mnist_cnn_dw.nk"),
                "--quant-json",
                str(
                    ROOT
                    / "benchmark"
                    / "tflm"
                    / "generated"
                    / "mnist_cnn_dw_int8_tflite_quant.json"
                ),
                *no_xnn,
            ]
        if model == "imagenet":
            return [
                py,
                str(TFLITE / "mobilenetv4_imagenet_int8_bench.py"),
                "--num-threads",
                "1",
                *no_xnn,
            ]
    raise ValueError(model)


def _prebuild(models: list[str], cfg: SuiteCfg) -> None:
    print(
        f"\n======== PREBUILD (untimed, {cfg.dtype.header_label}) ========",
        flush=True,
    )
    for xnnpack in (True, False):
        mode = "XNNPACK ON" if xnnpack else "XNNPACK OFF"
        print(f"-- {mode} --", flush=True)
        for model in models:
            print(f"  build netkit {model} ...", flush=True)
            _run(_netkit_build_cmd(model, cfg, xnnpack=xnnpack), cwd=NETKIT)
            binary = _netkit_binary(model, cfg, xnnpack=xnnpack)
            if not binary.is_file():
                raise RuntimeError(f"missing binary after build: {binary}")
    print("prebuild done.", flush=True)


def _warmup_runtime(model: str, runtime: str, cfg: SuiteCfg, *, xnnpack: bool) -> None:
    if runtime == "netkit":
        _run(_netkit_run_cmd(model, cfg, xnnpack=xnnpack), cwd=ROOT)
    else:
        _run(_tflite_cmd(model, cfg, xnnpack=xnnpack), cwd=ROOT)


def _execute_runtime(
    model: str, runtime: str, cfg: SuiteCfg, *, xnnpack: bool
) -> tuple[str, float]:
    if runtime == "netkit":
        return _run(_netkit_run_cmd(model, cfg, xnnpack=xnnpack), cwd=ROOT)
    return _run(_tflite_cmd(model, cfg, xnnpack=xnnpack), cwd=ROOT)


def _one_runtime(
    model: str,
    runtime: str,
    cfg: SuiteCfg,
    *,
    xnnpack: bool,
    order: str,
    discard_first: bool,
) -> RunResult:
    imagenet = model == "imagenet"
    if discard_first:
        print(
            f"  discard-1st [{order}] {runtime:6s} {model:8s} "
            f"xnn={'ON' if xnnpack else 'OFF':3s} ...",
            flush=True,
        )
        _execute_runtime(model, runtime, cfg, xnnpack=xnnpack)

    out, elapsed = _execute_runtime(model, runtime, cfg, xnnpack=xnnpack)
    fields = _parse_summary(out)
    metric_name, metric_us = _metric_from_summary(fields, imagenet=imagenet)
    summary_line = SUMMARY_RE.findall(out)[-1]
    print(
        f"  [{order}] {runtime:6s} {model:8s} xnn={'ON' if xnnpack else 'OFF':3s} "
        f"{metric_name}={metric_us:.3f} us  ({elapsed:.1f}s)",
        flush=True,
    )
    return RunResult(
        runtime=runtime,
        model=model,
        xnnpack=xnnpack,
        order=order,
        metric_name=metric_name,
        metric_us=metric_us,
        raw_summary=summary_line,
        elapsed_s=elapsed,
    )


def run_suite(
    models: list[str], cfg: SuiteCfg, *, do_warmup: bool = True
) -> tuple[list[RunResult], list[PairAvg]]:
    _prebuild(models, cfg)

    results: list[RunResult] = []
    avgs: list[PairAvg] = []

    for xnnpack in (True, False):
        mode = "XNNPACK ON" if xnnpack else "XNNPACK OFF"
        print(
            f"\n======== MEASURE {mode} ({cfg.dtype.header_label}) ========",
            flush=True,
        )
        for model in models:
            print(f"\n-- model={model} --", flush=True)

            if do_warmup:
                print(
                    f"  prime opposite (tflite) before Pass A ({model}) ...",
                    flush=True,
                )
                _warmup_runtime(model, "tflite", cfg, xnnpack=xnnpack)
            a_nk = _one_runtime(
                model,
                "netkit",
                cfg,
                xnnpack=xnnpack,
                order="A:nk→tf",
                discard_first=do_warmup,
            )
            a_tf = _one_runtime(
                model,
                "tflite",
                cfg,
                xnnpack=xnnpack,
                order="A:nk→tf",
                discard_first=do_warmup,
            )

            if do_warmup:
                print(
                    f"  prime opposite (netkit) before Pass B ({model}) ...",
                    flush=True,
                )
                _warmup_runtime(model, "netkit", cfg, xnnpack=xnnpack)
            b_tf = _one_runtime(
                model,
                "tflite",
                cfg,
                xnnpack=xnnpack,
                order="B:tf→nk",
                discard_first=do_warmup,
            )
            b_nk = _one_runtime(
                model,
                "netkit",
                cfg,
                xnnpack=xnnpack,
                order="B:tf→nk",
                discard_first=do_warmup,
            )

            results.extend([a_nk, a_tf, b_tf, b_nk])

            nk_flash, nk_ram = _netkit_runtime_flash_ram(model, cfg, xnnpack=xnnpack)
            tf_flash, tf_ram = _tflite_runtime_flash_ram()
            pair = PairAvg(
                model=model,
                xnnpack=xnnpack,
                netkit_us=(a_nk.metric_us + b_nk.metric_us) / 2.0,
                tflite_us=(a_tf.metric_us + b_tf.metric_us) / 2.0,
                metric_name=a_nk.metric_name,
                netkit_flash_bytes=nk_flash,
                tflite_flash_bytes=tf_flash,
                netkit_ram_bytes=nk_ram,
                tflite_ram_bytes=tf_ram,
                netkit_runs=[a_nk.metric_us, b_nk.metric_us],
                tflite_runs=[a_tf.metric_us, b_tf.metric_us],
            )
            avgs.append(pair)
            spread_nk = abs(a_nk.metric_us - b_nk.metric_us)
            spread_tf = abs(a_tf.metric_us - b_tf.metric_us)
            print(
                f"  order-avg {model:8s} netkit={pair.netkit_us:.3f}  "
                f"tflite={pair.tflite_us:.3f}  speedup(TF÷nk)={pair.speedup:.3f}×  "
                f"|Δnk|={spread_nk:.3f} |Δtf|={spread_tf:.3f}",
                flush=True,
            )
            print(
                f"  footprint {model:8s} "
                f"flash nk={_fmt_bytes(pair.netkit_flash_bytes)} "
                f"tf={_fmt_bytes(pair.tflite_flash_bytes)} "
                f"(TF÷nk={pair.flash_ratio:.3f}×)  "
                f"ram nk={_fmt_bytes(pair.netkit_ram_bytes)} "
                f"tf={_fmt_bytes(pair.tflite_ram_bytes)} "
                f"(TF÷nk={pair.ram_ratio:.3f}×)",
                flush=True,
            )
    return results, avgs


def _fmt_us(us: float) -> str:
    if us >= 1000.0:
        return f"{us:10.1f} us ({us / 1000.0:7.3f} ms)"
    return f"{us:10.3f} us"


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "       n/a"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):7.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:7.1f} KiB"
    return f"{n:7d} B"


def print_report(
    avgs: list[PairAvg],
    results: list[RunResult],
    dtype: DtypeProfile,
) -> None:
    print("\n" + "=" * 78)
    print(f"HOST A/B COMPLETE RESULTS  ({dtype.header_label} testing, CPU)")
    print(
        "Prebuild + discard first process per timed slot (keep 2nd); "
        "order-averaged over 2 swaps"
    )
    print(
        f"MNIST MLP batch={MLP_BATCH_INVOKES} invokes x {MLP_BATCH_PASSES} passes "
        f"(both sides); CNN/DW runs={CNN_RUNS}"
    )
    print("NETKIT_IM2COL=0 (direct) for all netkit builds")
    print(
        "Latency is warm-only: MLP batches discard pass 0; "
        "CNN discards run 0 + image 0 each run; "
        "ImageNet warm_mean discards the full first image pass."
    )
    print(
        "Flash/RAM = MCU-style runtime image only "
        "(netkit: bench ELF TEXT/DATA minus hard-coded test-image .o; "
        "tflite: core LiteRT CPU libs TEXT/DATA). "
        "Models (`.nk` / `.tflite`) and fixture images are excluded."
    )
    print("Ratio = TF ÷ netkit (same as latency).")
    print("=" * 78)

    for xnnpack in (True, False):
        mode = "XNNPACK ON" if xnnpack else "XNNPACK OFF (reference)"
        print(f"\n### {mode}\n")
        print(
            f"{'model':12s} {'metric':14s} {'netkit':22s} {'tflite':22s} "
            f"{'TF÷nk':>10s}"
        )
        print("-" * 78)
        for p in avgs:
            if p.xnnpack != xnnpack:
                continue
            print(
                f"{p.model:12s} {p.metric_name:14s} "
                f"{_fmt_us(p.netkit_us):22s} {_fmt_us(p.tflite_us):22s} "
                f"{p.speedup:9.3f}×"
            )
            print(
                f"{'':12s} {'  (pass A/B)':14s} "
                f"{p.netkit_runs[0]:8.3f} / {p.netkit_runs[1]:8.3f}     "
                f"{p.tflite_runs[0]:8.3f} / {p.tflite_runs[1]:8.3f}"
            )
            print(
                f"{'':12s} {'flash':14s} "
                f"{_fmt_bytes(p.netkit_flash_bytes):22s} "
                f"{_fmt_bytes(p.tflite_flash_bytes):22s} "
                f"{p.flash_ratio:9.3f}×"
            )
            print(
                f"{'':12s} {'ram':14s} "
                f"{_fmt_bytes(p.netkit_ram_bytes):22s} "
                f"{_fmt_bytes(p.tflite_ram_bytes):22s} "
                f"{p.ram_ratio:9.3f}×"
            )
        print()

    print("### Raw BENCHMARK_SUMMARY lines\n")
    for r in results:
        print(
            f"xnn={'ON' if r.xnnpack else 'OFF'} order={r.order} "
            f"BENCHMARK_SUMMARY {r.raw_summary}"
        )


def default_results_path(dtype: DtypeProfile) -> Path:
    return ROOT / "benchmark" / f"host_ab_suite_results_{dtype.name}.txt"


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CHOICES),
        choices=list(MODEL_CHOICES),
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip discarded warmup (not recommended; for debugging only)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Results file (default: benchmark/host_ab_suite_results_<dtype>.txt)",
    )
    return parser


def main_for_dtype(
    dtype: DtypeProfile,
    ensure_assets: Callable[[], None],
    description: str,
) -> int:
    parser = build_arg_parser(description)
    args = parser.parse_args()

    print(
        f"\n######## HOST A/B SUITE — {dtype.header_label} TESTING ########\n",
        flush=True,
    )
    ensure_assets()

    cfg = SuiteCfg(dtype=dtype)
    results, avgs = run_suite(args.models, cfg, do_warmup=not args.skip_warmup)

    print_report(avgs, results, dtype)

    out_path = args.out or default_results_path(dtype)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(avgs, results, dtype)
    out_path.write_text(buf.getvalue())
    print(f"\nWrote {out_path}")
    return 0


def cli_entry(
    dtype: DtypeProfile,
    ensure_assets: Callable[[], None],
    description: str,
    **_kwargs,
) -> None:
    try:
        raise SystemExit(main_for_dtype(dtype, ensure_assets, description))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
