"""Build and run AOT-generated C/C++ sources against embedded .nk test cases."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from netkit.aot_compile import AotLanguage, compile_aot
from netkit.format import HEADER_BYTES, unpack_header, weight_payload_bytes
from netkit.reader import read_nk, read_test_suite
from netkit.reference_forward import forward_cnn, forward_mlp

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
LIB = ROOT / "libnetkit.a"
XNNPACK_LIB_DIR = ROOT / "third_party" / "XNNPACK" / "netkit_lib"

AOT_MODELS = [
    "test_mlp.nk",
    "cnn_4x4_single.nk",
    "mlp_hand.nk",
    "cnn_hand.nk",
]


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise unittest.SkipTest(f"{name} not found")


def _require_lib() -> None:
    if not LIB.is_file():
        raise unittest.SkipTest("libnetkit.a missing — run `make lib` first")


def _lib_needs_xnnpack() -> bool:
    """True when libnetkit.a was built with NETKIT_USE_XNNPACK=1."""
    try:
        proc = subprocess.run(
            ["nm", "-gU", str(LIB)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return "XnnpackKernel" in proc.stdout or "XnnpackQuant" in proc.stdout


def _xnnpack_link_args() -> list[str]:
    if not _lib_needs_xnnpack():
        return []
    xnn = XNNPACK_LIB_DIR / "libXNNPACK.a"
    if not xnn.is_file():
        raise unittest.SkipTest(
            "libnetkit.a needs XNNPACK but third_party/XNNPACK/netkit_lib is missing"
        )
    args = [f"-Wl,-force_load,{xnn}", f"-L{XNNPACK_LIB_DIR}"]
    for name in (
        "xnnpack-microkernels-prod",
        "kleidiai",
        "pthreadpool",
        "cpuinfo",
    ):
        if (XNNPACK_LIB_DIR / f"lib{name}.a").is_file():
            args.append(f"-l{name}")
    args.extend(["-lpthread", "-lc++"])
    return args


def _reference_output(arch: dict, weights: np.ndarray, flat_input: np.ndarray) -> np.ndarray:
    if arch["network"] == "mlp":
        return forward_mlp(flat_input, arch, weights)
    return forward_cnn(flat_input, arch, weights)


def _write_cpp_harness(tmp: Path, symbol: str, case_input: list[float]) -> Path:
    harness = tmp / "harness.cpp"
    values = ", ".join(f"{v:.8f}f" for v in case_input)
    harness.write_text(
        f"""#include "{symbol}_aot.hpp"
#include <cstdio>
#include <stdalign.h>

int main(void)
{{
    alignas(std::max_align_t) static unsigned char arena_mem[netkit::aot::{symbol}::kArenaBytesRecommended];
    Arena arena;
    if (!netkit::aot::{symbol}::InitArena(arena, arena_mem, sizeof(arena_mem)))
        return 1;

    netkit::aot::{symbol}::Model model;
    if (!model.load(arena))
        return 1;

    float input[] = {{{values}}};
    float output[netkit::aot::{symbol}::kOutputElements] = {{}};
    if (!model.forward(arena, input, output))
        return 1;

    for (std::uint32_t i = 0; i < netkit::aot::{symbol}::kOutputElements; ++i)
        std::printf(i ? ",%.6f" : "%.6f", output[i]);
    std::printf("\\n");
    return 0;
}}
""",
        encoding="utf-8",
    )
    return harness


def _write_c_harness(tmp: Path, symbol: str, case_input: list[float]) -> Path:
    harness = tmp / "harness.c"
    values = ", ".join(f"{v:.8f}f" for v in case_input)
    prefix = symbol.upper()
    harness.write_text(
        f"""#include "{symbol}_aot.h"
#include <stdio.h>
#include <stdalign.h>

int main(void)
{{
    alignas(max_align_t) static unsigned char arena_mem[{prefix}_AOT_ARENA_BYTES_RECOMMENDED];
    nk_arena_t arena;
    if ({symbol}_aot_init_arena(&arena, arena_mem, sizeof(arena_mem)) != NK_OK)
        return 1;

    nk_model_t model;
    if ({symbol}_aot_load(&arena, &model) != NK_OK)
        return 1;

    float input[] = {{{values}}};
    float output[{prefix}_AOT_OUTPUT_ELEMENTS] = {{0.0f}};
    uint32_t output_count = 0;
    if ({symbol}_aot_run(&model, &arena, input, output, &output_count) != NK_OK)
        return 1;

    for (uint32_t i = 0; i < output_count; ++i)
        printf(i ? ",%.6f" : "%.6f", output[i]);
    printf("\\n");
    return 0;
}}
""",
        encoding="utf-8",
    )
    return harness


def _compile_and_run_cpp(
    tmp: Path,
    source: Path,
    harness: Path,
    binary: Path,
    *,
    extra_cppflags: list[str] | None = None,
) -> str:
    cppflags = ["-std=c++26", "-Wall", "-Wextra", f"-I{ROOT / 'include'}"]
    if extra_cppflags:
        cppflags.extend(extra_cppflags)
    subprocess.run(
        ["clang++", *cppflags, "-c", str(source), "-o", str(tmp / "aot.o")],
        cwd=tmp,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "clang++",
            *cppflags,
            str(harness),
            str(tmp / "aot.o"),
            str(LIB),
            *_xnnpack_link_args(),
            "-o",
            str(binary),
        ],
        cwd=tmp,
        check=True,
        capture_output=True,
        text=True,
    )
    proc = subprocess.run([str(binary)], cwd=tmp, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def _compile_and_run_c(tmp: Path, source: Path, harness: Path, binary: Path) -> str:
    subprocess.run(
        [
            "clang",
            "-std=c23",
            "-Wall",
            "-Wextra",
            f"-I{ROOT / 'include'}",
            "-c",
            str(source),
            "-o",
            str(tmp / "aot.o"),
        ],
        cwd=tmp,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "clang",
            "-std=c23",
            "-Wall",
            "-Wextra",
            f"-I{ROOT / 'include'}",
            "-c",
            str(harness),
            "-o",
            str(tmp / "harness.o"),
        ],
        cwd=tmp,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "clang++",
            "-std=c++26",
            str(tmp / "aot.o"),
            str(tmp / "harness.o"),
            str(LIB),
            *_xnnpack_link_args(),
            "-o",
            str(binary),
        ],
        cwd=tmp,
        check=True,
        capture_output=True,
        text=True,
    )
    proc = subprocess.run([str(binary)], cwd=tmp, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


class TestAotCompile(unittest.TestCase):
    def setUp(self) -> None:
        _require_tool("clang")
        _require_tool("clang++")
        _require_lib()

    def test_compile_cpp_and_c_outputs_exist(self) -> None:
        nk_path = MODELS / "test_mlp.nk"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cpp = compile_aot(nk_path, tmp / "cpp", language=AotLanguage.CPP)
            c = compile_aot(nk_path, tmp / "c", language=AotLanguage.C)
            self.assertTrue(cpp.header_path.name.endswith("_aot.hpp"))
            self.assertTrue(cpp.source_path.name.endswith("_aot.cpp"))
            self.assertTrue(c.header_path.name.endswith("_aot.h"))
            self.assertTrue(c.source_path.name.endswith("_aot.c"))
            self.assertGreater(cpp.nk_bytes, 0)
            self.assertEqual(cpp.input_elements, 2)
            self.assertEqual(cpp.output_elements, 2)

    def test_arena_constants_in_generated_headers(self) -> None:
        nk_path = MODELS / "mlp_hand.nk"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cpp = compile_aot(nk_path, tmp / "cpp", language=AotLanguage.CPP)
            c = compile_aot(nk_path, tmp / "c", language=AotLanguage.C)

            cpp_header = cpp.header_path.read_text(encoding="utf-8")
            c_header = c.header_path.read_text(encoding="utf-8")

            self.assertIn(f"kArenaBytesAfterLoad = {cpp.arena_bytes_after_load}u", cpp_header)
            self.assertIn(f"kArenaBytesAfterForward = {cpp.arena_bytes_after_forward}u", cpp_header)
            self.assertIn(f"kArenaBytesRecommended = {cpp.arena_bytes_recommended}u", cpp_header)
            self.assertIn("InitArena(Arena& arena", cpp_header)

            prefix = cpp.model_name.upper()
            self.assertIn(f"{prefix}_AOT_ARENA_BYTES_AFTER_LOAD {c.arena_bytes_after_load}u", c_header)
            self.assertIn(
                f"{prefix}_AOT_ARENA_BYTES_AFTER_FORWARD {c.arena_bytes_after_forward}u", c_header
            )
            self.assertIn(f"{prefix}_AOT_ARENA_BYTES_RECOMMENDED {c.arena_bytes_recommended}u", c_header)
            self.assertIn(f"{cpp.model_name}_aot_init_arena", c_header)

            self.assertGreaterEqual(cpp.arena_bytes_recommended, cpp.arena_bytes_after_forward)
            self.assertGreater(cpp.arena_bytes_after_forward, 0)
            self.assertEqual(cpp.arena_bytes_recommended % 64, 0)

            cpp_source = cpp.source_path.read_text(encoding="utf-8")
            self.assertIn("NETKIT_AOT_FLASH_CONST", cpp_source)
            self.assertIn('section(".rodata")', cpp_source)
            self.assertIn("defined(__ELF__)", cpp_source)
            self.assertTrue(cpp.lowered)
            self.assertIn("kLowered = true", cpp_header)
            self.assertIn("Kernels::FullyConnectedWithBias", cpp_source)
            self.assertNotIn("LoadMLPFromBuffer", cpp_source)

    def test_aot_mcu_flash_arena_reports_payload(self) -> None:
        nk_path = MODELS / "mlp_hand.nk"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            flash = compile_aot(
                nk_path,
                tmp / "flash",
                language=AotLanguage.CPP,
                lower=False,
            )
            header = unpack_header(Path(nk_path).read_bytes()[:HEADER_BYTES])
            payload_bytes = weight_payload_bytes(header)
            self.assertGreater(payload_bytes, 0)
            self.assertGreater(flash.arena_bytes_after_forward, 0)
            self.assertGreaterEqual(flash.arena_bytes_recommended, flash.arena_bytes_after_forward)
            self.assertEqual(flash.arena_bytes_recommended % 64, 0)

    def test_aot_runtime_matches_reference(self) -> None:
        for model_file in AOT_MODELS:
            nk_path = MODELS / model_file
            arch, weights = read_nk(nk_path)
            suite = read_test_suite(nk_path)
            self.assertGreater(len(suite.cases), 0, msg=f"no TCAS cases in {model_file}")
            case = suite.cases[0]
            expected = _reference_output(arch, weights, np.asarray(case.input, dtype=np.float32))

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                for language, compile_fn, write_harness, run_fn in (
                    (
                        "cpp",
                        lambda out: compile_aot(nk_path, out, language=AotLanguage.CPP),
                        _write_cpp_harness,
                        _compile_and_run_cpp,
                    ),
                    (
                        "c",
                        lambda out: compile_aot(nk_path, out, language=AotLanguage.C),
                        _write_c_harness,
                        _compile_and_run_c,
                    ),
                ):
                    with self.subTest(model=model_file, language=language):
                        out_dir = tmp / language / model_file
                        result = compile_fn(out_dir)
                        harness = write_harness(out_dir, result.model_name, case.input)
                        stdout = run_fn(
                            out_dir,
                            result.source_path,
                            harness,
                            out_dir / "aot_runner",
                        )
                        actual = np.array([float(v) for v in stdout.split(",")], dtype=np.float32)
                        np.testing.assert_allclose(
                            actual,
                            expected,
                            rtol=0,
                            atol=suite.tolerance,
                            err_msg=f"AOT {language} mismatch for {model_file}",
                        )

    def test_aot_mcu_target_compiles_and_runs(self) -> None:
        nk_path = MODELS / "mlp_hand.nk"
        arch, weights = read_nk(nk_path)
        suite = read_test_suite(nk_path)
        case = suite.cases[0]
        expected = _reference_output(arch, weights, np.asarray(case.input, dtype=np.float32))

        subprocess.run(["make", "NETKIT_TARGET=mcu", "lib"], cwd=ROOT, check=True)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                out_dir = tmp / "mcu"
                result = compile_aot(nk_path, out_dir, language=AotLanguage.CPP)
                harness = _write_cpp_harness(out_dir, result.model_name, case.input)
                stdout = _compile_and_run_cpp(
                    out_dir,
                    result.source_path,
                    harness,
                    out_dir / "aot_runner",
                    extra_cppflags=["-DNETKIT_TARGET_MCU=1"],
                )
                actual = np.array([float(v) for v in stdout.split(",")], dtype=np.float32)
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=0,
                    atol=suite.tolerance,
                    err_msg="MCU-target AOT mismatch for mlp_hand.nk",
                )
        finally:
            subprocess.run(["make", "NETKIT_TARGET=cpu", "lib"], cwd=ROOT, check=True)

    def test_aot_embed_interpreter_path_still_available(self) -> None:
        nk_path = MODELS / "test_mlp.nk"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            result = compile_aot(nk_path, tmp / "embed", language=AotLanguage.CPP, lower=False)
            source = result.source_path.read_text(encoding="utf-8")
            self.assertFalse(result.lowered)
            self.assertIn("LoadMLPFromBuffer", source)
            self.assertIn("kNkBlob", source)

    def test_aot_optimize_matches_reference(self) -> None:
        nk_path = MODELS / "cnn_extended_ops.nk"
        arch, weights = read_nk(nk_path)
        suite = read_test_suite(nk_path)
        case = suite.cases[0]
        expected = _reference_output(arch, weights, np.asarray(case.input, dtype=np.float32))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_dir = tmp / "optimized"
            result = compile_aot(nk_path, out_dir, language=AotLanguage.CPP, optimize=True)
            self.assertTrue(result.optimized)
            self.assertIn("fold_batch_norm_into_dense", result.optimizations_applied)
            harness = _write_cpp_harness(out_dir, result.model_name, case.input)
            stdout = _compile_and_run_cpp(
                out_dir,
                result.source_path,
                harness,
                out_dir / "aot_runner",
            )
            actual = np.array([float(v) for v in stdout.split(",")], dtype=np.float32)
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=0,
                atol=suite.tolerance,
                err_msg="optimized AOT mismatch for cnn_extended_ops.nk",
            )


if __name__ == "__main__":
    unittest.main()
