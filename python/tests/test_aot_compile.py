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
from netkit.reader import read_nk, read_test_suite
from netkit.reference_forward import forward_cnn, forward_mlp

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
LIB = ROOT / "libnetkit.a"

AOT_MODELS = [
    "test_mlp.nk",
    "cnn_4x4_single.nk",
]


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise unittest.SkipTest(f"{name} not found")


def _require_lib() -> None:
    if not LIB.is_file():
        raise unittest.SkipTest("libnetkit.a missing — run `make lib` first")


def _reference_output(arch: dict, weights: np.ndarray, flat_input: np.ndarray) -> np.ndarray:
    if arch["network"] == "mlp":
        return forward_mlp(flat_input, arch, weights)
    return forward_cnn(flat_input, arch, weights)


def _write_cpp_harness(tmp: Path, symbol: str, case_input: list[float]) -> Path:
    harness = tmp / "harness.cpp"
    values = ", ".join(f"{v:.8f}f" for v in case_input)
    harness.write_text(
        f"""#include "{symbol}_aot.hpp"
#include "arena_util.hpp"
#include <cstdio>

int main(void)
{{
    alignas(std::max_align_t) static unsigned char arena_mem[Arena::kDefaultCapacity];
    ArenaUtil::Scoped arena_scope(Arena::kDefaultCapacity, arena_mem);
    if (!arena_scope)
        return 1;

    netkit::aot::{symbol}::Model model;
    if (!model.load(arena_scope.Get()))
        return 1;

    float input[] = {{{values}}};
    float output[netkit::aot::{symbol}::kOutputElements] = {{}};
    if (!model.forward(arena_scope.Get(), input, output))
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
    harness.write_text(
        f"""#include "{symbol}_aot.h"
#include <stdio.h>
#include <stdalign.h>

int main(void)
{{
    alignas(max_align_t) static unsigned char arena_mem[NK_ARENA_DEFAULT_CAPACITY];
    nk_arena_t arena;
#if defined(NETKIT_ARENA_HEAP)
    if (nk_arena_init_heap(&arena, NK_ARENA_DEFAULT_CAPACITY) != NK_OK)
        return 1;
#else
    nk_arena_init(&arena, arena_mem, sizeof(arena_mem));
#endif

    nk_model_t model;
    if ({symbol}_aot_load(&arena, &model) != NK_OK)
        return 1;

    float input[] = {{{values}}};
    float output[{symbol.upper()}_AOT_OUTPUT_ELEMENTS] = {{0.0f}};
    uint32_t output_count = 0;
    if ({symbol}_aot_run(&model, &arena, input, output, &output_count) != NK_OK)
        return 1;

    for (uint32_t i = 0; i < output_count; ++i)
        printf(i ? ",%.6f" : "%.6f", output[i]);
    printf("\\n");

#if defined(NETKIT_ARENA_HEAP)
    nk_arena_destroy_heap(&arena);
#endif
    return 0;
}}
""",
        encoding="utf-8",
    )
    return harness


def _compile_and_run_cpp(tmp: Path, source: Path, harness: Path, binary: Path) -> str:
    subprocess.run(
        [
            "clang++",
            "-std=c++26",
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
            "clang++",
            "-std=c++26",
            "-Wall",
            "-Wextra",
            f"-I{ROOT / 'include'}",
            str(harness),
            str(tmp / "aot.o"),
            str(LIB),
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


if __name__ == "__main__":
    unittest.main()
